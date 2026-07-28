/*
 * Standalone ACLNN driver for flash_chunk_gated_delta_rule_fwd.
 *
 * This program keeps the external ACLNN driver contract in BHT/BHTD layout:
 *   q/k:  [B,H,T,K], fp16/bf16 storage
 *   v:    [B,H,T,V], fp16/bf16 storage
 *   g:    [B,H,T], float32
 *   beta: [B,H,T], float32
 *   o:    [B,H,T,V], fp16/bf16 storage
 *   chunkState: [B,H,total_chunks,K,V], fp16/bf16 chunk state from fwd_h
 *
 * solve_tri is the only step that uses a temporary layout: dense cases use
 * BSND [B,T,H,BT], and varlen cases use a TND view [T,H,BT], then convert back.
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "chunk_gated_delta_rule_aclnn.h"

namespace {

struct Params {
    int32_t device = 0;
    std::filesystem::path inputDir;
    std::filesystem::path outputDir;
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t tokens = 0;
    int64_t keyDim = 0;
    int64_t valueDim = 0;
    int64_t chunkSize = 64;
    double scale = 1.0;
    std::string dtype = "fp16";
    bool outputFinalState = false;
    std::vector<int64_t> cuSeqlens;
    std::vector<int64_t> chunkIndices;
};

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t result = 1;
    for (const auto dim : shape) {
        result *= dim;
    }
    return shape.empty() ? 0 : result;
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

size_t DTypeSize(aclDataType dtype)
{
    if (dtype == ACL_FLOAT) {
        return sizeof(float);
    }
    if (dtype == ACL_FLOAT16 || dtype == ACL_BF16) {
        return sizeof(uint16_t);
    }
    if (dtype == ACL_INT32) {
        return sizeof(int32_t);
    }
    throw std::runtime_error("unsupported dtype size");
}

aclFormat FormatForDim(size_t dim)
{
    if (dim == 3) {
        return aclFormat::ACL_FORMAT_NCL;
    }
    if (dim == 4) {
        return aclFormat::ACL_FORMAT_NCHW;
    }
    if (dim == 5) {
        return aclFormat::ACL_FORMAT_NCDHW;
    }
    return aclFormat::ACL_FORMAT_ND;
}

aclDataType StorageDType(const std::string &name)
{
    if (name == "fp16" || name == "float16" || name == "half") {
        return ACL_FLOAT16;
    }
    if (name == "bf16" || name == "bfloat16") {
        return ACL_BF16;
    }
    throw std::runtime_error("dtype must be fp16 or bf16");
}

bool ReadFileExact(const std::filesystem::path &path, size_t bytes, std::vector<uint8_t> &data)
{
    data.assign(bytes, 0);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "failed to open input file: " << path << "\n";
        return false;
    }
    if (bytes != 0) {
        in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
    }
    if (in.gcount() != static_cast<std::streamsize>(bytes)) {
        std::cerr << "input file has unexpected size: " << path << ", expected " << bytes
                  << " bytes, read " << in.gcount() << " bytes\n";
        return false;
    }
    char extra = 0;
    if (in.read(&extra, 1)) {
        std::cerr << "input file has extra bytes: " << path << "\n";
        return false;
    }
    return true;
}

bool WriteFileExact(const std::filesystem::path &path, const void *data, size_t bytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open output file: " << path << "\n";
        return false;
    }
    if (bytes != 0) {
        out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    }
    if (!out) {
        std::cerr << "failed to write output file: " << path << "\n";
        return false;
    }
    return true;
}

std::vector<int64_t> ParseIntList(const std::string &text)
{
    std::vector<int64_t> result;
    if (text.empty()) {
        return result;
    }
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(std::stoll(item));
        }
    }
    return result;
}

std::vector<int32_t> BuildActualSeqLengths(const std::vector<int64_t> &cuSeqlens)
{
    std::vector<int32_t> result;
    for (size_t i = 0; i + 1 < cuSeqlens.size(); ++i) {
        result.push_back(static_cast<int32_t>(cuSeqlens[i + 1] - cuSeqlens[i]));
    }
    return result;
}

bool ParseBool(const std::string &value)
{
    return value == "1" || value == "true" || value == "True" || value == "yes";
}

bool ParseArgs(int argc, char **argv, Params &params)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto requireValue = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--device") {
            params.device = static_cast<int32_t>(std::stoi(requireValue("--device")));
        } else if (arg == "--input-dir") {
            params.inputDir = requireValue("--input-dir");
        } else if (arg == "--output-dir") {
            params.outputDir = requireValue("--output-dir");
        } else if (arg == "--batch") {
            params.batch = std::stoll(requireValue("--batch"));
        } else if (arg == "--heads") {
            params.heads = std::stoll(requireValue("--heads"));
        } else if (arg == "--tokens") {
            params.tokens = std::stoll(requireValue("--tokens"));
        } else if (arg == "--key-dim") {
            params.keyDim = std::stoll(requireValue("--key-dim"));
        } else if (arg == "--value-dim") {
            params.valueDim = std::stoll(requireValue("--value-dim"));
        } else if (arg == "--chunk-size") {
            params.chunkSize = std::stoll(requireValue("--chunk-size"));
        } else if (arg == "--scale") {
            params.scale = std::stod(requireValue("--scale"));
        } else if (arg == "--dtype") {
            params.dtype = requireValue("--dtype");
        } else if (arg == "--cu-seqlens") {
            params.cuSeqlens = ParseIntList(requireValue("--cu-seqlens"));
        } else if (arg == "--chunk-indices") {
            params.chunkIndices = ParseIntList(requireValue("--chunk-indices"));
        } else if (arg == "--output-final-state") {
            params.outputFinalState = ParseBool(requireValue("--output-final-state"));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (params.inputDir.empty() || params.outputDir.empty()) {
        std::cerr << "--input-dir and --output-dir are required\n";
        return false;
    }
    if (params.batch <= 0 || params.heads <= 0 || params.tokens <= 0 || params.keyDim <= 0 ||
        params.valueDim <= 0 || params.chunkSize <= 0) {
        std::cerr << "batch/heads/tokens/key-dim/value-dim/chunk-size must be positive\n";
        return false;
    }
    if (!params.cuSeqlens.empty() && params.batch != 1) {
        std::cerr << "varlen mode expects --batch 1\n";
        return false;
    }
    if (!params.cuSeqlens.empty() && params.chunkIndices.empty()) {
        params.chunkIndices = gdr_aclnn::BuildChunkIndices(params.cuSeqlens, params.chunkSize);
    }
    return true;
}

struct DeviceTensor {
    void *addr = nullptr;
    aclTensor *tensor = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> storageShape;
    aclDataType dtype = ACL_FLOAT;
    size_t bytes = 0;
    bool ownsAddr = true;

    bool Create(const void *hostData, const std::vector<int64_t> &tensorShape, aclDataType tensorDType)
    {
        shape = tensorShape;
        dtype = tensorDType;
        bytes = static_cast<size_t>(Numel(shape)) * DTypeSize(dtype);
        ownsAddr = true;
        if (bytes > 0) {
            auto ret = aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMalloc failed: " << ret << "\n";
                return false;
            }
            if (hostData != nullptr) {
                ret = aclrtMemcpy(addr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
                if (ret != ACL_SUCCESS) {
                    std::cerr << "aclrtMemcpy H2D failed: " << ret << "\n";
                    return false;
                }
            } else {
                ret = aclrtMemset(addr, bytes, 0, bytes);
                if (ret != ACL_SUCCESS) {
                    std::cerr << "aclrtMemset failed: " << ret << "\n";
                    return false;
                }
            }
        }

        strides = ContiguousStrides(shape);
        storageShape = {Numel(shape)};
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, FormatForDim(shape.size()),
                                 storageShape.data(), storageShape.size(), addr);
        if (tensor == nullptr) {
            std::cerr << "aclCreateTensor failed\n";
            return false;
        }
        return true;
    }

    bool CreateView(void *deviceAddr, const std::vector<int64_t> &tensorShape, aclDataType tensorDType)
    {
        shape = tensorShape;
        dtype = tensorDType;
        bytes = static_cast<size_t>(Numel(shape)) * DTypeSize(dtype);
        addr = deviceAddr;
        ownsAddr = false;
        strides = ContiguousStrides(shape);
        storageShape = {Numel(shape)};
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, FormatForDim(shape.size()),
                                 storageShape.data(), storageShape.size(), addr);
        if (tensor == nullptr) {
            std::cerr << "aclCreateTensor view failed\n";
            return false;
        }
        return true;
    }

    void Destroy()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (ownsAddr && addr != nullptr) {
            aclrtFree(addr);
        }
        addr = nullptr;
        ownsAddr = true;
    }
};

bool ReadTensor(const std::filesystem::path &path, const std::vector<int64_t> &shape, aclDataType dtype,
                DeviceTensor &tensor)
{
    std::vector<uint8_t> bytes;
    const size_t expected = static_cast<size_t>(Numel(shape)) * DTypeSize(dtype);
    if (!ReadFileExact(path, expected, bytes)) {
        return false;
    }
    return tensor.Create(bytes.data(), shape, dtype);
}

bool CopyDeviceToBytes(const DeviceTensor &tensor, std::vector<uint8_t> &bytes)
{
    bytes.assign(tensor.bytes, 0);
    if (tensor.bytes == 0) {
        return true;
    }
    auto ret = aclrtMemcpy(bytes.data(), tensor.bytes, tensor.addr, tensor.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtMemcpy D2H failed: " << ret << "\n";
        return false;
    }
    return true;
}

int InitAcl(int32_t device, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed: " << ret << "\n";
        return ret;
    }
    ret = aclrtSetDevice(device);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed: " << ret << "\n";
        return ret;
    }
    ret = aclrtCreateContext(context, device);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateContext failed: " << ret << "\n";
        return ret;
    }
    ret = aclrtSetCurrentContext(*context);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetCurrentContext failed: " << ret << "\n";
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed: " << ret << "\n";
    }
    return ret;
}

bool WriteTensorFile(const std::filesystem::path &path, const DeviceTensor &tensor)
{
    std::vector<uint8_t> bytes;
    if (!CopyDeviceToBytes(tensor, bytes)) {
        return false;
    }
    return WriteFileExact(path, bytes.data(), bytes.size());
}

struct PipelineShapes {
    std::vector<int64_t> qShape;
    std::vector<int64_t> vShape;
    std::vector<int64_t> gateShape;
    std::vector<int64_t> actualSeqLengthsShape;
    std::vector<int64_t> initialStateShape;
    std::vector<int64_t> chunkStateShape;
    std::vector<int64_t> finalShape;
};

PipelineShapes MakePipelineShapes(const Params &params)
{
    const bool varlen = !params.cuSeqlens.empty();
    const int64_t totalChunks = varlen ? static_cast<int64_t>(params.chunkIndices.size() / 2)
                                       : (params.tokens + params.chunkSize - 1) / params.chunkSize;
    const int64_t stateCount = varlen ? static_cast<int64_t>(params.cuSeqlens.size() - 1) : params.batch;
    return {
        {params.batch, params.heads, params.tokens, params.keyDim},
        {params.batch, params.heads, params.tokens, params.valueDim},
        {params.batch, params.heads, params.tokens},
        {stateCount},
        {stateCount, params.heads, params.keyDim, params.valueDim},
        {params.batch, params.heads, totalChunks, params.keyDim, params.valueDim},
        {stateCount, params.heads, params.keyDim, params.valueDim},
    };
}

struct PipelineTensors {
    DeviceTensor q;
    DeviceTensor k;
    DeviceTensor v;
    DeviceTensor gIn;
    DeviceTensor beta;
    DeviceTensor initialState;
    DeviceTensor actualSeqLengths;
    DeviceTensor chunkState;
    DeviceTensor finalState;
    DeviceTensor o;

    void Destroy()
    {
        o.Destroy();
        finalState.Destroy();
        chunkState.Destroy();
        actualSeqLengths.Destroy();
        initialState.Destroy();
        beta.Destroy();
        gIn.Destroy();
        v.Destroy();
        k.Destroy();
        q.Destroy();
    }
};

bool PrepareChunkGatedDeltaRuleTensors(const Params &params, const PipelineShapes &shapes, aclDataType inputDType,
                                       PipelineTensors &tensors)
{
    const bool varlen = !params.cuSeqlens.empty();
    bool ok = true;
    ok = ok && ReadTensor(params.inputDir / "q.bin", shapes.qShape, inputDType, tensors.q);
    ok = ok && ReadTensor(params.inputDir / "k.bin", shapes.qShape, inputDType, tensors.k);
    ok = ok && ReadTensor(params.inputDir / "v.bin", shapes.vShape, inputDType, tensors.v);
    ok = ok && ReadTensor(params.inputDir / "g.bin", shapes.gateShape, ACL_FLOAT, tensors.gIn);
    ok = ok && ReadTensor(params.inputDir / "beta.bin", shapes.gateShape, ACL_FLOAT, tensors.beta);
    if (!ok) {
        std::cerr << "failed to read input tensors\n";
        return false;
    }

    if (varlen) {
        const std::vector<int32_t> lengths = BuildActualSeqLengths(params.cuSeqlens);
        ok = ok && tensors.actualSeqLengths.Create(lengths.data(), shapes.actualSeqLengthsShape, ACL_INT32);
    }
    ok = ok && tensors.initialState.Create(nullptr, shapes.initialStateShape, inputDType);
    ok = ok && tensors.chunkState.Create(nullptr, shapes.chunkStateShape, inputDType);
    if (params.outputFinalState) {
        ok = ok && tensors.finalState.Create(nullptr, shapes.finalShape, ACL_FLOAT);
    } else {
        ok = ok && tensors.finalState.Create(nullptr, {1}, inputDType);
    }
    ok = ok && tensors.o.Create(nullptr, shapes.vShape, inputDType);
    if (!ok) {
        std::cerr << "failed to prepare chunk gated delta rule tensors\n";
    }
    return ok;
}

bool RunChunkGatedDeltaRule(const Params &params, aclDataType inputDType, PipelineTensors &tensors,
                            aclrtStream stream)
{
    (void)inputDType;
    const aclTensor *actualSeqLengths = params.cuSeqlens.empty() ? nullptr : tensors.actualSeqLengths.tensor;
    return gdr_aclnn::ChunkGatedDeltaRule(
        tensors.q.tensor,
        tensors.k.tensor,
        tensors.v.tensor,
        tensors.beta.tensor,
        tensors.initialState.tensor,
        actualSeqLengths,
        tensors.gIn.tensor,
        static_cast<float>(params.scale),
        params.chunkSize,
        tensors.o.tensor,
        tensors.finalState.tensor,
        tensors.chunkState.tensor,
        stream);
}

bool WriteChunkGatedDeltaRuleOutputs(const Params &params, const PipelineTensors &tensors)
{
    std::filesystem::create_directories(params.outputDir);
    bool ok = true;
    ok = ok && WriteTensorFile(params.outputDir / "o.bin", tensors.o);
    ok = ok && WriteTensorFile(params.outputDir / "chunkState.bin", tensors.chunkState);
    if (params.outputFinalState) {
        ok = ok && WriteTensorFile(params.outputDir / "final_state.bin", tensors.finalState);
    }
    return ok;
}

int RunPipeline(const Params &params)
{
    const aclDataType inputDType = StorageDType(params.dtype);
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(params.device, &context, &stream);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(params.device);
        aclFinalize();
        return 1;
    }

    bool ok = true;
    PipelineTensors tensors;

    try {
        const PipelineShapes shapes = MakePipelineShapes(params);
        ok = ok && PrepareChunkGatedDeltaRuleTensors(params, shapes, inputDType, tensors);
        ok = ok && RunChunkGatedDeltaRule(params, inputDType, tensors, stream);
        ok = ok && WriteChunkGatedDeltaRuleOutputs(params, tensors);
    } catch (const std::exception &exc) {
        std::cerr << "driver failed: " << exc.what() << "\n";
        ok = false;
    }

    tensors.Destroy();

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(params.device);
    aclFinalize();

    if (!ok) {
        return 1;
    }
    std::cout << "flash_chunk_gated_delta_rule_fwd_aclnn ok\n";
    return 0;
}

int Run(const Params &params)
{
    return RunPipeline(params);
}

}  // namespace

int main(int argc, char **argv)
{
    Params params;
    try {
        if (!ParseArgs(argc, argv, params)) {
            return 2;
        }
        return Run(params);
    } catch (const std::exception &exc) {
        std::cerr << "argument error: " << exc.what() << "\n";
        return 2;
    }
}
