/*
 * README-shape ACLNN calling example for ChunkGatedDeltaRule.
 *
 * External files follow ops-transformer README TND shapes:
 *   query/key:          [T, Nk, Dk], fp16/bf16 storage
 *   value/out:          [T, Nv, Dv], fp16/bf16 storage
 *   beta/g:             [T, Nv], float32 storage in this example
 *   actual_seq_lengths: [B], int32
 *   initial_state:      [B, Nv, Dv, Dk], fp16/bf16 storage
 *   final_state:        [B, Nv, Dv, Dk], fp16/bf16 storage
 *
 * The current gdr_aclnn::ChunkGatedDeltaRule helper still accepts BHT/BHTD.
 * This example adapts layouts at the calling layer and does not modify the
 * helper interface.
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    int64_t tokens = 0;
    int64_t keyHeads = 0;
    int64_t valueHeads = 0;
    int64_t keyDim = 0;
    int64_t valueDim = 0;
    int64_t chunkSize = 64;
    double scale = 1.0;
    std::string dtype = "bf16";
};

size_t TensorBytes(const std::vector<int64_t> &shape, aclDataType dtype)
{
    return static_cast<size_t>(gdr_aclnn::Numel(shape)) * gdr_aclnn::DTypeSize(dtype);
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

bool WriteFileExact(const std::filesystem::path &path, const std::vector<uint8_t> &data)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "failed to open output file: " << path << "\n";
        return false;
    }
    if (!data.empty()) {
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    if (!out) {
        std::cerr << "failed to write output file: " << path << "\n";
        return false;
    }
    return true;
}

bool ReadTensorBytes(const std::filesystem::path &path, const std::vector<int64_t> &shape, aclDataType dtype,
                     std::vector<uint8_t> &data)
{
    return ReadFileExact(path, TensorBytes(shape, dtype), data);
}

bool ReadInt32Tensor(const std::filesystem::path &path, int64_t count, std::vector<int32_t> &data)
{
    std::vector<uint8_t> bytes;
    if (!ReadFileExact(path, static_cast<size_t>(count) * sizeof(int32_t), bytes)) {
        return false;
    }
    data.resize(static_cast<size_t>(count));
    if (!bytes.empty()) {
        std::memcpy(data.data(), bytes.data(), bytes.size());
    }
    return true;
}

bool UploadTensor(gdr_aclnn::ManagedTensor &tensor, const std::vector<int64_t> &shape, aclDataType dtype,
                  const std::vector<uint8_t> &host)
{
    if (!tensor.Create(shape, dtype)) {
        return false;
    }
    if (host.size() != tensor.bytes) {
        std::cerr << "host tensor byte size mismatch, expected " << tensor.bytes << ", got " << host.size() << "\n";
        return false;
    }
    if (!host.empty()) {
        const auto ret = aclrtMemcpy(tensor.addr, tensor.bytes, host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy H2D failed: " << ret << "\n";
            return false;
        }
    }
    return true;
}

bool UploadInt32Tensor(gdr_aclnn::ManagedTensor &tensor, const std::vector<int32_t> &host)
{
    std::vector<uint8_t> bytes(host.size() * sizeof(int32_t));
    if (!host.empty()) {
        std::memcpy(bytes.data(), host.data(), bytes.size());
    }
    return UploadTensor(tensor, {static_cast<int64_t>(host.size())}, ACL_INT32, bytes);
}

bool DownloadTensor(const gdr_aclnn::ManagedTensor &tensor, std::vector<uint8_t> &host)
{
    host.assign(tensor.bytes, 0);
    if (tensor.bytes == 0) {
        return true;
    }
    const auto ret = aclrtMemcpy(host.data(), host.size(), tensor.addr, tensor.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtMemcpy D2H failed: " << ret << "\n";
        return false;
    }
    return true;
}

int64_t CountChunks(const std::vector<int32_t> &actualSeqLengths, int64_t chunkSize)
{
    int64_t chunks = 0;
    for (const auto len : actualSeqLengths) {
        chunks += (static_cast<int64_t>(len) + chunkSize - 1) / chunkSize;
    }
    return chunks;
}

bool ValidateParams(const Params &params, const std::vector<int32_t> &actualSeqLengths)
{
    if (params.inputDir.empty() || params.outputDir.empty()) {
        std::cerr << "--input-dir and --output-dir are required\n";
        return false;
    }
    if (params.batch <= 0 || params.tokens <= 0 || params.keyHeads <= 0 || params.valueHeads <= 0 ||
        params.keyDim <= 0 || params.valueDim <= 0 || params.chunkSize <= 0) {
        std::cerr << "batch/tokens/key-heads/value-heads/key-dim/value-dim/chunk-size must be positive\n";
        return false;
    }
    if (params.keyHeads != params.valueHeads) {
        std::cerr << "this adapter requires --key-heads == --value-heads because the current helper has one head "
                     "dimension\n";
        return false;
    }
    int64_t total = 0;
    for (const auto len : actualSeqLengths) {
        if (len <= 0) {
            std::cerr << "actual_seq_lengths entries must be positive\n";
            return false;
        }
        total += len;
    }
    if (total != params.tokens) {
        std::cerr << "sum(actual_seq_lengths) must equal --tokens, got " << total << " vs " << params.tokens << "\n";
        return false;
    }
    return true;
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
        } else if (arg == "--tokens") {
            params.tokens = std::stoll(requireValue("--tokens"));
        } else if (arg == "--key-heads") {
            params.keyHeads = std::stoll(requireValue("--key-heads"));
        } else if (arg == "--value-heads") {
            params.valueHeads = std::stoll(requireValue("--value-heads"));
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
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
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

struct PipelineTensors {
    gdr_aclnn::ManagedTensor queryTnd;
    gdr_aclnn::ManagedTensor keyTnd;
    gdr_aclnn::ManagedTensor valueTnd;
    gdr_aclnn::ManagedTensor betaTh;
    gdr_aclnn::ManagedTensor gTh;
    gdr_aclnn::ManagedTensor initialStateBhvK;
    gdr_aclnn::ManagedTensor actualSeqLengths;
    gdr_aclnn::ManagedTensor outTnd;
    gdr_aclnn::ManagedTensor finalStateBhvK;
    gdr_aclnn::ManagedTensor chunkStateExternal;

    void Destroy()
    {
        chunkStateExternal.Destroy();
        finalStateBhvK.Destroy();
        outTnd.Destroy();
        actualSeqLengths.Destroy();
        initialStateBhvK.Destroy();
        gTh.Destroy();
        betaTh.Destroy();
        valueTnd.Destroy();
        keyTnd.Destroy();
        queryTnd.Destroy();
    }
};

bool RunPipeline(const Params &params)
{
    const aclDataType inputDType = StorageDType(params.dtype);
    const int64_t heads = params.valueHeads;

    std::vector<int32_t> actualSeqLengths;
    if (!ReadInt32Tensor(params.inputDir / "actual_seq_lengths.bin", params.batch, actualSeqLengths) ||
        !ValidateParams(params, actualSeqLengths)) {
        return false;
    }
    const int64_t totalChunks = CountChunks(actualSeqLengths, params.chunkSize);

    std::vector<uint8_t> queryTnd;
    std::vector<uint8_t> keyTnd;
    std::vector<uint8_t> valueTnd;
    std::vector<uint8_t> betaTh;
    std::vector<uint8_t> gTh;
    std::vector<uint8_t> initialBhvK;
    bool ok = true;
    ok = ok && ReadTensorBytes(params.inputDir / "query.bin", {params.tokens, params.keyHeads, params.keyDim},
                               inputDType, queryTnd);
    ok = ok && ReadTensorBytes(params.inputDir / "key.bin", {params.tokens, params.keyHeads, params.keyDim},
                               inputDType, keyTnd);
    ok = ok && ReadTensorBytes(params.inputDir / "value.bin", {params.tokens, params.valueHeads, params.valueDim},
                               inputDType, valueTnd);
    ok = ok && ReadTensorBytes(params.inputDir / "beta.bin", {params.tokens, params.valueHeads}, ACL_FLOAT, betaTh);
    ok = ok && ReadTensorBytes(params.inputDir / "g.bin", {params.tokens, params.valueHeads}, ACL_FLOAT, gTh);
    ok = ok && ReadTensorBytes(params.inputDir / "initial_state.bin",
                               {params.batch, params.valueHeads, params.valueDim, params.keyDim}, inputDType,
                               initialBhvK);
    if (!ok) {
        return false;
    }

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(params.device, &context, &stream);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(params.device);
        aclFinalize();
        return false;
    }

    PipelineTensors tensors;
    try {
        ok = ok && UploadTensor(tensors.queryTnd, {params.tokens, heads, params.keyDim}, inputDType, queryTnd);
        ok = ok && UploadTensor(tensors.keyTnd, {params.tokens, heads, params.keyDim}, inputDType, keyTnd);
        ok = ok && UploadTensor(tensors.valueTnd, {params.tokens, heads, params.valueDim}, inputDType, valueTnd);
        ok = ok && UploadTensor(tensors.betaTh, {params.tokens, heads}, ACL_FLOAT, betaTh);
        ok = ok && UploadTensor(tensors.gTh, {params.tokens, heads}, ACL_FLOAT, gTh);
        ok = ok && UploadTensor(tensors.initialStateBhvK, {params.batch, heads, params.valueDim, params.keyDim},
                                inputDType, initialBhvK);
        ok = ok && UploadInt32Tensor(tensors.actualSeqLengths, actualSeqLengths);

        ok = ok && tensors.outTnd.Create({params.tokens, heads, params.valueDim}, inputDType);
        ok = ok && tensors.finalStateBhvK.Create({params.batch, heads, params.valueDim, params.keyDim}, inputDType);
        ok = ok && tensors.chunkStateExternal.Create({totalChunks, heads, params.valueDim, params.keyDim}, inputDType);
        if (!ok) {
            throw std::runtime_error("failed to prepare device tensors");
        }

        ok = ok && gdr_aclnn::ChunkGatedDeltaRule(
                       tensors.queryTnd.tensor, tensors.keyTnd.tensor, tensors.valueTnd.tensor, tensors.betaTh.tensor,
                       tensors.initialStateBhvK.tensor, tensors.actualSeqLengths.tensor, tensors.gTh.tensor,
                       static_cast<float>(params.scale), params.chunkSize, tensors.outTnd.tensor,
                       tensors.finalStateBhvK.tensor, tensors.chunkStateExternal.tensor, stream);

        if (ok) {
            std::vector<uint8_t> outTndBytes;
            std::vector<uint8_t> finalStateBytes;
            std::vector<uint8_t> chunkStateBytes;
            ok = ok && DownloadTensor(tensors.outTnd, outTndBytes);
            ok = ok && DownloadTensor(tensors.finalStateBhvK, finalStateBytes);
            ok = ok && DownloadTensor(tensors.chunkStateExternal, chunkStateBytes);
            if (!ok) {
                throw std::runtime_error("failed to download output tensors");
            }
            std::filesystem::create_directories(params.outputDir);
            ok = ok && WriteFileExact(params.outputDir / "out.bin", outTndBytes);
            ok = ok && WriteFileExact(params.outputDir / "final_state.bin", finalStateBytes);
            ok = ok && WriteFileExact(params.outputDir / "chunkState.bin", chunkStateBytes);
        }
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

    return ok;
}

}  // namespace

int main(int argc, char **argv)
{
    Params params;
    try {
        if (!ParseArgs(argc, argv, params)) {
            return 2;
        }
        if (!RunPipeline(params)) {
            return 1;
        }
        std::cout << "chunk_gated_delta_rule_tnd_example ok\n";
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "argument error: " << exc.what() << "\n";
        return 2;
    }
}
