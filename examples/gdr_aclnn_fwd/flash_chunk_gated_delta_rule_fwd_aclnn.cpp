/*
 * Standalone ACLNN driver for flash_chunk_gated_delta_rule_fwd.
 *
 * This program keeps the external ACLNN driver contract in BHT/BHTD layout:
 *   q/k:  [B,H,T,K], fp16/bf16 storage
 *   v:    [B,H,T,V], fp16/bf16 storage
 *   g:    [B,H,T], float32
 *   beta: [B,H,T], float32
 *   o:    [B,H,T,V], fp16/bf16 storage
 *   A:    [B,H,T,chunk_size], fp16/bf16 storage after solve_tri
 *
 * solve_tri is the only step that uses a temporary layout: dense cases use
 * BSND [B,T,H,BT], and varlen cases use a TND view [T,H,BT], then convert back.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_chunk_fwd_o.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule_fwd_h.h"
#include "aclnnop/aclnn_chunk_local_cumsum.h"
#include "aclnnop/aclnn_chunk_scaled_dot_kkt.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_recompute_wu_fwd.h"

extern "C" {
__attribute__((visibility("default"))) aclnnStatus aclnnSolveTriGetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSolveTri(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
}

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
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
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

std::vector<int64_t> BuildChunkIndices(const std::vector<int64_t> &cuSeqlens, int64_t chunkSize)
{
    std::vector<int64_t> result;
    for (size_t seq = 0; seq + 1 < cuSeqlens.size(); ++seq) {
        const int64_t len = cuSeqlens[seq + 1] - cuSeqlens[seq];
        for (int64_t chunk = 0; chunk < (len + chunkSize - 1) / chunkSize; ++chunk) {
            result.push_back(static_cast<int64_t>(seq));
            result.push_back(chunk);
        }
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
        params.chunkIndices = BuildChunkIndices(params.cuSeqlens, params.chunkSize);
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

template <typename GetWorkspaceFn, typename LaunchFn>
bool RunOp(const char *name, GetWorkspaceFn getWorkspace, LaunchFn launch, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = getWorkspace(&workspaceSize, &executor);
    if (status != ACL_SUCCESS) {
        std::cerr << name << "GetWorkspaceSize failed: " << status << "\n";
        return false;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "workspace aclrtMalloc failed for " << name << ": " << ret << "\n";
            return false;
        }
    }

    status = launch(workspace, workspaceSize, executor, stream);
    if (status != ACL_SUCCESS) {
        std::cerr << name << " failed: " << status << "\n";
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        return false;
    }

    auto ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSynchronizeStream failed after " << name << ": " << ret << "\n";
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        return false;
    }

    if (workspace != nullptr) {
        ret = aclrtFree(workspace);
        if (ret != ACL_SUCCESS) {
            std::cerr << "workspace aclrtFree failed for " << name << ": " << ret << "\n";
            return false;
        }
    }
    return true;
}

class IntArrayHolder {
public:
    explicit IntArrayHolder(const std::vector<int64_t> &values)
    {
        if (!values.empty()) {
            array_ = aclCreateIntArray(values.data(), values.size());
            if (array_ == nullptr) {
                throw std::runtime_error("aclCreateIntArray failed");
            }
        }
    }

    ~IntArrayHolder()
    {
        if (array_ != nullptr) {
            aclDestroyIntArray(array_);
        }
    }

    const aclIntArray *get() const
    {
        return array_;
    }

private:
    aclIntArray *array_ = nullptr;
};

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

int Run(const Params &params)
{
    const aclDataType inputDType = StorageDType(params.dtype);
    const bool varlen = !params.cuSeqlens.empty();
    const int64_t totalChunks = varlen ? static_cast<int64_t>(params.chunkIndices.size() / 2)
                                       : (params.tokens + params.chunkSize - 1) / params.chunkSize;
    const int64_t stateCount = varlen ? static_cast<int64_t>(params.cuSeqlens.size() - 1) : params.batch;

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(params.device, &context, &stream);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(params.device);
        aclFinalize();
        return 1;
    }

    bool ok = true;
    DeviceTensor q;
    DeviceTensor k;
    DeviceTensor v;
    DeviceTensor gIn;
    DeviceTensor beta;
    DeviceTensor gCum;
    DeviceTensor aKktFloat;
    DeviceTensor aKktCastBhtd;
    DeviceTensor aSolveInStorage;
    DeviceTensor aSolveOutStorage;
    DeviceTensor aSolveInView;
    DeviceTensor aSolveOutView;
    DeviceTensor aSolveBhtd;
    DeviceTensor w;
    DeviceTensor u;
    DeviceTensor h;
    DeviceTensor vNew;
    DeviceTensor finalState;
    DeviceTensor o;

    try {
        const std::vector<int64_t> qShape = {params.batch, params.heads, params.tokens, params.keyDim};
        const std::vector<int64_t> vShape = {params.batch, params.heads, params.tokens, params.valueDim};
        const std::vector<int64_t> gateShape = {params.batch, params.heads, params.tokens};
        const std::vector<int64_t> aShape = {params.batch, params.heads, params.tokens, params.chunkSize};
        const std::vector<int64_t> aSolveStorageShape = {params.batch, params.tokens, params.heads, params.chunkSize};
        const std::vector<int64_t> aSolveViewShape = {params.tokens, params.heads, params.chunkSize};
        const std::vector<int64_t> solvePerm = {0, 2, 1, 3};
        const std::vector<int64_t> hShape = {params.batch, params.heads, totalChunks, params.keyDim, params.valueDim};
        const std::vector<int64_t> finalShape = {stateCount, params.heads, params.keyDim, params.valueDim};

        ok = ok && ReadTensor(params.inputDir / "q.bin", qShape, inputDType, q);
        ok = ok && ReadTensor(params.inputDir / "k.bin", qShape, inputDType, k);
        ok = ok && ReadTensor(params.inputDir / "v.bin", vShape, inputDType, v);
        ok = ok && ReadTensor(params.inputDir / "g.bin", gateShape, ACL_FLOAT, gIn);
        ok = ok && ReadTensor(params.inputDir / "beta.bin", gateShape, ACL_FLOAT, beta);
        if (!ok) {
            throw std::runtime_error("failed to read input tensors");
        }

        ok = ok && gCum.Create(nullptr, gateShape, ACL_FLOAT);
        ok = ok && aKktFloat.Create(nullptr, aShape, ACL_FLOAT);
        if (!ok) {
            throw std::runtime_error("failed to allocate cumsum/KKT outputs");
        }

        IntArrayHolder cuArray(params.cuSeqlens);
        IntArrayHolder chunkArray(params.chunkIndices);
        char outputDtype[] = "float32";

        ok = ok && RunOp(
            "aclnnChunkLocalCumsum",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnChunkLocalCumsumGetWorkspaceSize(
                    gIn.tensor, cuArray.get(), chunkArray.get(), params.chunkSize, false, 1.0, true, outputDtype,
                    gCum.tensor, workspaceSize, executor);
            },
            aclnnChunkLocalCumsum, stream);

        ok = ok && RunOp(
            "aclnnChunkScaledDotKkt",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnChunkScaledDotKktGetWorkspaceSize(
                    k.tensor, gCum.tensor, beta.tensor, cuArray.get(), chunkArray.get(), params.chunkSize,
                    aKktFloat.tensor, workspaceSize, executor);
            },
            aclnnChunkScaledDotKkt, stream);
        if (!ok) {
            throw std::runtime_error("cumsum/KKT op failed");
        }

        std::filesystem::create_directories(params.outputDir);
        ok = ok && aKktCastBhtd.Create(nullptr, aShape, inputDType);
        ok = ok && aSolveInStorage.Create(nullptr, aSolveStorageShape, inputDType);
        ok = ok && aSolveOutStorage.Create(nullptr, aSolveStorageShape, inputDType);
        ok = ok && aSolveBhtd.Create(nullptr, aShape, inputDType);
        if (varlen) {
            ok = ok && aSolveInView.CreateView(aSolveInStorage.addr, aSolveViewShape, inputDType);
            ok = ok && aSolveOutView.CreateView(aSolveOutStorage.addr, aSolveViewShape, inputDType);
        }
        if (!ok) {
            throw std::runtime_error("failed to prepare solve_tri tensors");
        }

        ok = ok && RunOp(
            "aclnnCast(A_kkt)",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnCastGetWorkspaceSize(
                    aKktFloat.tensor, inputDType, aKktCastBhtd.tensor, workspaceSize, executor);
            },
            aclnnCast, stream);
        if (!ok) {
            throw std::runtime_error("A KKT cast op failed");
        }

        IntArrayHolder solvePermArray(solvePerm);
        ok = ok && RunOp(
            "aclnnPermute(A_to_solve)",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnPermuteGetWorkspaceSize(
                    aKktCastBhtd.tensor, solvePermArray.get(), aSolveInStorage.tensor, workspaceSize, executor);
            },
            aclnnPermute, stream);
        if (!ok) {
            throw std::runtime_error("A to solve layout transpose op failed");
        }

        aclTensor *solveInTensor = varlen ? aSolveInView.tensor : aSolveInStorage.tensor;
        aclTensor *solveOutTensor = varlen ? aSolveOutView.tensor : aSolveOutStorage.tensor;
        const char *solveLayout = varlen ? "tnd" : "bsnd";
        ok = ok && RunOp(
            "aclnnSolveTri",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnSolveTriGetWorkspaceSize(
                    solveInTensor, cuArray.get(), chunkArray.get(), solveLayout, solveOutTensor, workspaceSize,
                    executor);
            },
            aclnnSolveTri, stream);
        if (!ok) {
            throw std::runtime_error("solve_tri op failed");
        }

        ok = ok && RunOp(
            "aclnnPermute(A_to_bhtd)",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnPermuteGetWorkspaceSize(
                    aSolveOutStorage.tensor, solvePermArray.get(), aSolveBhtd.tensor, workspaceSize, executor);
            },
            aclnnPermute, stream);
        if (!ok) {
            throw std::runtime_error("failed to convert solve_tri output back to BHTD");
        }

        ok = ok && w.Create(nullptr, qShape, inputDType);
        ok = ok && u.Create(nullptr, vShape, inputDType);
        if (!ok) {
            throw std::runtime_error("failed to prepare recompute tensors");
        }

        ok = ok && RunOp(
            "aclnnRecomputeWUFwd",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnRecomputeWUFwdGetWorkspaceSize(
                    k.tensor, v.tensor, beta.tensor, aSolveBhtd.tensor, gCum.tensor, nullptr, cuArray.get(),
                    chunkArray.get(), params.chunkSize, w.tensor, u.tensor, workspaceSize, executor);
            },
            aclnnRecomputeWUFwd, stream);
        if (!ok) {
            throw std::runtime_error("recompute_w_u op failed");
        }

        ok = ok && h.Create(nullptr, hShape, inputDType);
        ok = ok && vNew.Create(nullptr, vShape, inputDType);
        if (params.outputFinalState) {
            ok = ok && finalState.Create(nullptr, finalShape, ACL_FLOAT);
        } else {
            ok = ok && finalState.Create(nullptr, {1}, inputDType);
        }
        if (!ok) {
            throw std::runtime_error("failed to prepare fwd_h tensors");
        }

        ok = ok && RunOp(
            "aclnnChunkGatedDeltaRuleFwdH",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
                    k.tensor, w.tensor, u.tensor, gCum.tensor, nullptr, nullptr, params.outputFinalState,
                    params.chunkSize, true, cuArray.get(), chunkArray.get(), false, false, h.tensor, vNew.tensor,
                    finalState.tensor, workspaceSize, executor);
            },
            aclnnChunkGatedDeltaRuleFwdH, stream);
        if (!ok) {
            throw std::runtime_error("fwd_h op failed");
        }

        ok = ok && o.Create(nullptr, vShape, inputDType);
        ok = ok && RunOp(
            "aclnnChunkFwdO",
            [&](uint64_t *workspaceSize, aclOpExecutor **executor) {
                return aclnnChunkFwdOGetWorkspaceSize(
                    q.tensor, k.tensor, vNew.tensor, h.tensor, gCum.tensor, cuArray.get(), chunkArray.get(),
                    params.scale, params.chunkSize, o.tensor, workspaceSize, executor);
            },
            aclnnChunkFwdO, stream);
        if (!ok) {
            throw std::runtime_error("fwd_o op failed");
        }

        ok = ok && WriteTensorFile(params.outputDir / "g.bin", gCum);
        ok = ok && WriteTensorFile(params.outputDir / "o.bin", o);
        ok = ok && WriteTensorFile(params.outputDir / "A.bin", aSolveBhtd);
        if (params.outputFinalState) {
            ok = ok && WriteTensorFile(params.outputDir / "final_state.bin", finalState);
        }
    } catch (const std::exception &exc) {
        std::cerr << "driver failed: " << exc.what() << "\n";
        ok = false;
    }

    o.Destroy();
    finalState.Destroy();
    vNew.Destroy();
    h.Destroy();
    u.Destroy();
    w.Destroy();
    aSolveBhtd.Destroy();
    aSolveOutView.Destroy();
    aSolveInView.Destroy();
    aSolveOutStorage.Destroy();
    aSolveInStorage.Destroy();
    aKktCastBhtd.Destroy();
    aKktFloat.Destroy();
    gCum.Destroy();
    beta.Destroy();
    gIn.Destroy();
    v.Destroy();
    k.Destroy();
    q.Destroy();

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
