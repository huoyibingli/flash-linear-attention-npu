#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
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

namespace gdr_aclnn {

inline std::vector<int64_t> BuildChunkIndices(const std::vector<int64_t> &cuSeqlens, int64_t chunkSize)
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

inline int64_t NextPowerOfTwo(int64_t value)
{
    int64_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

inline int64_t CumsumBlockT(int64_t chunkSize)
{
    return NextPowerOfTwo((int64_t{1} << 17) / chunkSize);
}

inline int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t result = 1;
    for (const auto dim : shape) {
        result *= dim;
    }
    return shape.empty() ? 0 : result;
}

inline std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

inline size_t DTypeSize(aclDataType dtype)
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
    if (dtype == ACL_INT64) {
        return sizeof(int64_t);
    }
    throw std::runtime_error("unsupported dtype size");
}

inline aclFormat FormatForDim(size_t dim)
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

inline bool GetTensorShape(const aclTensor *tensor, std::vector<int64_t> &shape)
{
    int64_t *dims = nullptr;
    uint64_t dimsNum = 0;
    const auto status = aclGetViewShape(tensor, &dims, &dimsNum);
    if (status != ACL_SUCCESS) {
        std::cerr << "aclGetViewShape failed: " << status << "\n";
        return false;
    }
    shape.assign(dims, dims + dimsNum);
    return true;
}

inline bool GetTensorDType(const aclTensor *tensor, aclDataType &dtype)
{
    const auto status = aclGetDataType(tensor, &dtype);
    if (status != ACL_SUCCESS) {
        std::cerr << "aclGetDataType failed: " << status << "\n";
        return false;
    }
    return true;
}

class ManagedTensor {
public:
    ~ManagedTensor()
    {
        Destroy();
    }

    bool Create(const std::vector<int64_t> &tensorShape, aclDataType tensorDType)
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
            ret = aclrtMemset(addr, bytes, 0, bytes);
            if (ret != ACL_SUCCESS) {
                std::cerr << "aclrtMemset failed: " << ret << "\n";
                return false;
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

    void *addr = nullptr;
    aclTensor *tensor = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> storageShape;
    aclDataType dtype = ACL_FLOAT;
    size_t bytes = 0;
    bool ownsAddr = true;
};

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

inline bool ReadActualSeqLengths(const aclTensor *actualSeqLengths, std::vector<int64_t> &lengths)
{
    lengths.clear();
    if (actualSeqLengths == nullptr) {
        return true;
    }

    std::vector<int64_t> shape;
    if (!GetTensorShape(actualSeqLengths, shape)) {
        return false;
    }
    if (shape.size() != 1) {
        std::cerr << "actualSeqLengths expects rank 1\n";
        return false;
    }

    aclDataType dtype = ACL_INT32;
    if (!GetTensorDType(actualSeqLengths, dtype)) {
        return false;
    }

    void *deviceAddr = nullptr;
    auto status = aclGetRawTensorAddr(actualSeqLengths, &deviceAddr);
    if (status != ACL_SUCCESS || deviceAddr == nullptr) {
        std::cerr << "aclGetRawTensorAddr(actualSeqLengths) failed: " << status << "\n";
        return false;
    }

    const int64_t count = Numel(shape);
    lengths.resize(static_cast<size_t>(count));
    if (dtype == ACL_INT32) {
        std::vector<int32_t> host(static_cast<size_t>(count), 0);
        auto ret = aclrtMemcpy(host.data(), host.size() * sizeof(int32_t), deviceAddr,
                               host.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy actualSeqLengths D2H failed: " << ret << "\n";
            return false;
        }
        for (int64_t i = 0; i < count; ++i) {
            lengths[static_cast<size_t>(i)] = host[static_cast<size_t>(i)];
        }
    } else if (dtype == ACL_INT64) {
        auto ret = aclrtMemcpy(lengths.data(), lengths.size() * sizeof(int64_t), deviceAddr,
                               lengths.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy actualSeqLengths D2H failed: " << ret << "\n";
            return false;
        }
    } else {
        std::cerr << "actualSeqLengths expects int32 or int64 dtype\n";
        return false;
    }

    return true;
}

inline std::vector<int64_t> BuildCuSeqlens(const std::vector<int64_t> &actualSeqLengths)
{
    if (actualSeqLengths.empty()) {
        return {};
    }
    std::vector<int64_t> cuSeqlens;
    cuSeqlens.reserve(actualSeqLengths.size() + 1);
    cuSeqlens.push_back(0);
    for (const auto len : actualSeqLengths) {
        cuSeqlens.push_back(cuSeqlens.back() + len);
    }
    return cuSeqlens;
}

inline int64_t CountChunks(const std::vector<int64_t> &lengths, int64_t totalTokens, int64_t chunkSize)
{
    if (chunkSize <= 0) {
        throw std::runtime_error("chunkSize must be positive");
    }
    if (lengths.empty()) {
        return (totalTokens + chunkSize - 1) / chunkSize;
    }
    int64_t chunks = 0;
    for (const auto len : lengths) {
        chunks += (len + chunkSize - 1) / chunkSize;
    }
    return chunks;
}

inline bool CheckChunkStateShape(const std::vector<int64_t> &chunkStateShape, const std::vector<int64_t> &lengths,
                                 int64_t totalTokens, int64_t chunkSize)
{
    try {
        const int64_t expectedChunks = CountChunks(lengths, totalTokens, chunkSize);
        if (chunkStateShape[2] != expectedChunks) {
            std::cerr << "chunkState total_chunks mismatch, expected " << expectedChunks << ", got "
                      << chunkStateShape[2] << "\n";
            return false;
        }
    } catch (const std::exception &exc) {
        std::cerr << exc.what() << "\n";
        return false;
    }
    return true;
}

inline bool ChunkGatedDeltaRuleImpl(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const std::vector<int64_t> &cuSeqlens,
    const std::vector<int64_t> &chunkIndices,
    int64_t chunkSize,
    double scale,
    const aclTensor *initialState,
    bool outputFinalState,
    aclDataType inputDType,
    aclTensor *gOut,
    aclTensor *aKktFloat,
    aclTensor *aKktCastBhtd,
    aclTensor *aSolveInStorage,
    aclTensor *aSolveOutStorage,
    aclTensor *aSolveInView,
    aclTensor *aSolveOutView,
    aclTensor *a,
    aclTensor *w,
    aclTensor *u,
    aclTensor *chunkState,
    aclTensor *vNew,
    aclTensor *finalState,
    aclTensor *o,
    aclrtStream stream)
{
    // Mirrors Python flash_chunk_gated_delta_rule_fwd:
    // cumsum -> KKT -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o.
    const bool varlen = !cuSeqlens.empty();
    if (varlen && (aSolveInView == nullptr || aSolveOutView == nullptr)) {
        std::cerr << "varlen mode requires explicit solve_tri TND view tensors\n";
        return false;
    }

    const std::vector<int64_t> cumsumChunkIndices =
        varlen ? BuildChunkIndices(cuSeqlens, CumsumBlockT(chunkSize)) : std::vector<int64_t>{};
    const std::vector<int64_t> solvePerm = {0, 2, 1, 3};
    IntArrayHolder cuArray(cuSeqlens);
    IntArrayHolder chunkArray(chunkIndices);
    IntArrayHolder cumsumChunkArray(cumsumChunkIndices);
    IntArrayHolder solvePermArray(solvePerm);
    char outputDtype[] = "float32";

    // 1. Accumulate gate values inside each chunk: g[B,H,T] -> gOut[B,H,T] in fp32.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnChunkLocalCumsumGetWorkspaceSize(
            g, cuArray.get(), cumsumChunkArray.get(), chunkSize, false, 1.0, true, outputDtype, gOut,
            &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkLocalCumsumGetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnChunkLocalCumsum: " << ret << "\n";
                return false;
            }
        }

        status = aclnnChunkLocalCumsum(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkLocalCumsum failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnChunkLocalCumsum: " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnChunkLocalCumsum: " << ret << "\n";
                return false;
            }
        }
    }

    // 2. Build KKT matrix in fp32 for numerical stability.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnChunkScaledDotKktGetWorkspaceSize(
            k, gOut, beta, cuArray.get(), chunkArray.get(), chunkSize, aKktFloat, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkScaledDotKktGetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnChunkScaledDotKkt: " << ret << "\n";
                return false;
            }
        }

        status = aclnnChunkScaledDotKkt(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkScaledDotKkt failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnChunkScaledDotKkt: " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnChunkScaledDotKkt: " << ret << "\n";
                return false;
            }
        }
    }

    // 3. Cast KKT back to the input storage dtype before solve_tri.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status =
            aclnnCastGetWorkspaceSize(aKktFloat, inputDType, aKktCastBhtd, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnCast(A_kkt)GetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnCast(A_kkt): " << ret << "\n";
                return false;
            }
        }

        status = aclnnCast(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnCast(A_kkt) failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnCast(A_kkt): " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnCast(A_kkt): " << ret << "\n";
                return false;
            }
        }
    }

    // 4. Convert A from BHTD storage to solve_tri's expected BSND/TND memory order.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnPermuteGetWorkspaceSize(
            aKktCastBhtd, solvePermArray.get(), aSolveInStorage, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnPermute(A_to_solve)GetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnPermute(A_to_solve): " << ret << "\n";
                return false;
            }
        }

        status = aclnnPermute(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnPermute(A_to_solve) failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnPermute(A_to_solve): " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnPermute(A_to_solve): " << ret << "\n";
                return false;
            }
        }
    }

    const aclTensor *solveInTensor = varlen ? aSolveInView : aSolveInStorage;
    const aclTensor *solveOutTensor = varlen ? aSolveOutView : aSolveOutStorage;
    const char *solveLayout = varlen ? "tnd" : "bsnd";

    // 5. Solve the per-chunk triangular system.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnSolveTriGetWorkspaceSize(
            solveInTensor, cuArray.get(), chunkArray.get(), solveLayout, solveOutTensor, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnSolveTriGetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnSolveTri: " << ret << "\n";
                return false;
            }
        }

        status = aclnnSolveTri(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnSolveTri failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnSolveTri: " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnSolveTri: " << ret << "\n";
                return false;
            }
        }
    }

    // 6. Convert solved A back to the external BHTD layout.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status =
            aclnnPermuteGetWorkspaceSize(aSolveOutStorage, solvePermArray.get(), a, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnPermute(A_to_bhtd)GetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnPermute(A_to_bhtd): " << ret << "\n";
                return false;
            }
        }

        status = aclnnPermute(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnPermute(A_to_bhtd) failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnPermute(A_to_bhtd): " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnPermute(A_to_bhtd): " << ret << "\n";
                return false;
            }
        }
    }

    // 7. Recompute W and U from k, v, beta, solved A, and cumulative gates.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnRecomputeWUFwdGetWorkspaceSize(
            k, v, beta, a, gOut, nullptr, cuArray.get(), chunkArray.get(), chunkSize, w, u, &workspaceSize,
            &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnRecomputeWUFwdGetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnRecomputeWUFwd: " << ret << "\n";
                return false;
            }
        }

        status = aclnnRecomputeWUFwd(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnRecomputeWUFwd failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnRecomputeWUFwd: " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnRecomputeWUFwd: " << ret << "\n";
                return false;
            }
        }
    }

    // 8. Run the chunk recurrent state update and expose chunkState plus vNew.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
            k, w, u, gOut, nullptr, initialState, outputFinalState, chunkSize, true, cuArray.get(), chunkArray.get(),
            false, false, chunkState, vNew, finalState, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnChunkGatedDeltaRuleFwdH: " << ret << "\n";
                return false;
            }
        }

        status = aclnnChunkGatedDeltaRuleFwdH(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkGatedDeltaRuleFwdH failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnChunkGatedDeltaRuleFwdH: " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnChunkGatedDeltaRuleFwdH: " << ret << "\n";
                return false;
            }
        }
    }

    // 9. Compute final attention output o from q, k, vNew, h, and cumulative gates.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnChunkFwdOGetWorkspaceSize(
            q, k, vNew, chunkState, gOut, cuArray.get(), chunkArray.get(), scale, chunkSize, o, &workspaceSize,
            &executor);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkFwdOGetWorkspaceSize failed: " << status << "\n";
            return false;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtMalloc failed for aclnnChunkFwdO: " << ret << "\n";
                return false;
            }
        }

        status = aclnnChunkFwdO(workspace, workspaceSize, executor, stream);
        if (status != ACL_SUCCESS) {
            std::cerr << "aclnnChunkFwdO failed: " << status << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }

        auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSynchronizeStream failed after aclnnChunkFwdO: " << ret << "\n";
            if (workspace != nullptr) {
                aclrtFree(workspace);
            }
            return false;
        }
        if (workspace != nullptr) {
            ret = aclrtFree(workspace);
            if (ret != ACL_SUCCESS) {
                std::cerr << "workspace aclrtFree failed for aclnnChunkFwdO: " << ret << "\n";
                return false;
            }
        }
    }

    return true;
}

inline bool ChunkGatedDeltaRule(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    const aclTensor *initialState,
    const aclTensor *actualSeqLengths,
    const aclTensor *gOptional,
    float scaleValue,
    int64_t chunkSize,
    aclTensor *out,
    aclTensor *finalState,
    aclTensor *chunkState,
    aclrtStream stream)
{
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr || out == nullptr ||
        chunkState == nullptr) {
        std::cerr << "query/key/value/beta/out/chunkState must not be null\n";
        return false;
    }

    std::vector<int64_t> queryShape;
    std::vector<int64_t> valueShape;
    std::vector<int64_t> chunkStateShape;
    std::vector<int64_t> finalStateShape;
    if (!GetTensorShape(query, queryShape) || !GetTensorShape(value, valueShape) ||
        !GetTensorShape(chunkState, chunkStateShape)) {
        return false;
    }
    if (queryShape.size() != 4 || valueShape.size() != 4 || chunkStateShape.size() != 5) {
        std::cerr << "current helper expects BHT/BHTD tensors and B,H,chunk,K,V chunkState\n";
        return false;
    }

    aclDataType inputDType = ACL_FLOAT16;
    if (!GetTensorDType(query, inputDType)) {
        return false;
    }

    std::vector<int64_t> actualLengths;
    if (!ReadActualSeqLengths(actualSeqLengths, actualLengths)) {
        return false;
    }
    const std::vector<int64_t> cuSeqlens = BuildCuSeqlens(actualLengths);
    const bool varlen = !cuSeqlens.empty();
    const int64_t batch = queryShape[0];
    const int64_t heads = queryShape[1];
    const int64_t tokens = queryShape[2];
    const int64_t keyDim = queryShape[3];
    const int64_t valueDim = valueShape[3];
    if (!CheckChunkStateShape(chunkStateShape, actualLengths, tokens, chunkSize)) {
        return false;
    }
    const std::vector<int64_t> chunkIndices = varlen ? BuildChunkIndices(cuSeqlens, chunkSize) : std::vector<int64_t>{};

    const std::vector<int64_t> gateShape = {batch, heads, tokens};
    const std::vector<int64_t> aShape = {batch, heads, tokens, chunkSize};
    const std::vector<int64_t> aSolveStorageShape = {batch, tokens, heads, chunkSize};
    const std::vector<int64_t> aSolveViewShape = {tokens, heads, chunkSize};

    ManagedTensor zeroG;
    const aclTensor *g = gOptional;
    if (g == nullptr) {
        if (!zeroG.Create(gateShape, ACL_FLOAT)) {
            return false;
        }
        g = zeroG.tensor;
    }

    ManagedTensor gOut;
    ManagedTensor aKktFloat;
    ManagedTensor aKktCastBhtd;
    ManagedTensor aSolveInStorage;
    ManagedTensor aSolveOutStorage;
    ManagedTensor aSolveInView;
    ManagedTensor aSolveOutView;
    ManagedTensor a;
    ManagedTensor w;
    ManagedTensor u;
    ManagedTensor vNew;
    if (!gOut.Create(gateShape, ACL_FLOAT) || !aKktFloat.Create(aShape, ACL_FLOAT) ||
        !aKktCastBhtd.Create(aShape, inputDType) || !aSolveInStorage.Create(aSolveStorageShape, inputDType) ||
        !aSolveOutStorage.Create(aSolveStorageShape, inputDType) || !a.Create(aShape, inputDType) ||
        !w.Create(queryShape, inputDType) || !u.Create(valueShape, inputDType) ||
        !vNew.Create(valueShape, inputDType)) {
        return false;
    }
    if (varlen) {
        if (!aSolveInView.CreateView(aSolveInStorage.addr, aSolveViewShape, inputDType) ||
            !aSolveOutView.CreateView(aSolveOutStorage.addr, aSolveViewShape, inputDType)) {
            return false;
        }
    }

    std::vector<int64_t> finalShape;
    bool outputFinalState = false;
    if (finalState != nullptr && GetTensorShape(finalState, finalShape)) {
        outputFinalState = finalShape.size() == 4;
    }

    return ChunkGatedDeltaRuleImpl(
        query, key, value, g, beta, cuSeqlens, chunkIndices, chunkSize, scaleValue, initialState, outputFinalState,
        inputDType,
        gOut.tensor, aKktFloat.tensor, aKktCastBhtd.tensor, aSolveInStorage.tensor, aSolveOutStorage.tensor,
        varlen ? aSolveInView.tensor : nullptr, varlen ? aSolveOutView.tensor : nullptr, a.tensor, w.tensor, u.tensor,
        chunkState, vNew.tensor, finalState, out, stream);
}

}  // namespace gdr_aclnn
