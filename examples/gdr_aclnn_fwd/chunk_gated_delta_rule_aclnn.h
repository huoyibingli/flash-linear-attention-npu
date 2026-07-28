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

inline bool ChunkGatedDeltaRule(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const std::vector<int64_t> &cuSeqlens,
    const std::vector<int64_t> &chunkIndices,
    int64_t chunkSize,
    double scale,
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
    aclTensor *h,
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

    // 8. Run the chunk recurrent state update and expose h plus vNew.
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto status = aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
            k, w, u, gOut, nullptr, nullptr, outputFinalState, chunkSize, true, cuArray.get(), chunkArray.get(),
            false, false, h, vNew, finalState, &workspaceSize, &executor);
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
            q, k, vNew, h, gOut, cuArray.get(), chunkArray.get(), scale, chunkSize, o, &workspaceSize, &executor);
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

}  // namespace gdr_aclnn
