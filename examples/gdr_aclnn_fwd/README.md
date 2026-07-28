# ChunkGatedDeltaRule ACLNN Example Interface

## 功能说明

`gdr_aclnn::ChunkGatedDeltaRule` 用于在 C++ 示例中完成 chunk 版 Gated Delta Rule 前向计算。接口参数命名参考 `ops-transformer/attention/chunk_gated_delta_rule/README.md` 的业务参数，但内部实现保持当前 example 的子算子拼接流程：

```text
ChunkLocalCumsum -> ChunkScaledDotKkt -> Cast -> Permute -> SolveTri
-> Permute -> RecomputeWUFwd -> ChunkGatedDeltaRuleFwdH -> ChunkFwdO
```

当前接口仍使用 example 约定的 BHT/BHTD 输入布局，不是 README 中融合算子的 TND 布局。

## 接口原型

```cpp
bool gdr_aclnn::ChunkGatedDeltaRule(
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
    aclrtStream stream);
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `query` | 输入 | GDR 公式中的 q。当前 shape 为 `[B,H,T,K]`，dtype 为 fp16/bf16。 |
| `key` | 输入 | GDR 公式中的 k。当前 shape 为 `[B,H,T,K]`，dtype 与 `query` 一致。 |
| `value` | 输入 | GDR 公式中的 v。当前 shape 为 `[B,H,T,V]`，dtype 与 `query` 一致。 |
| `beta` | 输入 | 更新强度 beta。当前 shape 为 `[B,H,T]`，dtype 为 fp32。 |
| `initialState` | 输入 | 初始状态矩阵，传入 `ChunkGatedDeltaRuleFwdH`。当前 shape 为 `[stateCount,H,K,V]`。 |
| `actualSeqLengths` | 输入 | 每条序列的有效长度，shape 为 `[B]`，dtype 支持 int32/int64。dense 场景传 `nullptr`。 |
| `gOptional` | 输入 | 衰减系数 g，shape 为 `[B,H,T]`，dtype 为 fp32。传 `nullptr` 时内部使用全 0 g。 |
| `scaleValue` | 属性 | query 缩放因子。 |
| `chunkSize` | 属性 | chunk 大小，必须显式传入。varlen 场景会用它从 `actualSeqLengths` 生成 `cuSeqlens` 和 `chunkIndices`。 |
| `out` | 输出 | attention 输出，当前 shape 为 `[B,H,T,V]`，dtype 与 `query` 一致。 |
| `finalState` | 输出 | 最终状态矩阵。当前 shape 为 `[stateCount,H,K,V]`；传 dummy tensor 时不会作为有效输出使用。 |
| `chunkState` | 输出 | 每个 chunk 的状态矩阵，即原 `h` 输出。当前 shape 为 `[B,H,totalChunks,K,V]`。 |
| `stream` | 输入 | ACL runtime stream。 |

## Shape 约束

当前 example 使用 BHT/BHTD 布局：

```text
query/key:       [B,H,T,K]
value/out:       [B,H,T,V]
beta/gOptional:  [B,H,T]
chunkState:      [B,H,totalChunks,K,V]
initialState:    [stateCount,H,K,V]
finalState:      [stateCount,H,K,V]
```

其中：

```text
stateCount = dense ? B : actualSeqLengths.shape[0]
totalChunks = dense ? ceil(T / chunkSize)
                    : sum_i ceil(actualSeqLengths[i] / chunkSize)
```

`ChunkGatedDeltaRule` 会校验 `chunkState.shape[2]` 是否等于按 `chunkSize` 计算出的 `totalChunks`。

## 使用方法

调用方负责创建业务输入输出 `aclTensor`，中间 tensor 由 `ChunkGatedDeltaRule` 内部分配和释放。

```cpp
const aclTensor *actualSeqLengths = nullptr;  // dense 场景

bool ok = gdr_aclnn::ChunkGatedDeltaRule(
    query,
    key,
    value,
    beta,
    initialState,
    actualSeqLengths,
    g,
    scaleValue,
    chunkSize,
    out,
    finalState,
    chunkState,
    stream);
```

varlen 场景需要传入 `actualSeqLengths`：

```cpp
// actualSeqLengths = [64, 64]
bool ok = gdr_aclnn::ChunkGatedDeltaRule(
    query,
    key,
    value,
    beta,
    initialState,
    actualSeqLengths,
    g,
    scaleValue,
    chunkSize,
    out,
    finalState,
    chunkState,
    stream);
```

## Driver 输出

`flash_chunk_gated_delta_rule_fwd_aclnn.cpp` 当前写出：

```text
o.bin
chunkState.bin
final_state.bin  // 仅 --output-final-state true 时写出
```

`g/A/w/u/vNew` 等张量是内部中间结果，不再作为接口出参或默认输出文件。

## 与 ops-transformer README 的差异

该 example 的参数名和业务出参参考 README，但仍保留以下差异：

- 内部不是融合 `aclnnChunkGatedDeltaRule`，而是多个 ACLNN 子算子拼接。
- 当前布局是 BHT/BHTD；README 中融合算子为 TND。
- `chunkSize` 是显式入参；README 融合接口没有该参数。
- `chunkState` 是额外输出，用于保存每个 chunk 的状态矩阵。
- 当前 helper 返回 `bool`，不是标准 aclnn 两段式 `aclnnStatus` 接口。
