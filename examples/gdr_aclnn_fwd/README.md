# ChunkGatedDeltaRule ACLNN Example Interface

## 功能说明

`gdr_aclnn::ChunkGatedDeltaRule` 用于在 C++ 示例中完成 chunk 版 Gated Delta Rule 前向计算。接口参数命名参考 `ops-transformer/attention/chunk_gated_delta_rule/README.md` 的业务参数，但内部实现保持当前 example 的子算子拼接流程：

```text
ChunkLocalCumsum -> ChunkScaledDotKkt -> Cast -> Permute -> SolveTri
-> Permute -> RecomputeWUFwd -> ChunkGatedDeltaRuleFwdH -> ChunkFwdO
```

接口支持两类输入 shape：README 中融合算子的 TND 布局，以及历史 example 使用的 BHT/BHTD 布局。TND 布局会在接口内部通过 `aclnnPermute` 转为当前小算子拼接需要的 BHT/BHTD 布局。

## 接口原型

```cpp
aclnnStatus gdr_aclnn::ChunkGatedDeltaRule(
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

## 新增接口说明

该接口用于在调用层替换原融合算子 `aclnnChunkGatedDeltaRule` 的调用。README/TND 场景下，除以下两点外，参数含义和 shape 尽量与融合算子保持一致：

- 新增显式属性 `chunkSize`，用于内部生成 `cuSeqlens` 和 `chunkIndices`。
- 新增输出 `chunkState`，用于保存每个 chunk 开始前的状态矩阵。

README/TND 调用时，输入输出 tensor 使用融合算子 README 中的业务 shape：

```text
query/key:              [T,Nk,Dk]
value/out:              [T,Nv,Dv]
beta/gOptional:         [T,Nv]
initialState/finalState:[B,Nv,Dv,Dk]
actualSeqLengths:       [B]
chunkState:             [totalChunks,Nv,Dv,Dk]
```

维度含义：

| 维度 | 含义 |
| --- | --- |
| `B` | batch size，批大小；TND 变长场景下等于 `actualSeqLengths` 的元素个数。 |
| `T` | token 总数或序列长度；dense BHTD 场景下表示每个 batch 的固定序列长度，TND 场景下表示所有序列拼接后的 token 总数。 |
| `Nk` | query/key 的 head 数。 |
| `Nv` | value/out 的 head 数；当前小算子拼接实现要求 `Nk == Nv`。 |
| `Dk` / `K` | query/key 每个 head 的特征维度。 |
| `Dv` / `V` | value/out 每个 head 的特征维度。 |
| `H` | 历史 BHT/BHTD 布局中的 head 数，对应当前实现里的统一 head 维度。 |
| `stateCount` | 状态矩阵的 batch 维度，dense 场景等于 `B`，变长场景等于 `actualSeqLengths.shape[0]`。 |
| `totalChunks` | 所有序列按 `chunkSize` 切分后的 chunk 总数。 |

接口内部会将 TND 输入通过 `aclnnPermute` 转为当前小算子拼接实现需要的 BHT/BHTD 布局，执行完成后再将 `out`、`finalState` 和 `chunkState` 转回 README/TND 布局。历史 BHT/BHTD 调用仍保留，接口会根据 `query` 的 rank 自动分发。

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `query` | 输入 | GDR 公式中的 q。README shape 为 `[T,Nk,Dk]`，兼容旧 shape `[B,H,T,K]`，dtype 为 fp16/bf16。 |
| `key` | 输入 | GDR 公式中的 k。README shape 为 `[T,Nk,Dk]`，兼容旧 shape `[B,H,T,K]`，dtype 与 `query` 一致。 |
| `value` | 输入 | GDR 公式中的 v。README shape 为 `[T,Nv,Dv]`，兼容旧 shape `[B,H,T,V]`，dtype 与 `query` 一致。 |
| `beta` | 输入 | 更新强度 beta。README shape 为 `[T,Nv]`，兼容旧 shape `[B,H,T]`，当前 dtype 为 fp32。 |
| `initialState` | 输入 | 初始状态矩阵。README shape 为 `[B,Nv,Dv,Dk]`，兼容旧 shape `[stateCount,H,K,V]`。 |
| `actualSeqLengths` | 输入 | 每条序列的有效长度，shape 为 `[B]`，dtype 支持 int32/int64。README/TND 调用必须传入。旧 BHT dense 场景可传 `nullptr`。 |
| `gOptional` | 输入 | 衰减系数 g。README shape 为 `[T,Nv]`，兼容旧 shape `[B,H,T]`，dtype 为 fp32。传 `nullptr` 时内部使用全 0 g。 |
| `scaleValue` | 属性 | query 缩放因子。 |
| `chunkSize` | 属性 | chunk 大小，必须显式传入。接口会用它从 `actualSeqLengths` 生成 `cuSeqlens` 和 `chunkIndices`。 |
| `out` | 输出 | attention 输出。README shape 为 `[T,Nv,Dv]`，兼容旧 shape `[B,H,T,V]`，dtype 与 `query` 一致。 |
| `finalState` | 输出 | 最终状态矩阵。README shape 为 `[B,Nv,Dv,Dk]`，兼容旧 shape `[stateCount,H,K,V]`。 |
| `chunkState` | 输出 | 每个 chunk 开始前的状态矩阵，即原 `h` 输出。README shape 为 `[totalChunks,Nv,Dv,Dk]`，兼容旧 shape `[B,H,totalChunks,K,V]`。 |
| `stream` | 输入 | ACL runtime stream。 |

## Shape 约束

README/TND 布局：

```text
query/key:              [T,Nk,Dk]
value/out:              [T,Nv,Dv]
beta/gOptional:         [T,Nv]
initialState/finalState:[B,Nv,Dv,Dk]
actualSeqLengths:       [B]
chunkState:             [totalChunks,Nv,Dv,Dk]
```

当前 TND 适配要求 `Nk == Nv`。README 融合算子允许 `Nv % Nk == 0`，但当前小算子拼接 helper 只有一个 head 维度。

历史 BHT/BHTD 布局仍保留：

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

`ChunkGatedDeltaRule` 会校验 `chunkState` 的 chunk 维度是否等于按 `chunkSize` 计算出的 `totalChunks`。

## chunkState 与 finalState

`chunk` 只沿 token 序列维度 `T` 切分，不切 batch 维、head 维，也不切 `Dk/K` 或 `Dv/V` 特征维。每个 chunk 只读取自己范围内的 `query/key/value/beta/gOptional`，后续 chunk 通过起始 state 继承前面 chunk 压缩后的历史信息，不会重新读取前面 chunk 的 token。

`chunkState` 是 `aclnnChunkGatedDeltaRuleFwdH` 的 `h` 输出，保存的是每个 chunk 开始前的 state 缓存。可以按如下逻辑理解：

```text
state = initialState

for chunk_id in range(totalChunks):
    chunkState[chunk_id] = state
    state = update(state, current_chunk_tokens)

finalState = state
```

因此 `chunkState` 和 `finalState` 的关系是：

- `chunkState[0]` 是第 0 个 chunk 开始前的状态，通常等于 `initialState` 或全 0 初始状态。
- `chunkState[i]` 是第 `i` 个 chunk 的输入状态，也就是第 `i - 1` 个 chunk 处理完成后的状态。
- `finalState` 是最后一个有效 token 处理完成后的最终状态。
- `finalState` 通常不等于最后一个 `chunkState`。最后一个 `chunkState` 是最后一个 chunk 的输入状态，`finalState` 是最后一个 chunk 的输出状态。

以 `T = 555`、`chunkSize = 128` 为例，token 维会切成 5 个 chunk：

```text
chunk0: token [0,   127]
chunk1: token [128, 255]
chunk2: token [256, 383]
chunk3: token [384, 511]
chunk4: token [512, 554]
```

状态传递关系为：

```text
initialState -> chunk0 -> state_1 -> chunk1 -> state_2 -> ... -> chunk4 -> finalState

chunkState[0] = initialState
chunkState[1] = state_1
chunkState[2] = state_2
chunkState[3] = state_3
chunkState[4] = state_4
finalState    = chunk4 处理完成后的 state
```

当 `T` 不能被 `chunkSize` 整除时，最后一个 chunk 可能不是满 chunk。正确语义下，padding token 不应更新状态，`finalState` 对应最后一个有效 token 处理完成后的状态。

README/TND 布局下，`chunkState` 输出 shape 是 `[totalChunks,Nv,Dv,Dk]`。变长场景中 `totalChunks` 按 `actualSeqLengths` 逐条序列累加得到，chunk 顺序与 `actualSeqLengths` 的序列顺序一致。历史 BHT/BHTD 布局下，`chunkState` shape 是 `[B,H,totalChunks,K,V]`。

## 使用方法

调用方负责创建业务输入输出 `aclTensor`，中间 tensor 由 `ChunkGatedDeltaRule` 内部分配和释放。

```cpp
const aclTensor *actualSeqLengths = nullptr;  // dense 场景

aclnnStatus status = gdr_aclnn::ChunkGatedDeltaRule(
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
bool ok = status == ACL_SUCCESS;
```

varlen 场景需要传入 `actualSeqLengths`：

```cpp
// actualSeqLengths = [64, 64]
aclnnStatus status = gdr_aclnn::ChunkGatedDeltaRule(
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
bool ok = status == ACL_SUCCESS;
```

## README TND 调用示例

`chunk_gated_delta_rule_tnd_example.cpp` 提供 README-shape 调用示例，外部输入使用
`ops-transformer/attention/chunk_gated_delta_rule/README.md` 中的 TND shape，直接调用新增的
`ChunkGatedDeltaRule` TND 分支。

输入目录文件：

```text
query.bin               [T,Nk,Dk]       fp16/bf16
key.bin                 [T,Nk,Dk]       fp16/bf16
value.bin               [T,Nv,Dv]       fp16/bf16
beta.bin                [T,Nv]          fp32
g.bin                   [T,Nv]          fp32
initial_state.bin       [B,Nv,Dv,Dk]    fp16/bf16
actual_seq_lengths.bin  [B]             int32
```

示例命令：

```bash
./chunk_gated_delta_rule_tnd_example \
  --input-dir ./input \
  --output-dir ./output \
  --batch 2 \
  --tokens 128 \
  --key-heads 2 \
  --value-heads 2 \
  --key-dim 128 \
  --value-dim 128 \
  --chunk-size 64 \
  --dtype bf16 \
  --scale 0.1
```

输出目录文件：

```text
out.bin          [T,Nv,Dv]              fp16/bf16
final_state.bin  [B,Nv,Dv,Dk]           fp16/bf16
chunkState.bin   [totalChunks,Nv,Dv,Dk] fp16/bf16
```

该示例当前只支持 `Nk == Nv`。README 融合算子允许 `Nv % Nk == 0`，但现有 helper 的入参只有一个 head
维度，不能在当前小算子拼接实现中无损表达不同的 key/value head 数。

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
- README/TND 布局已支持，但当前要求 `Nk == Nv`。
- `chunkSize` 是新增的显式入参；README 融合接口没有该参数。
- `chunkState` 是额外输出，用于保存每个 chunk 的状态矩阵。
- 当前 helper 返回 `aclnnStatus`，但不是标准 aclnn 两段式 `GetWorkspaceSize + Run` 接口。
