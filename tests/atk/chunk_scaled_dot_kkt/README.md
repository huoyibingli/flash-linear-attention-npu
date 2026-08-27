# ChunkScaledDotKkt ATK 工程

本目录提供 `chunk_scaled_dot_kkt` 的 ATK 单算子工程，包含 `executor_chunk_scaled_dot_kkt.py`、`gen_chunk_scaled_dot_kkt.py`、`chunk_scaled_dot_kkt.yaml`、`atk_chunk_scaled_dot_kkt.json`。

## 输入约束

- `k` 必须为 4D，shape 为 `[B,Hk,T,K]`，head-first 排布。
- `g/beta` 必须为 3D，shape 为 `[B,Hv,T]`，并且 `B/T` 与 `k` 一致。
- GVA 要求 `Hv % Hk == 0`；输出 `A` 的 head 维与 `k` 对齐，shape 为 `[B,Hk,T,chunk_size]`。
- `k` 支持 `BFLOAT16/FLOAT16`；`g/beta/A` 按源码 README 使用 `FLOAT`。
- `chunk_size` 支持 `16/32/64/128`；当前版本不支持 `gk` 分支。
- `cu_seqlens` 和 `chunk_indices` 必须同时传入或同时省略；`chunk_indices` 以 `[seq_id, chunk_id]` 成对存放。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆体系（三路多标杆）

与 `chunk_kda_fwd` 相同的三路多标杆体系，executor 按 ATK 任务角色分发：

| 节点 | benchmark 任务 | 普通任务 |
| --- | --- | --- |
| NPU DUT | - | `fla_npu.ops.ascendc.chunk_scaled_dot_kkt`（FP32 输出） |
| CPU 标杆 | Torch FP64 golden（单线程） | Torch 同精度对照（FP32 计算，单线程） |
| GPU 标杆 | Torch FP64 golden（CUDA） | Triton 同精度对照 |

- 三路输入由同一 `case_spec` seed 确定性生成：先量化到用例 dtype，再按角色转换到目标
  精度，因此 NPU DUT、同精度对照、FP64 golden 与 Triton 对照使用同一份数值。
- 精度标准沿用 YAML 中 `cv_fused_double_benchmark`，executor 不另设阈值。
- CPU 双标杆拓扑直接使用统一脚本 `run_test_cpu.sh -scope=accuracy`（`--bm_device cpu`），
  不需要单独启动 ATK server。
- GPU 分布式拓扑参照 [`../chunk_kda_fwd/README.md`](../chunk_kda_fwd/README.md)：远端 GPU
  容器运行 `atk server`，本机 NPU 侧发起 `--bm_device gpu` 的 accuracy 任务。GPU 节点的
  benchmark 任务是 CUDA Torch FP64 golden，普通任务是 Triton 同精度对照。GPU 容器只需要
  CUDA Torch、Triton 和本仓库源码路径（`PYTHONPATH` 包含仓库根目录），不需要安装 CANN
  或 NPU wheel。

### Triton 对照

- 默认 callable 为 `fla.ops.triton.triton_core.chunk_scaled_dot_kkt:chunk_scaled_dot_kkt_fwd`，
  即仓内 Triton 实现的 `gk=None` 分支，只依赖 torch + triton，可在纯 CUDA 环境导入。
- 可通过 `KKT_ATK_TRITON_CALLABLE` 覆盖，格式 `<python_module>:<callable>`；callable
  签名必须与仓内实现一致：
  `chunk_scaled_dot_kkt_fwd(k, g=None, gk=None, beta=None, cu_seqlens=None, chunk_indices=None, chunk_size=64, output_dtype=torch.float32)`。
- 仓内 Triton 实现的 `g/beta` 布局是 `[B,T,Hv]`、输出 `A` 是 `[B,T,Hv,BT]`；executor 会
  把 head-first `[B,Hv,T]` 输入转置后调用，并把输出重排为 `[B,Hv,T,BT]` 再交给 ATK。
- Triton 路径当前只覆盖 dense 用例（`cu_seqlens=None`），与当前 ATK 全量用例一致；
  `chunk_size>64` 走仓内 sub-block Triton kernel，语义与参考实现一致。

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `KKT_ATK_TRITON_CALLABLE` | `fla.ops.triton.triton_core.chunk_scaled_dot_kkt:chunk_scaled_dot_kkt_fwd` | GPU Triton 对照实现 |
| `KKT_ATK_TRACE_SEED` | 未设置 | 设为 `1` 时打印节点角色与运行 seed |

## 标杆来源

torch_custom/fla_npu/test/test_npu_chunk_scaled_dot_kkt.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/README.md; fla/ops/triton/triton_core/chunk_scaled_dot_kkt.py（Triton 对照）

CPU 标杆、Triton 标杆、NPU DUT 调用和 FunctionApi 均在本目录的 `executor_chunk_scaled_dot_kkt.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 16, "chunk_size": 16, "op": "chunk_scaled_dot_kkt", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 16, "chunk_size": 16, "op": "chunk_scaled_dot_kkt", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## Tiling key 覆盖

tiling key 由 dtype（fp16=10/bf16=20）与 chunk_size 档位（64=0/16=1/32=2/128=3，左移 8 位）决定，共 8 个。
主精度用例集（`atk_chunk_scaled_dot_kkt.json`）覆盖全部 8 个 tiling key。生成器基表与扩展流程只产生
`chunk_size` 64/128 档位，`gen_chunk_scaled_dot_kkt.py` 的 `_ensure_tiling_key_coverage` 会为缺失档位
自动补充固定覆盖 shape（当前为 case 196-199 的 BT16/BT32，shape `B=1, HK=2, HV=2, T=128, K=128`），
并替换尾部合成 shape 以保持用例总数 200 不变；`gen_cases` 默认参数（`-dt 100`）即可复现，无需手工追加。

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
