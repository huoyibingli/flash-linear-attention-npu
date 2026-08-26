# ChunkLocalCumsum ATK 工程

本目录提供 `chunk_local_cumsum` 的 ATK 单算子工程，包含 `executor_chunk_local_cumsum.py`、`gen_chunk_local_cumsum.py`、`chunk_local_cumsum.yaml`、`atk_chunk_local_cumsum.json`。

## 输入约束

- `g` 必须为 rank 3，shape 为 `[B,H,T]`，当前 AscendC kernel 仅支持 `head_first=true`。
- `g/out` 支持 `FLOAT/FLOAT16/BFLOAT16`，kernel 内部按 FP32 累加后转换为输出 dtype。
- `chunk_size` 必须为 2 的幂，并满足 host tiling 推导出的 `block_t >= chunk_size`。
- `reverse` 控制 chunk 内累加方向；`scale` 为输出缩放系数。
- 变长模式下 `cu_seqlens` 非空时，`chunk_indices_out` 必须非空且元素数为偶数，同时要求 `B=1`。
- `output_dtype` 支持 `float32/float16/bfloat16` 及跟随输入 dtype 的别名。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆体系（三路多标杆）

与 `chunk_kda_fwd` 相同的三路多标杆体系，executor 按 ATK 任务角色分发：

| 节点 | benchmark 任务 | 普通任务 |
| --- | --- | --- |
| NPU DUT | - | `fla_npu.ops.ascendc.chunk_local_cumsum`（`head_first=true`，FP32 输出） |
| CPU 标杆 | Torch FP64 golden（单线程） | Torch 同精度对照（FP32 累加，单线程） |
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

- 默认 callable 为 `fla.ops.triton.triton_core.cumsum:chunk_local_cumsum`，即仓内 Triton
  实现，只依赖 torch + triton，可在纯 CUDA 环境导入。
- 可通过 `CUMSUM_ATK_TRITON_CALLABLE` 覆盖，格式 `<python_module>:<callable>`；callable
  签名必须与仓内实现一致：
  `chunk_local_cumsum(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, chunk_indices_out=None, head_first=False, output_dtype=torch.float32)`。
- 仓内 Triton 实现按 `[B,T,H]` 计算；executor 会把 head-first `[B,H,T]` 输入转置后调用，
  并把输出转回 `[B,H,T]` 再交给 ATK。
- Triton 路径当前只覆盖 dense 用例（`cu_seqlens=None`），与当前 ATK 全量用例一致。
- ATK 用例的 `scale` 均为 2 的幂，"先乘 scale 再累加"与"先累加再乘 scale"在 FP32 下
  逐位一致，AscendC/Triton/参考实现三路语义一致。

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CUMSUM_ATK_TRITON_CALLABLE` | `fla.ops.triton.triton_core.cumsum:chunk_local_cumsum` | GPU Triton 对照实现 |
| `CUMSUM_ATK_TRACE_SEED` | 未设置 | 设为 `1` 时打印节点角色与运行 seed |

## 标杆来源

torch_custom/fla_npu/test/test_npu_chunk_local_cumsum.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/README.md; fla/ops/triton/triton_core/cumsum.py（Triton 对照）

CPU 标杆、Triton 标杆、NPU DUT 调用和 FunctionApi 均在本目录的 `executor_chunk_local_cumsum.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "H": 1, "T": 16, "chunk_size": 8, "reverse": false, "scale": 1.0, "op": "chunk_local_cumsum", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "H": 1, "T": 16, "chunk_size": 8, "reverse": false, "scale": 1.0, "op": "chunk_local_cumsum", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
