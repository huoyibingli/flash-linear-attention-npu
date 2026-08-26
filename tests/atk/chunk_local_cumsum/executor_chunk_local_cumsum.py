"""chunk_local_cumsum 的 ATK executor（三路多标杆）。

标杆体系与 chunk_kda_fwd 相同，按 ATK 任务角色分发：

  * NPU DUT：fla_npu.ops.ascendc.chunk_local_cumsum
  * CPU 标杆：benchmark 任务跑 Torch FP64 golden，普通任务跑同精度对照
  * GPU 标杆：benchmark 任务跑 Torch FP64 golden，普通任务跑 Triton 同精度对照

输入生成、CPU 标杆、Triton 标杆、NPU DUT 和 FunctionApi 都放在本算子目录中；
公共文件只提供基础工具函数。
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _chunks,
    _finite_tuple,
    _marker_device,
    _randn,
)


OP_NAME = "chunk_local_cumsum"
_DEFAULT_SEED = 20260817
_DEFAULT_TRITON_CALLABLE = "fla.ops.triton.triton_core.cumsum:chunk_local_cumsum"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", _DEFAULT_SEED))
    B, H, T = (int(spec[x]) for x in ("B", "H", "T"))
    return {
        "g": _randn((B, H, T), dtype_name, calc_dtype, device, seed + 1),
        "chunk_size": int(spec["chunk_size"]),
        "reverse": bool(spec.get("reverse", False)),
        "scale": float(spec.get("scale", 1.0)),
    }


def _chunk_local_cumsum_ref(inputs):
    g = inputs["g"]
    calc = torch.float64 if g.dtype == torch.float64 else torch.float32
    out = torch.zeros_like(g, dtype=calc)
    for start, end in _chunks(g.shape[-1], int(inputs["chunk_size"])):
        chunk = g[..., start:end].to(calc) * float(inputs["scale"])
        if bool(inputs["reverse"]):
            out[..., start:end] = torch.flip(torch.cumsum(torch.flip(chunk, dims=(-1,)), dim=-1), dims=(-1,))
        else:
            out[..., start:end] = torch.cumsum(chunk, dim=-1)
    return out.to(torch.float32)


@contextlib.contextmanager
def _single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _torch_fp64_golden(inputs: dict[str, Any], spec: dict[str, Any]):
    """CPU/GPU benchmark 任务：Torch FP64 golden。"""
    g = inputs["g"]
    if g.device.type not in {"cpu", "cuda"}:
        raise RuntimeError("Torch FP64 golden must run on an ATK CPU or GPU node")
    if g.dtype != torch.float64:
        raise RuntimeError("Torch FP64 golden received a non-FP64 g input")
    if g.device.type == "cpu":
        with _single_threaded_torch():
            return _chunk_local_cumsum_ref(inputs)
    return _chunk_local_cumsum_ref(inputs)


def _torch_same_precision(inputs: dict[str, Any], spec: dict[str, Any]):
    """CPU 普通任务：同精度对照，FP32 累加，与 kernel 计算边界一致。"""
    g = inputs["g"]
    if g.device.type != "cpu":
        raise RuntimeError("Torch same-precision control must run on an ATK CPU node")
    with _single_threaded_torch():
        return _chunk_local_cumsum_ref(inputs)


def _load_triton_callable() -> tuple[str, Callable]:
    target = os.environ.get("CUMSUM_ATK_TRITON_CALLABLE", _DEFAULT_TRITON_CALLABLE).strip()
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError("CUMSUM_ATK_TRITON_CALLABLE must use '<python_module>:<callable>' syntax")
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attribute, None)
    if not callable(callable_obj):
        raise RuntimeError(f"configured Triton target is not callable: {target}")
    return target, callable_obj


def _triton_same_precision(inputs: dict[str, Any], spec: dict[str, Any]):
    """GPU 普通任务：Triton 同精度对照。"""
    g = inputs["g"]
    if g.device.type != "cuda":
        raise RuntimeError("Triton same-precision control must run on an ATK GPU node")
    target, triton_callable = _load_triton_callable()
    # 仓内 Triton 实现按 [B, T, H] 计算；AscendC 入口使用 head-first [B, H, T]，
    # 这里转置后调用，输出再转回 head-first 交给 ATK。
    g_bth = g.transpose(1, 2).contiguous()
    out = triton_callable(
        g_bth,
        int(spec["chunk_size"]),
        reverse=bool(spec.get("reverse", False)),
        scale=float(spec.get("scale", 1.0)),
        cu_seqlens=None,
        chunk_indices_out=None,
        head_first=False,
        output_dtype=torch.float32,
    )
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Triton callable {target} must return a single tensor")
    return out.transpose(1, 2).contiguous()


def _run_positive_npu(inputs: dict[str, Any], spec: dict[str, Any]):
    """NPU DUT：fla_npu.ops.ascendc 公开入口。"""
    from fla_npu.ops import ascendc

    return ascendc.chunk_local_cumsum(
        inputs["g"],
        int(spec["chunk_size"]),
        cu_seqlens=None,
        chunk_indices_out=None,
        reverse=bool(spec.get("reverse", False)),
        scale=float(spec.get("scale", 1.0)),
        head_first=True,
        output_dtype="float32",
    )


@register("executor_chunk_local_cumsum")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.spec: Optional[dict[str, Any]] = None
        self.inputs: Optional[dict[str, Any]] = None
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        # benchmark 任务是 FP64 真值；普通 CPU 任务是同精度对照；
        # 普通 GPU 任务是 Triton 同精度对照。
        self.high_precision = self.device in {"cpu", "gpu"} and self.is_benchmark_task
        self.cpu_control = self.device == "cpu" and not self.is_benchmark_task
        self.triton_control = self.device == "gpu" and not self.is_benchmark_task

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = _case_spec(input_data, OP_NAME)
        self.inputs = build_inputs(
            self.spec,
            _marker_device(input_data),
            high_precision=self.high_precision,
        )
        if os.environ.get("CUMSUM_ATK_TRACE_SEED") == "1":
            print(
                "CUMSUM_ATK_RUNTIME_SEED",
                f"device={self.device}",
                f"benchmark={self.is_benchmark_task}",
                f"high_precision={self.high_precision}",
                f"cpu_control={self.cpu_control}",
                f"triton_control={self.triton_control}",
                f"seed={self.spec.get('seed', _DEFAULT_SEED)}",
                flush=True,
            )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.inputs is None or self.spec != spec:
            self.init_by_input_data(input_data)
        if self.high_precision:
            outputs = _torch_fp64_golden(self.inputs, self.spec)
        elif self.cpu_control:
            outputs = _torch_same_precision(self.inputs, self.spec)
        elif self.triton_control:
            outputs = _triton_same_precision(self.inputs, self.spec)
        elif self.device in {"npu", "pyaclnn"}:
            outputs = _run_positive_npu(self.inputs, self.spec)
        else:
            raise RuntimeError(
                f"{OP_NAME} 仅支持 NPU DUT、CPU 标杆与 GPU Triton 标杆节点，当前设备：{self.device!r}"
            )
        return _finite_tuple(outputs)
