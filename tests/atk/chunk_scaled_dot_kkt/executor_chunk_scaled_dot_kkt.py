"""chunk_scaled_dot_kkt 的 ATK executor（三路多标杆）。

标杆体系与 chunk_kda_fwd 相同，按 ATK 任务角色分发：

  * NPU DUT：fla_npu.ops.ascendc.chunk_scaled_dot_kkt
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
    _gate,
    _marker_device,
    _rand,
    _randn,
)


OP_NAME = "chunk_scaled_dot_kkt"
_DEFAULT_SEED = 20260817
_DEFAULT_TRITON_CALLABLE = "fla.ops.triton.triton_core.chunk_scaled_dot_kkt:chunk_scaled_dot_kkt_fwd"


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc_dtype = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", _DEFAULT_SEED))
    B, HK, HV, T, K = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K"))
    return {
        "k": _randn((B, HK, T, K), dtype_name, calc_dtype, device, seed + 1),
        "g": _gate((B, HV, T), torch.float64 if high_precision else torch.float32, device, seed + 2),
        "beta": _rand((B, HV, T), "fp32", torch.float64 if high_precision else torch.float32, device, seed + 3, 0.1, 0.9),
        "chunk_size": int(spec["chunk_size"]),
    }


def _chunk_scaled_dot_kkt_ref(inputs):
    k, g, beta = inputs["k"], inputs["g"], inputs["beta"]
    B, HK, T, _ = k.shape
    HV = g.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc = torch.float64 if k.dtype == torch.float64 else torch.float32
    if beta.shape != g.shape or HV % HK != 0:
        raise ValueError("chunk_scaled_dot_kkt requires beta.shape == g.shape and HV divisible by HK")
    head_ratio = HV // HK
    out = torch.zeros((B, HV, T, chunk_size), dtype=calc, device=k.device)
    for b in range(B):
        for hv in range(HV):
            hk = hv // head_ratio
            for start, end in _chunks(T, chunk_size):
                length = end - start
                k_chunk = k[b, hk, start:end].to(calc)
                g_chunk = g[b, hv, start:end].to(calc)
                beta_chunk = beta[b, hv, start:end].to(calc)
                score = torch.matmul(k_chunk, k_chunk.t())
                gate = torch.exp(torch.clamp(g_chunk[:, None] - g_chunk[None, :], -50.0, 50.0))
                mask = torch.tril(torch.ones((length, length), dtype=calc, device=k.device), diagonal=-1)
                out[b, hv, start:end, :length] = score * gate * beta_chunk[:, None] * mask
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
    k, g, beta = inputs["k"], inputs["g"], inputs["beta"]
    if k.device.type not in {"cpu", "cuda"}:
        raise RuntimeError("Torch FP64 golden must run on an ATK CPU or GPU node")
    if any(
        tensor.dtype != torch.float64
        for tensor in (k, g, beta)
        if tensor is not None and tensor.is_floating_point()
    ):
        raise RuntimeError("Torch FP64 golden received a non-FP64 floating input")
    if k.device.type == "cpu":
        with _single_threaded_torch():
            return _chunk_scaled_dot_kkt_ref(inputs)
    return _chunk_scaled_dot_kkt_ref(inputs)


def _torch_same_precision(inputs: dict[str, Any], spec: dict[str, Any]):
    """CPU 普通任务：同精度对照，FP32 计算，与 kernel 计算边界一致。"""
    k = inputs["k"]
    if k.device.type != "cpu":
        raise RuntimeError("Torch same-precision control must run on an ATK CPU node")
    with _single_threaded_torch():
        return _chunk_scaled_dot_kkt_ref(inputs)


def _load_triton_callable() -> tuple[str, Callable]:
    target = os.environ.get("KKT_ATK_TRITON_CALLABLE", _DEFAULT_TRITON_CALLABLE).strip()
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError("KKT_ATK_TRITON_CALLABLE must use '<python_module>:<callable>' syntax")
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attribute, None)
    if not callable(callable_obj):
        raise RuntimeError(f"configured Triton target is not callable: {target}")
    return target, callable_obj


def _triton_same_precision(inputs: dict[str, Any], spec: dict[str, Any]):
    """GPU 普通任务：Triton 同精度对照。"""
    k, g, beta = inputs["k"], inputs["g"], inputs["beta"]
    if k.device.type != "cuda":
        raise RuntimeError("Triton same-precision control must run on an ATK GPU node")
    target, triton_callable = _load_triton_callable()
    # 仓内 Triton 实现的 g/beta 布局是 [B, T, Hv]，输出 A 是 [B, T, Hv, BT]；
    # AscendC 入口使用 [B, Hv, T] 输入和 [B, Hv, T, BT] 输出，这里做布局适配。
    A = triton_callable(
        k=k.contiguous(),
        g=g.transpose(1, 2).contiguous(),
        gk=None,
        beta=beta.transpose(1, 2).contiguous(),
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=int(spec["chunk_size"]),
        output_dtype=torch.float32,
    )
    if not isinstance(A, torch.Tensor):
        raise RuntimeError(f"Triton callable {target} must return a single tensor")
    return A.permute(0, 2, 1, 3).contiguous()


def _run_positive_npu(inputs: dict[str, Any], spec: dict[str, Any]):
    """NPU DUT：fla_npu.ops.ascendc 公开入口。"""
    from fla_npu.ops import ascendc

    return ascendc.chunk_scaled_dot_kkt(
        inputs["k"],
        inputs["g"],
        inputs["beta"],
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=int(spec["chunk_size"]),
    )


@register("executor_chunk_scaled_dot_kkt")
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
        if os.environ.get("KKT_ATK_TRACE_SEED") == "1":
            print(
                "KKT_ATK_RUNTIME_SEED",
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
