#!/usr/bin/env python3
"""Compatibility smoke for legacy torch.ops.npu FLA NPU dispatcher entries."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
import fla_npu


class SkipLegacySmoke(RuntimeError):
    pass


def _setup_npu() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available")
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def _legacy_op(name: str):
    try:
        fla_npu.load_legacy_torch_ops()
    except FileNotFoundError as exc:
        message = str(exc)
        if "custom_aclnn_extension_lib" in message:
            raise SkipLegacySmoke(message) from exc
        raise
    if not hasattr(torch.ops.npu, name):
        raise RuntimeError(f"torch.ops.npu.{name} is not registered after load_legacy_torch_ops().")
    return getattr(torch.ops.npu, name)


def smoke_chunk_local_cumsum() -> None:
    op = _legacy_op("npu_chunk_local_cumsum")
    g = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 8)
    out = op(
        g.npu(),
        4,
        reverse=False,
        scale=1.0,
        head_first=True,
        output_dtype="float32",
    ).cpu()
    expected = torch.tensor([[[1, 3, 6, 10, 5, 11, 18, 26]]], dtype=torch.float32)
    if out.dtype != torch.float32:
        raise AssertionError(f"chunk_local_cumsum legacy dtype mismatch: {out.dtype}")
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def smoke_chunk_scaled_dot_kkt() -> None:
    op = _legacy_op("npu_chunk_scaled_dot_kkt")
    k = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4).to(torch.float16) * 0.01
    g = torch.zeros((1, 1, 4), dtype=torch.float16)
    beta = torch.ones((1, 1, 4), dtype=torch.float16)
    out = op(k.npu(), g.npu(), beta.npu(), chunk_size=4).cpu()

    score = k.float()[0, 0] @ k.float()[0, 0].T
    expected = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    expected[0, 0] = torch.tril(score, diagonal=-1)

    if tuple(out.shape) != (1, 1, 4, 4):
        raise AssertionError(f"chunk_scaled_dot_kkt legacy shape mismatch: {tuple(out.shape)}")
    if out.dtype != torch.float32:
        raise AssertionError(f"chunk_scaled_dot_kkt legacy dtype mismatch: {out.dtype}")
    torch.testing.assert_close(out, expected, rtol=5e-3, atol=5e-3)


def main() -> int:
    _setup_npu()
    try:
        smoke_chunk_local_cumsum()
        smoke_chunk_scaled_dot_kkt()
    except SkipLegacySmoke as exc:
        print(f"[SKIP] legacy torch.ops.npu smoke: {exc}")
        return 0
    print("[PASS] legacy torch.ops.npu smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
