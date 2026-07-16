#!/usr/bin/env python3
"""Compare Triton and AscendC preprocessing inside flash_chunk_gated_delta_rule_fwd."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

_LOCAL_OPP_ROOT = _REPO_ROOT / "build" / "local_opp"
if _LOCAL_OPP_ROOT.exists():
    os.environ.setdefault("FLA_NPU_OPP_PATH", str(_LOCAL_OPP_ROOT))
    _LOCAL_OP_API_LIB = None
    for vendor_dir in (_LOCAL_OPP_ROOT / "vendors").glob("*"):
        candidate = vendor_dir / "op_api" / "lib"
        if (candidate / "libcust_opapi.so").exists():
            _LOCAL_OP_API_LIB = candidate
            break
    if _LOCAL_OP_API_LIB is not None and _LOCAL_OP_API_LIB.exists():
        opapi_alias = _LOCAL_OP_API_LIB / "libopapi.so"
        if not opapi_alias.exists():
            opapi_alias.symlink_to("libcust_opapi.so")
        lib_path = str(_LOCAL_OP_API_LIB)
        ld_parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
        if __name__ == "__main__" and lib_path not in ld_parts and os.environ.get("FLA_NPU_TEST_REEXEC") != "1":
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = os.pathsep.join([lib_path, *ld_parts])
            env["FLA_NPU_TEST_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

import openpyxl
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import fla_npu  # noqa: F401

from examples import flash_gated_delta_rule as fg
from fla_npu.ops.triton import (
    chunk_local_cumsum as triton_chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd as triton_chunk_scaled_dot_kkt_fwd,
)


EPS = 1e-12
RATIO_DENOM_EPS = 1e-12
RATIO_MARE_THRESHOLD = 5.0
RATIO_MERE_THRESHOLD = 1.5
RATIO_RMSE_THRESHOLD = 1.5
RATIO_ERR_COUNT_THRESHOLD = 2.0
ERR_COUNT_ATOL = 5e-3
ERR_COUNT_RTOL = 5e-3


def require_kkt_schema() -> None:
    schemas = str(torch.ops.npu.npu_chunk_scaled_dot_kkt._schemas)
    if "cu_seqlens" not in schemas or "chunk_indices" not in schemas:
        raise RuntimeError(
            "Loaded npu_chunk_scaled_dot_kkt uses the old ABI. Reinstall the rebuilt fla_npu wheel before running "
            f"this test. schemas={schemas}"
        )


SELECTED_CASE_IDS = (
    "BSND_noGVA_V128_13",
    "TND_noGVA_V128_20",
    "TND_GVA_V256_21",
    "TND_GVA_V256_22",
    "TND_GVA_V256_27",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    original_id: str
    batch: int
    value_heads: int
    key_heads: int
    tokens: int
    value_dim: int
    key_dim: int
    chunk_size: int
    is_enable_gk: bool
    desc: str

    @property
    def is_varlen(self) -> bool:
        return self.case_id.startswith("TND_")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def load_cases(path: Path) -> dict[str, Case]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    cases: dict[str, Case] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        case = Case(
            case_id=str(row[0]),
            original_id=str(row[1]),
            batch=int(row[2]),
            value_heads=int(row[3]),
            key_heads=int(row[4]),
            tokens=int(row[5]),
            value_dim=int(row[6]),
            key_dim=int(row[7]),
            chunk_size=int(row[8]),
            is_enable_gk=_as_bool(row[9]),
            desc=str(row[10] or ""),
        )
        cases[case.case_id] = case
    return cases


def iter_selected_cases(cases: dict[str, Case], selected_ids: Iterable[str], limit: int) -> list[Case]:
    selected = []
    for case_id in selected_ids:
        if case_id not in cases:
            raise KeyError(f"{case_id} not found in test_case.xlsx")
        selected.append(cases[case_id])
    return selected if limit <= 0 else selected[:limit]


def iter_all_cases(cases: dict[str, Case], limit: int) -> list[Case]:
    selected = list(cases.values())
    return selected if limit <= 0 else selected[:limit]


def estimated_input_gib(case: Case) -> float:
    qk_bytes = case.batch * case.value_heads * case.tokens * case.key_dim * 2 * 2
    v_bytes = case.batch * case.value_heads * case.tokens * case.value_dim * 2
    gate_bytes = case.batch * case.tokens * case.value_heads * 4 * 2
    return (qk_bytes + v_bytes + gate_bytes) / (1024 ** 3)


def triton_unsupported_reason(case: Case) -> str:
    if case.chunk_size == 128:
        return "Triton KKT baseline for chunk_size=128 fails BiShengIR UB check on this environment"
    return ""


def _fmt_metric(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value == float("inf"):
        return "inf"
    if value != value:
        return "nan"
    return f"{value:.4g}"


def _analysis_brief(analysis: dict | None) -> str:
    if not analysis:
        return ""
    verdict = analysis.get("precision_verdict", "")
    if analysis.get("stage") == "single_kkt":
        ratios = analysis.get("dual", {}).get("ratios", {})
        return (
            f"{verdict}; "
            f"MARE={_fmt_metric(ratios.get('MARE_ratio'))}, "
            f"MERE={_fmt_metric(ratios.get('MERE_ratio'))}, "
            f"RMSE={_fmt_metric(ratios.get('RMSE_ratio'))}, "
            f"ERR_COUNT={_fmt_metric(ratios.get('ERR_COUNT_ratio'))}"
        )
    if analysis.get("stage") == "full_o":
        direct = analysis.get("direct", {})
        return (
            f"{verdict}; finite max_abs={_fmt_metric(direct.get('max_abs'))}, "
            f"mean_abs={_fmt_metric(direct.get('mean_abs'))}, "
            f"special_equal={direct.get('special_equal')}"
        )
    return f"{verdict}: {analysis.get('failure_reason', '')}"


def write_report(report_path: Path, records: list[dict], args: argparse.Namespace) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "pass": sum(1 for item in records if item["status"] == "PASS"),
        "fail": sum(1 for item in records if item["status"] == "FAIL"),
        "skip": sum(1 for item in records if item["status"] == "SKIP"),
    }
    precision_counts: dict[str, int] = {}
    for item in records:
        verdict = item.get("analysis", {}).get("precision_verdict")
        if verdict:
            precision_counts[verdict] = precision_counts.get(verdict, 0) + 1
    lines = [
        "# flash_chunk_gated_delta_rule_fwd AscendC 验证报告",
        "",
        f"- 时间: {datetime.now(timezone.utc).isoformat()}",
        f"- Excel: `{args.xlsx}`",
        f"- dtype: `{args.dtype}`",
        f"- device-id: `{args.device_id}`",
        f"- all: `{args.all}`",
        f"- limit: `{args.limit}`",
        f"- max-input-gib: `{args.max_input_gib}`",
        f"- 汇总: PASS={counts['pass']} FAIL={counts['fail']} SKIP={counts['skip']} TOTAL={len(records)}",
        "- 精度复判阈值: "
        f"MARE_ratio <= {RATIO_MARE_THRESHOLD}, "
        f"MERE_ratio <= {RATIO_MERE_THRESHOLD}, "
        f"RMSE_ratio <= {RATIO_RMSE_THRESHOLD}, "
        f"ERR_COUNT_ratio <= {RATIO_ERR_COUNT_THRESHOLD}",
        f"- 精度复判汇总: {precision_counts or {}}",
        "",
        "| case | status | precision | B | VH | KH | T | V | K | chunk | varlen | input GiB | reason | analysis |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for item in records:
        case = item["case"]
        reason = str(item.get("reason", "")).replace("\n", "<br>")
        analysis = _analysis_brief(item.get("analysis")).replace("\n", "<br>")
        precision = item.get("analysis", {}).get("precision_verdict", "")
        lines.append(
            f"| {case.case_id} | {item['status']} | {precision} | "
            f"{case.batch} | {case.value_heads} | {case.key_heads} | "
            f"{case.tokens} | {case.value_dim} | {case.key_dim} | {case.chunk_size} | {case.is_varlen} | "
            f"{item['estimated_input_gib']:.2f} | {reason} | {analysis} |"
        )

    failed_with_analysis = [item for item in records if item.get("analysis")]
    if failed_with_analysis:
        lines.extend(
            [
                "",
                "## 失败用例精度复判",
                "",
                "| case | stage | verdict | MARE_ratio | MERE_ratio | RMSE_ratio | ERR_COUNT_ratio | 结论 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in failed_with_analysis:
            case = item["case"]
            analysis = item["analysis"]
            if analysis.get("stage") == "single_kkt":
                ratios = analysis.get("dual", {}).get("ratios", {})
                conclusion = (
                    "四项 ratio 均满足阈值，不判定为精度问题。"
                    if analysis.get("precision_verdict") == "NO_PRECISION_ISSUE"
                    else "存在 ratio 超阈值，判定为精度问题。"
                )
                lines.append(
                    f"| {case.case_id} | {analysis.get('stage')} | {analysis.get('precision_verdict')} | "
                    f"{_fmt_metric(ratios.get('MARE_ratio'))} | "
                    f"{_fmt_metric(ratios.get('MERE_ratio'))} | "
                    f"{_fmt_metric(ratios.get('RMSE_ratio'))} | "
                    f"{_fmt_metric(ratios.get('ERR_COUNT_ratio'))} | {conclusion} |"
                )
            elif analysis.get("stage") == "full_o":
                direct = analysis.get("direct", {})
                if analysis.get("precision_verdict") == "SPECIAL_VALUE_MISMATCH":
                    conclusion = "有限值误差满足阈值，但 Inf/NaN special mask 不一致，属于 special-value 一致性问题。"
                elif analysis.get("precision_verdict") == "NO_PRECISION_ISSUE":
                    conclusion = "复判重跑 special mask 一致，有限值误差满足阈值；原始失败偏向偶发 special mask 差异。"
                else:
                    conclusion = analysis.get("failure_reason", "")
                lines.append(
                    f"| {case.case_id} | {analysis.get('stage')} | {analysis.get('precision_verdict')} | "
                    "N/A | N/A | N/A | N/A | "
                    f"{conclusion} actual_nan={direct.get('actual_nan')} expected_nan={direct.get('expected_nan')} "
                    f"nan_mismatch={direct.get('nan_mismatch')} |"
                )
            else:
                lines.append(
                    f"| {case.case_id} | {analysis.get('stage')} | {analysis.get('precision_verdict')} | "
                    f"N/A | N/A | N/A | N/A | {analysis.get('failure_reason', '')} |"
                )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "summary": counts | {"total": len(records)},
                "args": {
                    "xlsx": str(args.xlsx),
                    "dtype": args.dtype,
                    "device_id": args.device_id,
                    "all": args.all,
                    "limit": args.limit,
                    "max_input_gib": args.max_input_gib,
                },
                "records": [
                    {
                        "case_id": item["case"].case_id,
                        "status": item["status"],
                        "estimated_input_gib": item["estimated_input_gib"],
                        "reason": item.get("reason", ""),
                        "analysis": item.get("analysis", {}),
                    }
                    for item in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _make_cu_seqlens(total_tokens: int, seq_count: int, device: torch.device) -> torch.Tensor:
    seq_count = max(1, min(seq_count, total_tokens))
    base = total_tokens // seq_count
    rem = total_tokens % seq_count
    lengths = []
    for idx in range(seq_count):
        length = base + (1 if idx < rem else 0)
        if idx % 3 == 1 and length > 2:
            length -= 1
            lengths[-1] += 1
        lengths.append(length)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    offsets[-1] = total_tokens
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _varlen_seq_count(case: Case) -> int:
    known = {
        "TND_noGVA_V128_20": 8,
        "TND_GVA_V256_21": 128,
        "TND_GVA_V256_22": 128,
        "TND_GVA_V256_27": 32,
        "TND_GVA_V128_26": 17,
        "TND_GVA_V256_32": 667,
        "TND_GVA_V256_33": 13,
    }
    return known.get(case.case_id, max(1, min(32, case.tokens // max(1, case.chunk_size))))


def make_inputs(case: Case, device: torch.device, dtype: torch.dtype, seed: int) -> dict[str, torch.Tensor]:
    if case.value_heads % case.key_heads != 0:
        raise ValueError(f"{case.case_id}: value_heads must be divisible by key_heads for this test.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    repeat = case.value_heads // case.key_heads

    q_base = torch.randn(
        case.batch,
        case.key_heads,
        case.tokens,
        case.key_dim,
        dtype=torch.float32,
        generator=generator,
    )
    k_base = torch.randn(q_base.shape, dtype=torch.float32, generator=generator)
    q = F.normalize(q_base, p=2, dim=-1).to(dtype).repeat_interleave(repeat, dim=1).contiguous()
    k = F.normalize(k_base, p=2, dim=-1).to(dtype).repeat_interleave(repeat, dim=1).contiguous()
    v = (
        torch.randn(
            case.batch,
            case.value_heads,
            case.tokens,
            case.value_dim,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.2)
        .to(dtype)
    )
    beta = torch.sigmoid(
        torch.randn(case.batch, case.tokens, case.value_heads, dtype=torch.float32, generator=generator)
    )
    g = F.logsigmoid(torch.randn(case.batch, case.tokens, case.value_heads, dtype=torch.float32, generator=generator))

    return {
        "q": q.to(device),
        "k": k.to(device),
        "v": v.to(device),
        "beta": beta.to(device),
        "g": g.to(device),
    }


def make_metadata(case: Case, g: torch.Tensor):
    if not case.is_varlen:
        return None, None, None, None
    cu_seqlens = _make_cu_seqlens(case.tokens, _varlen_seq_count(case), g.device)
    return fg._ensure_varlen_metadata(
        g=g,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=None,
        chunk_indices=None,
        chunk_indices_list=None,
        chunk_size=case.chunk_size,
    )


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in inputs.items()}


def _metric(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    max_tol: float,
    mean_tol: float,
    mask: torch.Tensor | None = None,
) -> None:
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    if mask is not None:
        mask_cpu = mask.detach().cpu().bool()
        if mask_cpu.shape != actual_cpu.shape:
            mask_cpu = mask_cpu.expand(actual_cpu.shape)
        actual_cpu = actual_cpu[mask_cpu]
        expected_cpu = expected_cpu[mask_cpu]
    actual_nan = torch.isnan(actual_cpu)
    expected_nan = torch.isnan(expected_cpu)
    actual_posinf = torch.isposinf(actual_cpu)
    expected_posinf = torch.isposinf(expected_cpu)
    actual_neginf = torch.isneginf(actual_cpu)
    expected_neginf = torch.isneginf(expected_cpu)
    special_equal = bool(
        torch.equal(actual_nan, expected_nan)
        and torch.equal(actual_posinf, expected_posinf)
        and torch.equal(actual_neginf, expected_neginf)
    )
    finite_mask = torch.isfinite(actual_cpu) & torch.isfinite(expected_cpu)
    if finite_mask.any():
        diff = (actual_cpu[finite_mask] - expected_cpu[finite_mask]).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
    else:
        max_abs = 0.0
        mean_abs = 0.0
    special_count = int((~finite_mask).sum().item())
    ok = special_equal and max_abs <= max_tol and mean_abs <= mean_tol
    print(
        f"    {name}: shape={tuple(actual.shape)} dtype={actual.dtype} "
        f"max_abs={max_abs:.6e}/{max_tol:.1e} mean_abs={mean_abs:.6e}/{mean_tol:.1e} "
        f"special={special_count} special_equal={special_equal} ok={ok}"
    )
    if not ok:
        raise AssertionError(
            f"{name} mismatch: max_abs={max_abs}, mean_abs={mean_abs}, "
            f"special_count={special_count}, special_equal={special_equal}"
        )


def _assert_unchanged(name: str, before: torch.Tensor, after: torch.Tensor) -> None:
    _metric(f"{name}_unchanged", after, before, 0.0, 0.0)


def _new_error_acc() -> dict:
    return {
        "count": 0,
        "sum_abs": 0.0,
        "max_abs": 0.0,
        "sum_rel": 0.0,
        "max_rel": 0.0,
        "sum_sq": 0.0,
        "err_count": 0,
        "special_count": 0,
        "nan_mismatch": 0,
        "posinf_mismatch": 0,
        "neginf_mismatch": 0,
        "actual_nan": 0,
        "expected_nan": 0,
        "actual_posinf": 0,
        "expected_posinf": 0,
        "actual_neginf": 0,
        "expected_neginf": 0,
    }


def _update_error_acc(
    acc: dict,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    err_atol: float = ERR_COUNT_ATOL,
    err_rtol: float = ERR_COUNT_RTOL,
) -> None:
    actual_cpu = actual.detach().cpu().to(torch.float64)
    expected_cpu = expected.detach().cpu().to(torch.float64)
    actual_nan = torch.isnan(actual_cpu)
    expected_nan = torch.isnan(expected_cpu)
    actual_posinf = torch.isposinf(actual_cpu)
    expected_posinf = torch.isposinf(expected_cpu)
    actual_neginf = torch.isneginf(actual_cpu)
    expected_neginf = torch.isneginf(expected_cpu)

    acc["actual_nan"] += int(actual_nan.sum().item())
    acc["expected_nan"] += int(expected_nan.sum().item())
    acc["actual_posinf"] += int(actual_posinf.sum().item())
    acc["expected_posinf"] += int(expected_posinf.sum().item())
    acc["actual_neginf"] += int(actual_neginf.sum().item())
    acc["expected_neginf"] += int(expected_neginf.sum().item())
    acc["nan_mismatch"] += int((actual_nan != expected_nan).sum().item())
    acc["posinf_mismatch"] += int((actual_posinf != expected_posinf).sum().item())
    acc["neginf_mismatch"] += int((actual_neginf != expected_neginf).sum().item())

    finite_mask = torch.isfinite(actual_cpu) & torch.isfinite(expected_cpu)
    acc["special_count"] += int((~finite_mask).sum().item())
    if not finite_mask.any():
        return

    actual_f = actual_cpu[finite_mask]
    expected_f = expected_cpu[finite_mask]
    diff = (actual_f - expected_f).abs()
    rel = diff / (expected_f.abs() + EPS)
    acc["count"] += int(diff.numel())
    acc["sum_abs"] += float(diff.sum().item())
    acc["max_abs"] = max(acc["max_abs"], float(diff.max().item()))
    acc["sum_rel"] += float(rel.sum().item())
    acc["max_rel"] = max(acc["max_rel"], float(rel.max().item()))
    acc["sum_sq"] += float((diff * diff).sum().item())
    err_mask = diff > (err_atol + err_rtol * expected_f.abs())
    acc["err_count"] += int(err_mask.sum().item())


def _finalize_error_acc(acc: dict) -> dict:
    count = int(acc["count"])
    special_equal = (
        acc["nan_mismatch"] == 0
        and acc["posinf_mismatch"] == 0
        and acc["neginf_mismatch"] == 0
    )
    return {
        "count": count,
        "max_abs": acc["max_abs"],
        "mean_abs": acc["sum_abs"] / count if count else 0.0,
        "MARE": acc["max_rel"],
        "MERE": acc["sum_rel"] / count if count else 0.0,
        "RMSE": (acc["sum_sq"] / count) ** 0.5 if count else 0.0,
        "ERR_COUNT": int(acc["err_count"]),
        "special_count": int(acc["special_count"]),
        "special_equal": special_equal,
        "actual_nan": int(acc["actual_nan"]),
        "expected_nan": int(acc["expected_nan"]),
        "actual_posinf": int(acc["actual_posinf"]),
        "expected_posinf": int(acc["expected_posinf"]),
        "actual_neginf": int(acc["actual_neginf"]),
        "expected_neginf": int(acc["expected_neginf"]),
        "nan_mismatch": int(acc["nan_mismatch"]),
        "posinf_mismatch": int(acc["posinf_mismatch"]),
        "neginf_mismatch": int(acc["neginf_mismatch"]),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= RATIO_DENOM_EPS:
        return 0.0 if abs(numerator) <= RATIO_DENOM_EPS else float("inf")
    return numerator / denominator


def _safe_count_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def _ratio_summary(actual_metrics: dict, benchmark_metrics: dict) -> dict:
    ratios = {
        "MARE_ratio": _safe_ratio(actual_metrics["MARE"], benchmark_metrics["MARE"]),
        "MERE_ratio": _safe_ratio(actual_metrics["MERE"], benchmark_metrics["MERE"]),
        "RMSE_ratio": _safe_ratio(actual_metrics["RMSE"], benchmark_metrics["RMSE"]),
        "ERR_COUNT_ratio": _safe_count_ratio(actual_metrics["ERR_COUNT"], benchmark_metrics["ERR_COUNT"]),
    }
    checks = {
        "MARE_ratio": ratios["MARE_ratio"] <= RATIO_MARE_THRESHOLD,
        "MERE_ratio": ratios["MERE_ratio"] <= RATIO_MERE_THRESHOLD,
        "RMSE_ratio": ratios["RMSE_ratio"] <= RATIO_RMSE_THRESHOLD,
        "ERR_COUNT_ratio": ratios["ERR_COUNT_ratio"] <= RATIO_ERR_COUNT_THRESHOLD,
    }
    return {
        "actual": actual_metrics,
        "benchmark": benchmark_metrics,
        "ratios": ratios,
        "thresholds": {
            "MARE_ratio": RATIO_MARE_THRESHOLD,
            "MERE_ratio": RATIO_MERE_THRESHOLD,
            "RMSE_ratio": RATIO_RMSE_THRESHOLD,
            "ERR_COUNT_ratio": RATIO_ERR_COUNT_THRESHOLD,
        },
        "checks": checks,
        "pass": all(checks.values()) and actual_metrics["special_equal"] and benchmark_metrics["special_equal"],
    }


def _reference_chunk_ranges(
    total_t: int,
    chunk_size: int,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
):
    if cu_seqlens is None:
        for start in range(0, total_t, chunk_size):
            yield start, min(start + chunk_size, total_t)
        return
    if chunk_indices is None or len(chunk_indices) % 2 != 0:
        raise ValueError("chunk_indices must be a flat even-length list")
    for idx in range(0, len(chunk_indices), 2):
        seq_idx = int(chunk_indices[idx])
        local_chunk = int(chunk_indices[idx + 1])
        bos = int(cu_seqlens[seq_idx])
        eos = int(cu_seqlens[seq_idx + 1])
        start = bos + local_chunk * chunk_size
        end = min(start + chunk_size, eos)
        if start < end:
            yield start, end


def _kkt_head_indices(k_heads: int, gate_heads: int, out_heads: int, start: int, end: int) -> torch.Tensor:
    if out_heads == k_heads:
        return torch.arange(start, end, dtype=torch.long)
    if out_heads == gate_heads:
        if gate_heads % k_heads != 0:
            raise ValueError(f"GVA requires gate_heads divisible by k_heads, got {gate_heads}/{k_heads}")
        return torch.arange(start, end, dtype=torch.long) // (gate_heads // k_heads)
    raise ValueError(f"Unsupported KKT output heads: out={out_heads}, k={k_heads}, gate={gate_heads}")


def _dual_kkt_ratio_summary(
    actual: torch.Tensor,
    benchmark: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: list[int] | None,
    chunk_indices: list[int] | None,
    head_batch: int = 16,
) -> dict:
    k_ref = k.detach().float()
    g_ref = g.detach().float()
    beta_ref = beta.detach().float()
    actual_head_first = actual.detach()
    benchmark_head_first = benchmark.detach()
    bsz, k_heads, total_t, _ = k_ref.shape
    gate_heads = g_ref.shape[1]
    out_heads = actual_head_first.shape[1]
    actual_acc = _new_error_acc()
    benchmark_acc = _new_error_acc()
    mask_cache: dict[int, torch.Tensor] = {}
    device = k_ref.device

    for chunk_start, chunk_end in _reference_chunk_ranges(total_t, chunk_size, cu_seqlens, chunk_indices):
        valid = chunk_end - chunk_start
        if valid not in mask_cache:
            mask_cache[valid] = torch.tril(
                torch.ones((valid, valid), dtype=torch.bool, device=device),
                diagonal=-1,
            )
        mask = mask_cache[valid]
        for h0 in range(0, out_heads, head_batch):
            h1 = min(h0 + head_batch, out_heads)
            k_indices = _kkt_head_indices(k_heads, gate_heads, out_heads, h0, h1).to(device)
            k_block = k_ref[:, k_indices, chunk_start:chunk_end, :]
            g_block = g_ref[:, h0:h1, chunk_start:chunk_end]
            beta_block = beta_ref[:, h0:h1, chunk_start:chunk_end]
            scores = torch.matmul(k_block, k_block.transpose(-1, -2))
            gate = torch.exp(torch.clamp(g_block[..., :, None] - g_block[..., None, :], -50.0, 50.0))
            scaled = scores * gate * beta_block[..., :, None]
            ref = torch.zeros((bsz, h1 - h0, valid, chunk_size), dtype=torch.float32, device=device)
            ref[:, :, :, :valid] = torch.where(mask, scaled, torch.zeros_like(scaled))
            _update_error_acc(actual_acc, actual_head_first[:, h0:h1, chunk_start:chunk_end, :], ref)
            _update_error_acc(benchmark_acc, benchmark_head_first[:, h0:h1, chunk_start:chunk_end, :], ref)

    return _ratio_summary(_finalize_error_acc(actual_acc), _finalize_error_acc(benchmark_acc))


def _direct_tensor_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    acc = _new_error_acc()
    _update_error_acc(acc, actual, expected)
    return _finalize_error_acc(acc)


def _full_a_varlen_mask(case: Case, metadata) -> torch.Tensor | None:
    cu_seqlens, _, chunk_indices, _ = metadata
    if cu_seqlens is None or chunk_indices is None:
        return None
    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    chunk_tensor = fg._chunk_tensor(chunk_indices, case.chunk_size)
    if chunk_tensor is None:
        return None
    pairs = chunk_tensor.detach().cpu().view(-1, 2).tolist()
    mask = torch.zeros((1, 1, case.tokens, case.chunk_size), dtype=torch.bool)
    for seq_idx, chunk_idx in pairs:
        start = cu[int(seq_idx)] + int(chunk_idx) * case.chunk_size
        end = min(start + case.chunk_size, cu[int(seq_idx) + 1])
        for token in range(start, end):
            valid_cols = token - start
            if valid_cols > 0:
                mask[:, :, token, :valid_cols] = True
    return mask


def run_preprocess_compare(case: Case, inputs: dict[str, torch.Tensor], metadata) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens, _, chunk_indices, _ = metadata
    chunk_tensor = fg._chunk_tensor(chunk_indices, case.chunk_size)

    g_input_t = inputs["g"].clone()
    g_input_a = inputs["g"].transpose(1, 2).contiguous()
    g_before_t = g_input_t.clone()
    g_before_a = g_input_a.clone()
    k_before = inputs["k"].clone()
    beta_input_a = inputs["beta"].transpose(1, 2).contiguous().float()
    beta_before = beta_input_a.clone()

    g_ascendc = fg.chunk_local_cumsum_ascendc(
        g_input_a,
        chunk_size=case.chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=True,
    )
    torch.npu.synchronize()
    g_triton = triton_chunk_local_cumsum(
        g_input_t,
        chunk_size=case.chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )
    torch.npu.synchronize()
    g_triton_head_first = g_triton.transpose(1, 2).contiguous()

    _metric("single_cumsum", g_ascendc, g_triton_head_first, 5e-4, 5e-5)
    _assert_unchanged("cumsum_input_triton", g_before_t, g_input_t)
    _assert_unchanged("cumsum_input_ascendc", g_before_a, g_input_a)

    A_ascendc_head_first = fg.chunk_scaled_dot_kkt_fwd_ascendc(
        k=inputs["k"],
        g=g_ascendc,
        beta=beta_input_a,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_tensor,
        chunk_size=case.chunk_size,
        output_dtype=torch.float32,
    )
    A_ascendc = A_ascendc_head_first.transpose(1, 2).contiguous()
    torch.npu.synchronize()
    A_triton = triton_chunk_scaled_dot_kkt_fwd(
        k=inputs["k"],
        g=g_triton,
        beta=inputs["beta"],
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_tensor,
        chunk_size=case.chunk_size,
        output_dtype=torch.float32,
    )
    torch.npu.synchronize()

    _metric("single_kkt", A_ascendc, A_triton, 2e-2, 2e-3)
    _metric("cumsum_to_kkt_g", g_ascendc, g_triton_head_first, 5e-4, 5e-5)
    _metric("cumsum_to_kkt_A", A_ascendc, A_triton, 2e-2, 2e-3)
    _assert_unchanged("kkt_input_k", k_before, inputs["k"])
    _assert_unchanged("kkt_input_beta", beta_before, beta_input_a)
    return g_triton, A_triton


def flash_chunk_gated_delta_rule_fwd_triton_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_list: list[int] | None,
    chunk_indices,
    chunk_indices_list,
    chunk_size: int,
):
    g = triton_chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )
    torch.npu.synchronize()
    A = triton_chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=fg._chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    torch.npu.synchronize()
    A = fg.solve_tri_auto(
        A,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices_list=fg._chunk_list(chunk_indices_list, chunk_size),
        output_dtype=k.dtype,
    )
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()

    w, u = fg.recompute_w_u(
        k,
        v,
        beta,
        A,
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=fg._chunk_list(chunk_indices_list, chunk_size),
    )
    h, v_new, _ = fg.ascendc_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=fg._chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    o = fg.ascendc_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=fg._chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    return g.transpose(1, 2).contiguous(), o.transpose(1, 2).contiguous(), A, None


def run_full_forward_compare(case: Case, inputs: dict[str, torch.Tensor], metadata) -> None:
    cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list = metadata
    scale = case.key_dim ** -0.5

    baseline_inputs = _clone_inputs(inputs)
    actual_inputs = _clone_inputs(inputs)

    actual = fg.flash_chunk_gated_delta_rule_fwd(
        q=actual_inputs["q"],
        k=actual_inputs["k"],
        v=actual_inputs["v"],
        g=actual_inputs["g"],
        beta=actual_inputs["beta"],
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
    )
    torch.npu.synchronize()
    expected = flash_chunk_gated_delta_rule_fwd_triton_preprocess(
        q=baseline_inputs["q"],
        k=baseline_inputs["k"],
        v=baseline_inputs["v"],
        g=baseline_inputs["g"],
        beta=baseline_inputs["beta"],
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
    )
    torch.npu.synchronize()

    full_a_mask = _full_a_varlen_mask(case, metadata)
    for name, actual_tensor, expected_tensor in zip(("full_g", "full_o", "full_A"), actual[:3], expected[:3]):
        max_tol, mean_tol = (5e-4, 5e-5) if name == "full_g" else (5e-2, 5e-3)
        metric_mask = full_a_mask if name == "full_A" else None
        _metric(name, actual_tensor, expected_tensor, max_tol, mean_tol, mask=metric_mask)
    if actual[3] is not None or expected[3] is not None:
        raise AssertionError("final_state should be None when output_final_state=False")


def analyze_single_kkt_precision(case: Case, device: torch.device, dtype: torch.dtype, seed: int) -> dict:
    inputs = make_inputs(case, device, dtype, seed)
    metadata = make_metadata(case, inputs["g"])
    cu_seqlens, _, chunk_indices, _ = metadata
    chunk_tensor = fg._chunk_tensor(chunk_indices, case.chunk_size)
    cu_list = fg._as_int_list(cu_seqlens)
    chunk_list = fg._as_int_list(chunk_tensor)

    g_ascendc = fg.chunk_local_cumsum_ascendc(
        inputs["g"].transpose(1, 2).contiguous(),
        chunk_size=case.chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=True,
    )
    torch.npu.synchronize()
    g_triton = triton_chunk_local_cumsum(
        inputs["g"].clone(),
        chunk_size=case.chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )
    torch.npu.synchronize()
    beta_head_first = inputs["beta"].transpose(1, 2).contiguous().float()
    A_ascendc = fg.chunk_scaled_dot_kkt_fwd_ascendc(
        k=inputs["k"],
        g=g_ascendc,
        beta=beta_head_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_tensor,
        chunk_size=case.chunk_size,
        output_dtype=torch.float32,
    )
    torch.npu.synchronize()
    A_triton = triton_chunk_scaled_dot_kkt_fwd(
        k=inputs["k"],
        g=g_triton,
        beta=inputs["beta"],
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_tensor,
        chunk_size=case.chunk_size,
        output_dtype=torch.float32,
    ).transpose(1, 2).contiguous()
    torch.npu.synchronize()

    dual = _dual_kkt_ratio_summary(
        A_ascendc,
        A_triton,
        inputs["k"],
        g_ascendc,
        beta_head_first,
        chunk_size=case.chunk_size,
        cu_seqlens=cu_list,
        chunk_indices=chunk_list,
    )
    verdict = "NO_PRECISION_ISSUE" if dual["pass"] else "PRECISION_ISSUE"
    return {
        "stage": "single_kkt",
        "precision_verdict": verdict,
        "failure_reason": (
            "AscendC KKT and Triton KKT differ in direct allclose; dual-reference ratio uses "
            "CPU fp32 KKT as golden and Triton as benchmark."
        ),
        "dual": dual,
    }


def analyze_full_o_mismatch(case: Case, device: torch.device, dtype: torch.dtype, seed: int) -> dict:
    inputs = make_inputs(case, device, dtype, seed)
    metadata = make_metadata(case, inputs["g"])
    cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list = metadata
    scale = case.key_dim ** -0.5
    actual_inputs = _clone_inputs(inputs)
    baseline_inputs = _clone_inputs(inputs)
    actual = fg.flash_chunk_gated_delta_rule_fwd(
        q=actual_inputs["q"],
        k=actual_inputs["k"],
        v=actual_inputs["v"],
        g=actual_inputs["g"],
        beta=actual_inputs["beta"],
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
    )
    torch.npu.synchronize()
    expected = flash_chunk_gated_delta_rule_fwd_triton_preprocess(
        q=baseline_inputs["q"],
        k=baseline_inputs["k"],
        v=baseline_inputs["v"],
        g=baseline_inputs["g"],
        beta=baseline_inputs["beta"],
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_list=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
    )
    torch.npu.synchronize()
    direct = _direct_tensor_summary(actual[1], expected[1])
    finite_pass = direct["max_abs"] <= 5e-2 and direct["mean_abs"] <= 5e-3
    if finite_pass and not direct["special_equal"]:
        verdict = "SPECIAL_VALUE_MISMATCH"
    elif finite_pass:
        verdict = "NO_PRECISION_ISSUE"
    else:
        verdict = "PRECISION_ISSUE"
    return {
        "stage": "full_o",
        "precision_verdict": verdict,
        "failure_reason": (
            "full_o has no independent CPU dual-reference path in this script; "
            "finite values are compared directly and special-value masks are reported separately."
        ),
        "direct": direct,
    }


def analyze_failed_case(case: Case, device: torch.device, dtype: torch.dtype, seed: int, reason: str) -> dict:
    try:
        if "single_kkt mismatch" in reason:
            return analyze_single_kkt_precision(case, device, dtype, seed)
        if "full_o mismatch" in reason:
            return analyze_full_o_mismatch(case, device, dtype, seed)
        return {
            "stage": "unknown",
            "precision_verdict": "UNCLASSIFIED",
            "failure_reason": "No failure-specific ratio analyzer is available.",
        }
    except Exception as exc:
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        return {
            "stage": "analysis_error",
            "precision_verdict": "ANALYSIS_ERROR",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def run_case(case: Case, device: torch.device, dtype: torch.dtype, seed: int) -> None:
    print(
        f"[CASE] {case.case_id} original={case.original_id} "
        f"B={case.batch} VH={case.value_heads} KH={case.key_heads} T={case.tokens} "
        f"V={case.value_dim} K={case.key_dim} chunk={case.chunk_size} varlen={case.is_varlen}"
    )
    inputs = make_inputs(case, device, dtype, seed)
    metadata = make_metadata(case, inputs["g"])
    run_preprocess_compare(case, inputs, metadata)
    run_full_forward_compare(case, inputs, metadata)
    del inputs
    gc.collect()
    torch.npu.empty_cache()
    print(f"[PASS] {case.case_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path("/home/m00913889/codex04/bingli/test_case.xlsx"),
        help="Path to test_case.xlsx.",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Run every case from the xlsx file.")
    parser.add_argument("--continue-on-error", action="store_true", help="Run remaining cases after a case fails.")
    parser.add_argument(
        "--max-input-gib",
        type=float,
        default=0.0,
        help="Skip cases whose estimated input tensors exceed this GiB value. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-triton-unsupported",
        action="store_true",
        help="Skip cases known to fail in the Triton baseline rather than in the AscendC replacement.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/flash_chunk_gated_delta_rule_fwd_ascendc_report.md"),
        help="Markdown report path. A JSON file with the same basename is written as well.",
    )
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args(argv)

    if not torch.npu.is_available():
        raise RuntimeError("NPU device is not available")
    require_kkt_schema()
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device_id)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(f"npu:{args.device_id}")
    loaded_cases = load_cases(args.xlsx)
    cases = iter_all_cases(loaded_cases, args.limit) if args.all else iter_selected_cases(loaded_cases, SELECTED_CASE_IDS, args.limit)
    print("selected cases:", ", ".join(case.case_id for case in cases))
    failures: list[tuple[str, str]] = []
    records: list[dict] = []
    attempted = 0
    for idx, case in enumerate(cases):
        estimate = estimated_input_gib(case)
        if args.max_input_gib > 0 and estimate > args.max_input_gib:
            reason = f"estimated input tensors {estimate:.2f} GiB > max-input-gib {args.max_input_gib:.2f}"
            print(f"[SKIP] {case.case_id}: {reason}")
            records.append({"case": case, "status": "SKIP", "estimated_input_gib": estimate, "reason": reason})
            continue
        triton_reason = triton_unsupported_reason(case) if args.skip_triton_unsupported else ""
        if triton_reason:
            print(f"[SKIP] {case.case_id}: {triton_reason}")
            records.append({"case": case, "status": "SKIP", "estimated_input_gib": estimate, "reason": triton_reason})
            continue
        attempted += 1
        try:
            run_case(case, device, dtype, args.seed + idx)
            records.append({"case": case, "status": "PASS", "estimated_input_gib": estimate, "reason": ""})
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append((case.case_id, reason))
            print(f"[FAIL] {case.case_id}: {reason}")
            print(f"[ANALYZE] {case.case_id}: running failure precision reassessment")
            analysis = analyze_failed_case(case, device, dtype, args.seed + idx, reason)
            print(f"[ANALYZE] {case.case_id}: {analysis.get('precision_verdict')} {_analysis_brief(analysis)}")
            records.append(
                {
                    "case": case,
                    "status": "FAIL",
                    "estimated_input_gib": estimate,
                    "reason": reason,
                    "analysis": analysis,
                }
            )
            try:
                torch.npu.synchronize()
            except Exception:
                pass
            gc.collect()
            torch.npu.empty_cache()
            if not args.continue_on_error:
                break
    passed = attempted - len(failures)
    print(f"Results: {passed}/{attempted} attempted, {len(cases)} selected")
    write_report(args.report, records, args)
    print(f"Report: {args.report}")
    print(f"Report JSON: {args.report.with_suffix('.json')}")
    if failures:
        print("Failures:")
        for case_id, reason in failures:
            print(f"  {case_id}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
