"""Helper-resolution and promotion utilities for sim/synth/physics WM seams."""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping

from .common import mapping

HelperMode = Literal["disabled", "auto", "required"]


def resolve_helper(
    helper: Any,
    *,
    mode: HelperMode = "auto",
    name: str,
) -> tuple[Any | None, Dict[str, Any]]:
    if mode == "disabled":
        return None, {
            "status": "disabled",
            "mode": mode,
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
            "helper_weight": 0.0,
        }
    if helper is None:
        if mode == "required":
            raise ValueError(f"{name} helper required, but no helper was provided.")
        return None, {
            "status": "missing",
            "mode": mode,
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
            "helper_weight": 0.0,
        }

    benchmark_gate = mapping(getattr(helper, "benchmark_gate", {}))
    benchmark_ready = bool(benchmark_gate.get("ready", False))
    promotion_stage = "promoted" if benchmark_ready else "shadow_candidate"
    helper_weight = 0.7 if benchmark_ready else 0.25
    return helper, {
        "status": "available",
        "mode": mode,
        "promotion_stage": promotion_stage,
        "benchmark_gate_ready": benchmark_ready,
        "helper_weight": helper_weight,
        "benchmark_gate": benchmark_gate,
    }


def _coerce_helper_payload(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return mapping(payload)
    if hasattr(payload, "to_dict"):
        try:
            return mapping(payload.to_dict())
        except Exception:
            return {}
    if hasattr(payload, "__dict__"):
        return mapping(vars(payload))
    return {}


def infer_backend_payload(helper: Any, *, context: Mapping[str, Any]) -> Dict[str, Any]:
    if helper is None:
        return {}
    if hasattr(helper, "select_backend"):
        return _coerce_helper_payload(helper.select_backend(context=context))
    if hasattr(helper, "infer_context"):
        return _coerce_helper_payload(helper.infer_context(context=context))
    if callable(helper):
        try:
            return _coerce_helper_payload(helper(context=context))
        except TypeError:
            return _coerce_helper_payload(helper(context))
    return {}


def infer_branch_payload(
    helper: Any,
    *,
    job: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    if helper is None:
        return {}
    if hasattr(helper, "plan_branch"):
        return _coerce_helper_payload(helper.plan_branch(job=job, context=context))
    if hasattr(helper, "infer_job"):
        return _coerce_helper_payload(helper.infer_job(job=job, context=context))
    if callable(helper):
        try:
            return _coerce_helper_payload(helper(job=job, context=context))
        except TypeError:
            try:
                return _coerce_helper_payload(helper(job, context))
            except TypeError:
                return _coerce_helper_payload(helper(job))
    return {}
