"""Helper-resolution and promotion utilities for sim/synth/physics WM seams."""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping

from .common import mapping

HelperMode = Literal["disabled", "auto", "required"]


def _check_demotion(
    benchmark_gate: Dict[str, Any],
    evidence_signals: Dict[str, Any],
) -> tuple[bool, str]:
    """Check whether evidence signals warrant demotion of a promoted helper.

    Returns (should_demote, reason).
    """
    if not evidence_signals:
        return False, ""
    if bool(evidence_signals.get("benchmark_gate_revoked", False)):
        return True, "benchmark_gate_revoked"
    if bool(evidence_signals.get("evidence_failure", False)):
        return True, "evidence_failure"
    failure_rate = float(evidence_signals.get("recent_failure_rate", 0.0) or 0.0)
    failure_threshold = float(
        benchmark_gate.get("demotion_failure_threshold", 0.5) or 0.5
    )
    if failure_rate > failure_threshold:
        return True, f"failure_rate_{failure_rate:.2f}_exceeds_{failure_threshold:.2f}"
    return False, ""


def resolve_helper(
    helper: Any,
    *,
    mode: HelperMode = "auto",
    name: str,
    evidence_signals: Mapping[str, Any] | None = None,
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
    evidence = mapping(evidence_signals)

    # Demotion check: a promoted helper can be demoted back to shadow_candidate
    # if evidence signals indicate degradation.
    should_demote, demotion_reason = _check_demotion(benchmark_gate, evidence)
    if benchmark_ready and should_demote:
        return helper, {
            "status": "available",
            "mode": mode,
            "promotion_stage": "demoted_to_shadow",
            "benchmark_gate_ready": benchmark_ready,
            "helper_weight": 0.25,
            "benchmark_gate": benchmark_gate,
            "demotion_reason": demotion_reason,
            "evidence_signals": evidence,
        }

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
