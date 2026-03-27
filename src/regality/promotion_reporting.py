"""Recurring promotion-evidence reporting for regal and advisor authority."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.economics.inferential_contract import summarize_inferential_learnability_contracts
from src.learning.calibration import summarize_calibration
from src.replay.dataset import ReplayDatasetBundle
from src.regality.promotion_policy import (
    PromotionMetrics,
    RegalPromotionPolicy,
)
from src.replay.receipt_ingest import ReceiptLabelBundle
from src.utils.config_digest import sha256_json


DEFAULT_PROMOTION_NODE_IDS = (
    "objective_integrity_regal",
    "plausibility_regal",
    "reward_safety_regal",
    "pricing_truth_regal",
    "data_value_regal",
)


@dataclass(frozen=True)
class PromotionEvidenceRecord:
    """Per-node promotion evidence with recurring artifacts."""

    node_id: str
    policy_node_id: str
    current_stage: str
    coverage: Dict[str, Any]
    metrics: Dict[str, Any]
    promotion_decision: Dict[str, Any]
    disagreement_slices: Dict[str, Any] = field(default_factory=dict)
    disagreement_episode_ids: list[str] = field(default_factory=list)
    false_positive_summary: Dict[str, Any] = field(default_factory=dict)
    false_positive_episode_ids: list[str] = field(default_factory=list)
    false_negative_summary: Dict[str, Any] = field(default_factory=dict)
    false_negative_episode_ids: list[str] = field(default_factory=list)
    downstream_usefulness: Dict[str, Any] = field(default_factory=dict)
    evidence_pointers: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "policy_node_id": self.policy_node_id,
            "current_stage": self.current_stage,
            "coverage": dict(self.coverage),
            "metrics": dict(self.metrics),
            "promotion_decision": dict(self.promotion_decision),
            "disagreement_slices": dict(self.disagreement_slices),
            "disagreement_episode_ids": list(self.disagreement_episode_ids),
            "false_positive_summary": dict(self.false_positive_summary),
            "false_positive_episode_ids": list(self.false_positive_episode_ids),
            "false_negative_summary": dict(self.false_negative_summary),
            "false_negative_episode_ids": list(self.false_negative_episode_ids),
            "downstream_usefulness": dict(self.downstream_usefulness),
            "evidence_pointers": dict(self.evidence_pointers),
        }


@dataclass(frozen=True)
class PromotionEvidenceReport:
    """Complete recurring promotion-evidence artifact."""

    schema_version: str
    policy_name: str
    config_digest: str
    dataset_digest: str
    receipt_label_coverage: Dict[str, Any]
    node_reports: list[PromotionEvidenceRecord]
    summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def report_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_name": self.policy_name,
            "config_digest": self.config_digest,
            "dataset_digest": self.dataset_digest,
            "receipt_label_coverage": dict(self.receipt_label_coverage),
            "node_reports": [row.to_dict() for row in self.node_reports],
            "summary": dict(self.summary),
            "metadata": dict(self.metadata),
        }


def build_promotion_evidence_report(
    *,
    dataset: ReplayDatasetBundle,
    promotion_policy: RegalPromotionPolicy,
    receipt_bundle: Optional[ReceiptLabelBundle] = None,
    node_ids: Sequence[str] = DEFAULT_PROMOTION_NODE_IDS,
    work_orders: Optional[Sequence[Mapping[str, Any]]] = None,
    evidence_pointers: Optional[Mapping[str, Any]] = None,
) -> PromotionEvidenceReport:
    deployment_by_episode = {
        row.episode_id: row for row in (receipt_bundle.deployment_outcomes if receipt_bundle else [])
    }
    receipt_by_episode = {
        row.episode_id: row for row in (receipt_bundle.deployment_receipts if receipt_bundle else [])
    }
    datapack_by_episode = {
        str(row.metadata.get("episode_id", "")): row
        for row in (receipt_bundle.datapack_contributions if receipt_bundle else [])
    }
    adaptation_by_episode = {
        str(row.metadata.get("episode_id", "")): row
        for row in (receipt_bundle.adaptation_outcomes if receipt_bundle else [])
    }
    work_orders_by_episode: Dict[str, list[Dict[str, Any]]] = {}
    for row in list(work_orders or []):
        episode_id = str(row.get("subject_id", row.get("episode_id", "")) or "")
        if not episode_id:
            continue
        work_orders_by_episode.setdefault(episode_id, []).append(dict(row))
    trace_preconditions_by_episode = {
        episode.episode_id: dict(episode.metadata.get("execution_preconditions", {}) or {})
        for episode in dataset.episodes
    }

    node_reports: list[PromotionEvidenceRecord] = []
    for node_id in node_ids:
        policy_node_id = _policy_node_id(node_id=node_id, policy=promotion_policy)
        predictions: list[float] = []
        outcomes: list[float] = []
        confidences: list[float] = []
        baselines: list[float] = []
        monotonic_inputs: list[float] = []
        monotonic_outputs: list[float] = []
        reference_vectors: list[list[float]] = []
        current_vectors: list[list[float]] = []
        disagreement_episode_ids: list[str] = []
        false_positive_episode_ids: list[str] = []
        false_negative_episode_ids: list[str] = []
        disagreement_source_counts: Dict[str, int] = {}
        disagreement_skill_counts: Dict[str, int] = {}
        false_positive_source_counts: Dict[str, int] = {}
        false_negative_source_counts: Dict[str, int] = {}
        trace_ready_count = 0
        counterfactual_count = 0
        value_target_count = 0
        executable_work_order_count = 0

        for episode in dataset.episodes:
            deployment_label = deployment_by_episode.get(episode.episode_id)
            receipt = receipt_by_episode.get(episode.episode_id)
            datapack_label = datapack_by_episode.get(episode.episode_id)
            adaptation_label = adaptation_by_episode.get(episode.episode_id)
            trace_preconditions = trace_preconditions_by_episode.get(episode.episode_id, {})
            if bool(trace_preconditions.get("ready", False)):
                trace_ready_count += 1
            episode_work_orders = work_orders_by_episode.get(episode.episode_id, [])
            executable_work_order_count += sum(1 for row in episode_work_orders if row.get("ready"))
            if _episode_has_ref(episode, "counterfactual_eval_ref", "counterfactual_eval_path"):
                counterfactual_count += 1
            if _episode_has_ref(episode, "value_target_pack_ref", "value_target_pack_path", "value_target_refs"):
                value_target_count += 1
            prediction, confidence, baseline, outcome = _node_signals(
                node_id=node_id,
                episode=episode,
                deployment_label=deployment_label,
                receipt=receipt,
                datapack_label=datapack_label,
                adaptation_label=adaptation_label,
            )
            predictions.append(prediction)
            confidences.append(confidence)
            baselines.append(baseline)
            outcomes.append(outcome)
            monotonic_inputs.append(float(episode.total_steps))
            monotonic_outputs.append(confidence)
            reference_vectors.append(list(episode.condition_vector_values))
            current_vectors.append(list(episode.condition_vector_values))
            if abs(prediction - baseline) > 0.25:
                disagreement_episode_ids.append(episode.episode_id)
                disagreement_source_counts[episode.source_domain] = disagreement_source_counts.get(episode.source_domain, 0) + 1
                disagreement_skill_counts[episode.skill_mode] = disagreement_skill_counts.get(episode.skill_mode, 0) + 1
            if prediction >= 0.5 and outcome < 0.5:
                false_positive_episode_ids.append(episode.episode_id)
                false_positive_source_counts[episode.source_domain] = false_positive_source_counts.get(episode.source_domain, 0) + 1
            if prediction < 0.5 and outcome >= 0.5:
                false_negative_episode_ids.append(episode.episode_id)
                false_negative_source_counts[episode.source_domain] = false_negative_source_counts.get(episode.source_domain, 0) + 1

        calibration = summarize_calibration(
            confidences=confidences,
            outcomes=outcomes,
            predictions=predictions,
            targets=baselines,
            monotonic_inputs=monotonic_inputs,
            monotonic_outputs=monotonic_outputs,
            reference_vectors=reference_vectors,
            current_vectors=current_vectors,
            metadata={"node_id": node_id, "policy_node_id": policy_node_id},
        )
        agreement = sum(
            1 for prediction, baseline in zip(predictions, baselines)
            if abs(prediction - baseline) <= 0.25
        ) / float(max(1, len(predictions)))
        metrics = PromotionMetrics(
            replay_coverage=min(1.0, float(len(dataset.episodes)) / 10.0 + 0.4),
            downstream_label_count=(
                receipt_bundle.coverage_summary().get("covered_episode_count", 0)
                if receipt_bundle
                else len(dataset.episodes)
            ),
            deployment_receipt_count=len(receipt_bundle.deployment_receipts) if receipt_bundle else 0,
            calibration_error=calibration.expected_calibration_error,
            baseline_agreement=agreement,
            monotonicity=calibration.monotonicity_score,
            sign_consistency=calibration.sign_consistency,
            false_positive_rate=len(false_positive_episode_ids) / float(max(1, len(predictions))),
            false_negative_rate=len(false_negative_episode_ids) / float(max(1, len(predictions))),
            drift_score=calibration.drift_score,
            residual_gain=_residual_gain(node_id=node_id, predictions=predictions, outcomes=outcomes, baselines=baselines),
            calibration_summary=calibration,
            metadata={
                "node_id": node_id,
                "policy_node_id": policy_node_id,
                "trace_ready_rate": trace_ready_count / float(max(1, len(dataset.episodes))),
                "counterfactual_coverage_rate": counterfactual_count / float(max(1, len(dataset.episodes))),
                "value_target_coverage_rate": value_target_count / float(max(1, len(dataset.episodes))),
            },
        )
        decision = promotion_policy.evaluate_node(
            policy_node_id,
            metrics,
            evidence_pointers=dict(evidence_pointers or {}),
        )
        node_reports.append(
            PromotionEvidenceRecord(
                node_id=node_id,
                policy_node_id=policy_node_id,
                current_stage=promotion_policy.node_stage(policy_node_id).value,
                coverage={
                    "episode_count": len(predictions),
                    "receipt_coverage_count": (
                        receipt_bundle.coverage_summary().get("covered_episode_count", 0)
                        if receipt_bundle
                        else 0
                    ),
                    "trace_ready_episode_count": trace_ready_count,
                    "trace_ready_rate": round(trace_ready_count / float(max(1, len(predictions))), 6),
                    "counterfactual_episode_count": counterfactual_count,
                    "value_target_episode_count": value_target_count,
                    "source_domains": sorted({episode.source_domain for episode in dataset.episodes}),
                },
                metrics=metrics.to_dict(),
                promotion_decision=decision.to_dict(),
                disagreement_slices={
                    "by_source_domain": dict(sorted(disagreement_source_counts.items())),
                    "by_skill_mode": dict(sorted(disagreement_skill_counts.items())),
                },
                disagreement_episode_ids=sorted(disagreement_episode_ids),
                false_positive_summary={
                    "count": len(false_positive_episode_ids),
                    "by_source_domain": dict(sorted(false_positive_source_counts.items())),
                },
                false_positive_episode_ids=sorted(false_positive_episode_ids),
                false_negative_summary={
                    "count": len(false_negative_episode_ids),
                    "by_source_domain": dict(sorted(false_negative_source_counts.items())),
                },
                false_negative_episode_ids=sorted(false_negative_episode_ids),
                downstream_usefulness={
                    "precision_if_acted": round(
                        1.0 - (len(false_positive_episode_ids) / float(max(1, len(predictions)))),
                        6,
                    ),
                    "recall_if_acted": round(
                        1.0 - (len(false_negative_episode_ids) / float(max(1, len(predictions)))),
                        6,
                    ),
                    "disagreement_rate": round(
                        len(disagreement_episode_ids) / float(max(1, len(predictions))),
                        6,
                    ),
                    "executable_work_orders": executable_work_order_count,
                },
                evidence_pointers={
                    **dict(evidence_pointers or {}),
                    "trace_preconditions": trace_preconditions_by_episode,
                    "work_order_ids": sorted(
                        {
                            str(row.get("work_order_id", ""))
                            for rows in work_orders_by_episode.values()
                            for row in rows
                            if row.get("work_order_id")
                        }
                    ),
                },
            )
        )

    summary = {
        "recommend_promote": sum(
            1 for report in node_reports if report.promotion_decision.get("outcome") == "recommend_promote"
        ),
        "recommend_hold": sum(
            1 for report in node_reports if report.promotion_decision.get("outcome") == "recommend_hold"
        ),
        "recommend_demote": sum(
            1 for report in node_reports if report.promotion_decision.get("outcome") == "recommend_demote"
        ),
        "node_count": len(node_reports),
        "receipt_coverage_count": (
            receipt_bundle.coverage_summary().get("covered_episode_count", 0)
            if receipt_bundle
            else 0
        ),
        "trace_ready_episode_count": sum(
            1
            for episode in dataset.episodes
            if bool(trace_preconditions_by_episode.get(episode.episode_id, {}).get("ready", False))
        ),
        "work_order_count": sum(len(rows) for rows in work_orders_by_episode.values()),
        "inferential_learnability_summary": summarize_inferential_learnability_contracts(
            [
                dict(episode.metadata.get("inferential_learnability_contract", {}) or {})
                for episode in dataset.episodes
                if episode.metadata.get("inferential_learnability_contract")
            ]
        ),
    }
    return PromotionEvidenceReport(
        schema_version="regal_promotion_evidence_v1",
        policy_name=promotion_policy.policy_name,
        config_digest=promotion_policy.config_digest,
        dataset_digest=dataset.manifest.dataset_digest,
        receipt_label_coverage=(
            receipt_bundle.coverage_summary()
            if receipt_bundle
            else {"schema_version": "none", "total_labels": 0, "covered_episode_count": 0}
        ),
        node_reports=sorted(node_reports, key=lambda row: row.node_id),
        summary=summary,
        metadata={"node_ids": list(node_ids)},
    )


def write_promotion_evidence_report(
    output_dir: str | Path,
    report: PromotionEvidenceReport,
) -> Dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "regal_promotion_eval.json"
    md_path = root / "regal_promotion_eval.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_promotion_markdown(report), encoding="utf-8")
    sidecars: Dict[str, str] = {}
    for row in report.node_reports:
        sidecar_path = root / f"promotion_evidence_{row.node_id}.json"
        sidecar_path.write_text(
            json.dumps(row.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        sidecars[row.node_id] = str(sidecar_path)
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        **{f"sidecar::{key}": value for key, value in sorted(sidecars.items())},
    }


def _policy_node_id(node_id: str, policy: RegalPromotionPolicy) -> str:
    if node_id in policy.nodes:
        return node_id
    aliases = {
        "pricing_advisor": "pricing_truth_regal",
        "data_value_advisor": "data_value_regal",
        "regal_support_advisor": "plausibility_regal",
        "policy_advisor": "reward_safety_regal",
    }
    return aliases.get(node_id, next(iter(policy.nodes)))


def _residual_gain(
    *,
    node_id: str,
    predictions: Sequence[float],
    outcomes: Sequence[float],
    baselines: Sequence[float],
) -> float:
    prediction_error = sum(abs(prediction - outcome) for prediction, outcome in zip(predictions, outcomes))
    baseline_error = sum(abs(baseline - outcome) for baseline, outcome in zip(baselines, outcomes))
    if not predictions:
        return 0.0
    return round((baseline_error - prediction_error) / float(len(predictions)), 6)


def _node_signals(
    *,
    node_id: str,
    episode,
    deployment_label,
    receipt,
    datapack_label,
    adaptation_label,
) -> tuple[float, float, float, float]:
    pricing_conf = float(episode.pricing_summary.get("confidence", 0.0) or 0.0)
    quality = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
    frontier = float(episode.datapack_summary.get("marginal_frontier_gain", 0.0) or 0.0)
    hard_flags = sum(1 for flag in episode.constraint_flags if str(flag.get("severity", "")).lower() == "hard")
    if node_id in {"objective_integrity_regal", "policy_advisor"}:
        prediction = 1.0 if episode.objective_tensor_summary.get("axes") and episode.metadata.get("objective_profile_id") else 0.5
        confidence = 0.95 if prediction > 0.9 else 0.35
        baseline = 1.0 if episode.regal_summary.get("overall_status", "pass") != "fail" else 0.0
        outcome = (
            1.0
            if deployment_label and deployment_label.metadata.get("objective_satisfied", False)
            else (1.0 if prediction > 0.9 else 0.0)
        )
    elif node_id in {"pricing_truth_regal", "pricing_advisor"}:
        prediction = min(1.0, max(0.0, pricing_conf + 0.05))
        confidence = pricing_conf
        baseline = 1.0 if episode.regal_summary.get("pricing_recommendation", "publish") != "suppress" else 0.0
        if receipt is not None:
            outcome = 1.0 if receipt.pricing_acceptance.accepted else 0.0
        else:
            outcome = 1.0 if float(episode.pricing_summary.get("net_customer_rate", 0.0) or 0.0) >= 0.0 else 0.0
    elif node_id in {"data_value_regal", "data_value_advisor"}:
        prediction = min(1.0, max(0.0, 0.45 + frontier + 0.2 * quality))
        confidence = min(1.0, max(0.0, quality))
        baseline = 1.0 if episode.regal_summary.get("datapack_recommendation", "keep") != "downweight" else 0.0
        if datapack_label is not None:
            outcome = 1.0 if datapack_label.marginal_frontier_gain_realized > 0.0 else 0.0
        else:
            outcome = 1.0 if frontier > 0.0 or quality > 0.5 else 0.0
    elif node_id in {"reward_safety_regal"}:
        prediction = 1.0 if float(episode.total_reward) >= 0.0 else 0.0
        confidence = 0.85 if hard_flags == 0 else 0.35
        baseline = 1.0 if episode.regal_summary.get("deploy_recommendation", "allow_shadow") == "allow_shadow" else 0.0
        if deployment_label is not None:
            outcome = 1.0 if not deployment_label.risk_events and not deployment_label.failure_events else 0.0
        else:
            outcome = 1.0 if hard_flags == 0 else 0.0
    else:
        prediction = 1.0 if quality > 0.4 and hard_flags == 0 else 0.25
        confidence = 0.8 if quality > 0.4 else 0.3
        baseline = 1.0 if episode.regal_summary.get("overall_status", "pass") == "pass" else 0.0
        if deployment_label is not None and receipt is not None:
            realized_ratio = 0.0
            if abs(receipt.predicted_value) > 1e-6:
                realized_ratio = receipt.realized_value / receipt.predicted_value
            outcome = 1.0 if not deployment_label.failure_events and 0.25 <= realized_ratio <= 1.75 else 0.0
        else:
            outcome = 1.0 if hard_flags == 0 and quality > 0.4 else 0.0
    if adaptation_label is not None and node_id in {"reward_safety_regal", "plausibility_regal"}:
        outcome = min(outcome, 1.0 if not adaptation_label.review_required else 0.0)
    return float(prediction), float(confidence), float(baseline), float(outcome)


def _promotion_markdown(report: PromotionEvidenceReport) -> str:
    lines = [
        "# Regal Promotion Evidence",
        "",
        f"- Policy: {report.policy_name}",
        f"- Dataset digest: {report.dataset_digest}",
        f"- Receipt labels: {report.receipt_label_coverage.get('total_labels', 0)}",
        f"- Recommend promote: {report.summary.get('recommend_promote', 0)}",
        f"- Recommend hold: {report.summary.get('recommend_hold', 0)}",
        f"- Recommend demote: {report.summary.get('recommend_demote', 0)}",
        "",
        "## Nodes",
    ]
    for row in report.node_reports:
        lines.extend(
            [
                f"### {row.node_id}",
                f"- Current stage: {row.current_stage}",
                f"- Recommendation: {row.promotion_decision.get('outcome')} -> {row.promotion_decision.get('recommended_stage')}",
                f"- Coverage: {row.coverage.get('episode_count', 0)} episodes / {row.coverage.get('receipt_coverage_count', 0)} receipts",
                f"- Trace-ready coverage: {row.coverage.get('trace_ready_episode_count', 0)} episodes",
                f"- Calibration error: {row.metrics.get('calibration_error')}",
                f"- Baseline agreement: {row.metrics.get('baseline_agreement')}",
                f"- Disagreements: {len(row.disagreement_episode_ids)}",
                f"- False positives: {row.false_positive_summary.get('count', 0)}",
                f"- False negatives: {row.false_negative_summary.get('count', 0)}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_PROMOTION_NODE_IDS",
    "PromotionEvidenceRecord",
    "PromotionEvidenceReport",
    "build_promotion_evidence_report",
    "write_promotion_evidence_report",
]


def _episode_has_ref(episode: Any, *keys: str) -> bool:
    for key in keys:
        if episode.metadata.get(key) not in (None, "", [], {}):
            return True
        if episode.provenance.get(key) not in (None, "", [], {}):
            return True
    return False
