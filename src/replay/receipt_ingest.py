"""Receipt-label ingestion for synthetic, simulated, and future real deployment outcomes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.economics.receipt_schema import DeploymentReceiptRecord, PricingAcceptanceLabel
from src.ontology.deployment_labels import (
    AdaptationOutcomeLabel,
    DatapackContributionLabel,
    DeploymentOutcomeLabel,
)
from src.replay.dataset import ReplayDatasetBundle
from src.utils.config_digest import sha256_json


RECEIPT_LABEL_BUNDLE_SCHEMA_VERSION = "receipt_label_bundle_v1"


@dataclass(frozen=True)
class ReceiptLabelBundle:
    """Deterministic collection of downstream outcome labels and receipts."""

    schema_version: str
    label_mode: str
    deployment_outcomes: list[DeploymentOutcomeLabel]
    adaptation_outcomes: list[AdaptationOutcomeLabel]
    datapack_contributions: list[DatapackContributionLabel]
    deployment_receipts: list[DeploymentReceiptRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bundle_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_mode": self.label_mode,
            "deployment_outcomes": [row.to_dict() for row in self.deployment_outcomes],
            "adaptation_outcomes": [row.to_dict() for row in self.adaptation_outcomes],
            "datapack_contributions": [row.to_dict() for row in self.datapack_contributions],
            "deployment_receipts": [row.to_dict() for row in self.deployment_receipts],
            "metadata": dict(self.metadata),
        }

    def coverage_summary(self) -> Dict[str, Any]:
        source_domains = sorted(
            {
                row.source_domain
                for row in self.deployment_outcomes
                if row.source_domain
            }
            | {
                row.source_domain
                for row in self.deployment_receipts
                if row.source_domain
            }
        )
        episode_ids = sorted(
            {
                row.episode_id
                for row in self.deployment_outcomes
            }
            | {
                row.episode_id
                for row in self.deployment_receipts
            }
        )
        return {
            "schema_version": self.schema_version,
            "label_mode": self.label_mode,
            "total_labels": (
                len(self.deployment_outcomes)
                + len(self.adaptation_outcomes)
                + len(self.datapack_contributions)
                + len(self.deployment_receipts)
            ),
            "deployment_outcomes": len(self.deployment_outcomes),
            "adaptation_outcomes": len(self.adaptation_outcomes),
            "datapack_contributions": len(self.datapack_contributions),
            "deployment_receipts": len(self.deployment_receipts),
            "covered_episode_ids": episode_ids,
            "covered_episode_count": len(episode_ids),
            "source_domains": source_domains,
            "bundle_digest": self.bundle_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReceiptLabelBundle":
        return cls(
            schema_version=str(payload.get("schema_version", RECEIPT_LABEL_BUNDLE_SCHEMA_VERSION)),
            label_mode=str(payload.get("label_mode", "synthetic_shadow")),
            deployment_outcomes=[
                DeploymentOutcomeLabel.from_dict(row)
                for row in list(payload.get("deployment_outcomes", []) or [])
            ],
            adaptation_outcomes=[
                AdaptationOutcomeLabel.from_dict(row)
                for row in list(payload.get("adaptation_outcomes", []) or [])
            ],
            datapack_contributions=[
                DatapackContributionLabel.from_dict(row)
                for row in list(payload.get("datapack_contributions", []) or [])
            ],
            deployment_receipts=[
                DeploymentReceiptRecord.from_dict(row)
                for row in list(payload.get("deployment_receipts", []) or [])
            ],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def load_receipt_label_bundle(root_dir: str | Path) -> ReceiptLabelBundle:
    root = Path(root_dir)
    summary_path = root / "receipt_label_bundle.json"
    if summary_path.exists():
        return ReceiptLabelBundle.from_dict(json.loads(summary_path.read_text(encoding="utf-8")))
    return ReceiptLabelBundle(
        schema_version=RECEIPT_LABEL_BUNDLE_SCHEMA_VERSION,
        label_mode="unknown",
        deployment_outcomes=_load_jsonl(root / "deployment_outcomes.jsonl", DeploymentOutcomeLabel.from_dict),
        adaptation_outcomes=_load_jsonl(root / "adaptation_outcomes.jsonl", AdaptationOutcomeLabel.from_dict),
        datapack_contributions=_load_jsonl(root / "datapack_contributions.jsonl", DatapackContributionLabel.from_dict),
        deployment_receipts=_load_jsonl(root / "deployment_receipts.jsonl", DeploymentReceiptRecord.from_dict),
        metadata={"root_dir": str(root)},
    )


def write_receipt_label_bundle(bundle: ReceiptLabelBundle, output_dir: str | Path) -> Dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "deployment_outcomes.jsonl", [row.to_dict() for row in bundle.deployment_outcomes])
    _write_jsonl(root / "adaptation_outcomes.jsonl", [row.to_dict() for row in bundle.adaptation_outcomes])
    _write_jsonl(root / "datapack_contributions.jsonl", [row.to_dict() for row in bundle.datapack_contributions])
    _write_jsonl(root / "deployment_receipts.jsonl", [row.to_dict() for row in bundle.deployment_receipts])
    (root / "receipt_label_bundle.json").write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = bundle.coverage_summary()
    (root / "receipt_label_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "bundle": str(root / "receipt_label_bundle.json"),
        "summary": str(root / "receipt_label_summary.json"),
        "deployment_outcomes": str(root / "deployment_outcomes.jsonl"),
        "adaptation_outcomes": str(root / "adaptation_outcomes.jsonl"),
        "datapack_contributions": str(root / "datapack_contributions.jsonl"),
        "deployment_receipts": str(root / "deployment_receipts.jsonl"),
    }


def resolve_receipt_label_bundle(
    *,
    dataset: ReplayDatasetBundle,
    receipt_label_dir: Optional[str | Path] = None,
    allow_synthetic: bool = True,
    label_mode: str = "synthetic_shadow",
) -> ReceiptLabelBundle:
    if receipt_label_dir:
        root = Path(receipt_label_dir)
        if root.exists():
            return load_receipt_label_bundle(root)
    if not allow_synthetic:
        return ReceiptLabelBundle(
            schema_version=RECEIPT_LABEL_BUNDLE_SCHEMA_VERSION,
            label_mode="unavailable",
            deployment_outcomes=[],
            adaptation_outcomes=[],
            datapack_contributions=[],
            deployment_receipts=[],
            metadata={"reason": "receipt_labels_unavailable"},
        )
    return build_synthetic_receipt_label_bundle(dataset, label_mode=label_mode)


def build_synthetic_receipt_label_bundle(
    dataset: ReplayDatasetBundle,
    *,
    label_mode: str = "synthetic_shadow",
) -> ReceiptLabelBundle:
    deployment_outcomes: list[DeploymentOutcomeLabel] = []
    adaptation_outcomes: list[AdaptationOutcomeLabel] = []
    datapack_contributions: list[DatapackContributionLabel] = []
    deployment_receipts: list[DeploymentReceiptRecord] = []

    for episode in dataset.episodes:
        axes = dict(episode.econ_tensor_summary.get("axes", {}) or {})
        predicted_value = float(
            axes.get(
                "value_earned",
                episode.pricing_summary.get("net_customer_rate", 0.0),
            )
        )
        quality = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
        frontier_gain = float(episode.datapack_summary.get("marginal_frontier_gain", 0.0) or 0.0)
        hard_flags = sum(
            1
            for flag in episode.constraint_flags
            if str(flag.get("severity", "")).lower() == "hard"
        )
        warn_flags = sum(
            1
            for flag in episode.constraint_flags
            if str(flag.get("severity", "")).lower() == "warn"
        )
        realized_multiplier = max(
            0.25,
            min(
                1.35,
                0.82
                + 0.18 * quality
                + 0.12 * frontier_gain
                - 0.14 * hard_flags
                - 0.05 * warn_flags,
            ),
        )
        realized_value = float(predicted_value * realized_multiplier)
        quoted_rate = float(episode.pricing_summary.get("net_customer_rate", 0.0) or 0.0)
        billed_rate = max(0.0, min(quoted_rate, realized_value + 0.05 * max(1, episode.total_steps)))
        pricing_accepted = bool(quoted_rate <= max(0.0, realized_value + 0.1))
        objective_satisfied = hard_flags == 0 and realized_value >= 0.0
        failure_events = ["constraint_integrity_failure"] * hard_flags
        risk_events = ["constraint_warning"] * warn_flags
        datapack_id = str(
            episode.metadata.get("datapack_id")
            or episode.datapack_summary.get("datapack_id")
            or episode.episode_id
        )

        outcome = DeploymentOutcomeLabel(
            schema_version="deployment_outcome_label_v1",
            run_id=episode.run_id,
            episode_id=episode.episode_id,
            source_domain=episode.source_domain,
            deployment_id=f"deploy_{episode.episode_id}",
            objective_profile_id=str(episode.metadata.get("objective_profile_id", "balanced_contract")),
            predicted_value=predicted_value,
            realized_value=realized_value,
            pricing_accepted=pricing_accepted,
            failure_events=failure_events,
            risk_events=risk_events,
            provenance={
                "source": label_mode,
                "dataset_digest": dataset.manifest.dataset_digest,
            },
            metadata={
                "objective_satisfied": objective_satisfied,
                "quality_score": quality,
                "frontier_gain": frontier_gain,
            },
        )
        pricing = PricingAcceptanceLabel(
            schema_version="pricing_acceptance_label_v1",
            receipt_id=f"receipt_{episode.episode_id}",
            run_id=episode.run_id,
            episode_id=episode.episode_id,
            quoted_rate=quoted_rate,
            accepted_rate=billed_rate,
            accepted=pricing_accepted,
            reasons=["synthetic_shadow_accept"] if pricing_accepted else ["synthetic_shadow_reject"],
            metadata={"label_mode": label_mode},
        )
        adaptation = AdaptationOutcomeLabel(
            schema_version="adaptation_outcome_label_v1",
            run_id=episode.run_id,
            adaptation_id=f"adapt_{episode.episode_id}",
            recommended_mode="offline_td3_bc_shadow",
            realized_mode="offline_td3_bc_shadow" if realized_value >= predicted_value * 0.8 else "behavior_cloning_refresh",
            expected_gain=max(0.0, frontier_gain + 0.15 * predicted_value),
            realized_gain=max(0.0, frontier_gain + 0.15 * realized_value - 0.05 * hard_flags),
            compute_cost=max(0.01, 0.04 * max(1, episode.total_steps)),
            risk_cost=float(hard_flags) * 0.15,
            review_required=hard_flags > 0,
            provenance={"source": label_mode},
            metadata={"episode_id": episode.episode_id},
        )
        datapack = DatapackContributionLabel(
            schema_version="datapack_contribution_label_v1",
            datapack_id=datapack_id,
            run_id=episode.run_id,
            marginal_frontier_gain_predicted=frontier_gain,
            marginal_frontier_gain_realized=max(0.0, frontier_gain * realized_multiplier),
            data_share_credit_predicted=float(episode.datapack_summary.get("data_share_credit", 0.0) or 0.0),
            data_share_credit_realized=max(
                0.0,
                float(episode.datapack_summary.get("data_share_credit", 0.0) or 0.0) * realized_multiplier,
            ),
            downweight_recommended=quality < 0.45 or hard_flags > 0,
            provenance={"source": label_mode},
            metadata={"episode_id": episode.episode_id},
        )
        receipt = DeploymentReceiptRecord(
            schema_version="deployment_receipt_record_v1",
            run_id=episode.run_id,
            episode_id=episode.episode_id,
            deployment_id=outcome.deployment_id,
            source_domain=episode.source_domain,
            objective_profile_id=outcome.objective_profile_id,
            predicted_value=predicted_value,
            realized_value=realized_value,
            quoted_rate=quoted_rate,
            billed_rate=billed_rate,
            pricing_acceptance=pricing,
            adaptation_outcome_ref=adaptation.label_id,
            datapack_label_ref=datapack.label_id,
            provenance={"source": label_mode},
            metadata={"quality_score": quality, "frontier_gain": frontier_gain},
        )
        deployment_outcomes.append(outcome)
        adaptation_outcomes.append(adaptation)
        datapack_contributions.append(datapack)
        deployment_receipts.append(receipt)

    return ReceiptLabelBundle(
        schema_version=RECEIPT_LABEL_BUNDLE_SCHEMA_VERSION,
        label_mode=label_mode,
        deployment_outcomes=sorted(
            deployment_outcomes,
            key=lambda row: (row.run_id, row.episode_id, row.label_id),
        ),
        adaptation_outcomes=sorted(
            adaptation_outcomes,
            key=lambda row: (row.run_id, row.adaptation_id, row.label_id),
        ),
        datapack_contributions=sorted(
            datapack_contributions,
            key=lambda row: (row.run_id, row.datapack_id, row.label_id),
        ),
        deployment_receipts=sorted(
            deployment_receipts,
            key=lambda row: (row.run_id, row.episode_id, row.record_id),
        ),
        metadata={
            "dataset_digest": dataset.manifest.dataset_digest,
            "run_ids": list(dataset.manifest.run_ids),
        },
    )


def _load_jsonl(path: Path, factory) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(factory(json.loads(line)))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


__all__ = [
    "RECEIPT_LABEL_BUNDLE_SCHEMA_VERSION",
    "ReceiptLabelBundle",
    "load_receipt_label_bundle",
    "write_receipt_label_bundle",
    "resolve_receipt_label_bundle",
    "build_synthetic_receipt_label_bundle",
]
