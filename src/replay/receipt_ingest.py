"""Receipt-label ingestion for synthetic, simulated, and future real deployment outcomes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.economics.receipt_schema import DeploymentReceiptRecord, PricingAcceptanceLabel
from src.ontology.deployment_labels import (
    AdaptationOutcomeLabel,
    DatapackContributionLabel,
    DeploymentOutcomeLabel,
)
from src.replay.dataset import ReplayDatasetBuilder, ReplayDatasetBundle, load_replay_dataset
from src.replay.preconditions import (
    build_replay_execution_preconditions,
    summarize_replay_execution_preconditions,
)
from src.training.training_manifest import load_training_runtime_manifest
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
        source_domain_counts: Dict[str, int] = {}
        for row in self.deployment_outcomes:
            if row.source_domain:
                source_domain_counts[row.source_domain] = source_domain_counts.get(row.source_domain, 0) + 1
        for adaptation_row in self.adaptation_outcomes:
            if adaptation_row.source_domain:
                source_domain_counts[adaptation_row.source_domain] = source_domain_counts.get(adaptation_row.source_domain, 0) + 1
        for datapack_row in self.datapack_contributions:
            if datapack_row.source_domain:
                source_domain_counts[datapack_row.source_domain] = source_domain_counts.get(datapack_row.source_domain, 0) + 1
        for receipt_row in self.deployment_receipts:
            if receipt_row.source_domain:
                source_domain_counts[receipt_row.source_domain] = source_domain_counts.get(receipt_row.source_domain, 0) + 1
        source_domains = sorted(source_domain_counts)
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
            "source_domain_counts": dict(sorted(source_domain_counts.items())),
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
            if (root / "receipt_label_bundle.json").exists() or (root / "deployment_receipts.jsonl").exists():
                return load_receipt_label_bundle(root)
            if (root / "training_runtime_manifest.json").exists() or (root / "online_episode_receipts.jsonl").exists():
                return build_training_run_receipt_label_bundle(root, replay_dataset_dir=dataset.root_dir, label_mode=label_mode)
            if any(root.glob("episode_*")):
                return build_rollout_receipt_label_bundle(root, label_mode="sim_rollout")
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
    return _build_receipt_label_bundle(
        dataset=dataset,
        observed_outcomes={},
        label_mode=label_mode,
        metadata={
            "dataset_digest": dataset.manifest.dataset_digest,
            "run_ids": list(dataset.manifest.run_ids),
            "observation_source": "synthetic_defaults",
        },
    )


def build_workcell_episode_log_receipt_label_bundle(
    episode_log_path: str | Path,
    *,
    run_id: Optional[str] = None,
    source_domain: str = "sim_rollout",
    objective_profile_id: str = "balanced_contract",
    label_mode: str = "sim_rollout",
) -> ReceiptLabelBundle:
    dataset = ReplayDatasetBuilder().add_workcell_episode_log(
        episode_log_path,
        run_id=run_id,
        source_domain=source_domain,
        objective_profile_id=objective_profile_id,
    ).build()
    return _build_receipt_label_bundle(
        dataset=dataset,
        observed_outcomes=_observed_outcomes_from_dataset(dataset, label_mode=label_mode),
        label_mode=label_mode,
        metadata={
            "episode_log_path": str(episode_log_path),
            "observation_source": "workcell_episode_log",
        },
    )


def build_rollout_receipt_label_bundle(
    rollout_root: str | Path,
    *,
    scenario_id: Optional[str] = None,
    run_id: Optional[str] = None,
    source_domain: str = "sim_rollout",
    objective_profile_id: str = "balanced_contract",
    label_mode: str = "sim_rollout",
) -> ReceiptLabelBundle:
    dataset = ReplayDatasetBuilder().add_rollout_bundle(
        rollout_root,
        scenario_id=scenario_id,
        run_id=run_id,
        source_domain=source_domain,
        objective_profile_id=objective_profile_id,
    ).build()
    return _build_receipt_label_bundle(
        dataset=dataset,
        observed_outcomes=_observed_outcomes_from_dataset(dataset, label_mode=label_mode),
        label_mode=label_mode,
        metadata={
            "rollout_root": str(rollout_root),
            "scenario_id": scenario_id,
            "observation_source": "rollout_bundle",
        },
    )


def _load_json_object(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _training_run_future_training_context(
    root: Path,
    *,
    manifest,
    dataset: Optional[ReplayDatasetBundle],
) -> Dict[str, Any]:
    manifest_path = root / "training_runtime_manifest.json"
    artifact_paths = dict(getattr(manifest, "artifact_paths", {}) or {})
    promotion_ledger_path = _existing_path(
        getattr(manifest, "promotion_ledger_path", None) if manifest is not None else None,
        artifact_paths.get("promotion_ledger_ref"),
        getattr(manifest, "promotion_evidence_path", None) if manifest is not None else None,
        artifact_paths.get("regal_promotion_eval"),
    )
    budget_settlement_path = _existing_path(
        getattr(manifest, "budget_settlement_path", None) if manifest is not None else None,
        artifact_paths.get("budget_settlement_report"),
    )
    budget_settlement_payload = _load_json_object(budget_settlement_path)
    budget_settlement_live = bool(
        budget_settlement_payload.get(
            "budget_settlement_live",
            getattr(manifest, "budget_settlement_live", False) if manifest is not None else False,
        )
    )
    replay_roundtrip_complete = bool(
        dataset is not None
        and (
            getattr(manifest, "replay_dataset_dir", None)
            or artifact_paths.get("online_replay_dataset_manifest")
            or artifact_paths.get("replay_dataset_manifest")
            or dataset.root_dir
        )
    )
    promotion_trace_complete = bool(
        promotion_ledger_path
        and (
            getattr(manifest, "promotion_evidence_path", None)
            or artifact_paths.get("regal_promotion_eval")
            or artifact_paths.get("receipt_label_bundle")
        )
    )
    future_training_artifacts: Dict[str, Any] = {}
    if manifest_path.exists():
        future_training_artifacts["training_runtime_manifest"] = str(manifest_path)
    if promotion_ledger_path is not None:
        future_training_artifacts["promotion_ledger_ref"] = str(promotion_ledger_path)
    if budget_settlement_path is not None:
        future_training_artifacts["budget_settlement_report"] = str(budget_settlement_path)
    future_training_signals = {
        "replay_roundtrip_complete": replay_roundtrip_complete,
        "promotion_trace_complete": promotion_trace_complete,
        "budget_settlement_live": budget_settlement_live,
    }
    return {
        "future_training_artifacts": dict(sorted(future_training_artifacts.items())),
        "future_training_signals": dict(sorted(future_training_signals.items())),
        "budget_settlement_payload": budget_settlement_payload,
        "inferential_learnability_summary": dict(
            getattr(manifest, "inferential_learnability_summary", {}) or {}
        ),
        "inferential_work_order_summary": dict(
            getattr(manifest, "inferential_work_order_summary", {}) or {}
        ),
    }


def _episode_signal_overrides_from_observed(observed: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract per-episode backend-truth fields from observed receipt rows."""
    payload = dict(observed or {})
    overrides: Dict[str, Any] = {}
    if "scene_tracks_non_stub" in payload:
        overrides["scene_tracks_non_stub"] = bool(payload.get("scene_tracks_non_stub"))
    for key in (
        "scene_tracks_backend",
        "openvla_backend_selected",
        "teacher_runtime_backend_selected",
        "teacher_backend_selected",
        "vision_backbone_selected",
        "openvla_vision_backbone_selected",
        "teacher_runtime_vision_backbone_selected",
        "semantic_grounding_mode",
        "grounding_mode",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            overrides[key] = value
    for key in (
        "semantic_memory_grounded",
        "semantic_grounding_heuristic",
    ):
        if key in payload:
            overrides[key] = bool(payload.get(key))
    return overrides


def _merge_future_training_signals(
    base: Optional[Mapping[str, Any]],
    updates: Mapping[str, Any],
) -> Dict[str, bool]:
    merged: Dict[str, bool] = {
        str(key): bool(value)
        for key, value in dict(base or {}).items()
    }
    for key, value in dict(updates).items():
        normalized_key = str(key)
        merged[normalized_key] = bool(merged.get(normalized_key, False) or value)
    return dict(sorted(merged.items()))


def _enrich_training_run_dataset(
    dataset: ReplayDatasetBundle,
    *,
    root: Path,
    manifest,
    observed_outcomes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ReplayDatasetBundle:
    context = _training_run_future_training_context(root, manifest=manifest, dataset=dataset)
    future_training_artifacts = dict(context.get("future_training_artifacts", {}) or {})
    future_training_signals = dict(context.get("future_training_signals", {}) or {})
    inferential_learnability_summary = dict(
        context.get("inferential_learnability_summary", {}) or {}
    )
    inferential_work_order_summary = dict(
        context.get("inferential_work_order_summary", {}) or {}
    )
    if not future_training_artifacts and not future_training_signals:
        return dataset

    steps_by_episode: Dict[str, list[Any]] = {}
    for step in dataset.steps:
        steps_by_episode.setdefault(step.episode_id, []).append(step)
    windows_by_episode: Dict[str, list[Any]] = {}
    for window in dataset.windows:
        windows_by_episode.setdefault(window.episode_id, []).append(window)

    enriched_episodes = []
    reports = []
    for episode in dataset.episodes:
        observed_row = dict((observed_outcomes or {}).get(episode.episode_id, {}) or {})
        observed_overrides = _episode_signal_overrides_from_observed(observed_row)
        existing_artifacts = episode.metadata.get("future_training_artifacts", {})
        existing_signals = episode.metadata.get("future_training_signals", {})
        merged_artifacts = {
            **dict(existing_artifacts or {}),
            **future_training_artifacts,
        }
        merged_signals = _merge_future_training_signals(existing_signals, future_training_signals)
        metadata = {
            **dict(episode.metadata),
            **observed_overrides,
            "future_training_artifacts": merged_artifacts,
            "future_training_signals": merged_signals,
            "training_runtime_manifest": future_training_artifacts.get("training_runtime_manifest"),
            "promotion_ledger_ref": future_training_artifacts.get("promotion_ledger_ref"),
            "budget_settlement_live": merged_signals.get("budget_settlement_live", False),
        }
        provenance = {
            **dict(episode.provenance),
            **future_training_artifacts,
        }
        enriched_episode = replace(
            episode,
            metadata=metadata,
            provenance=provenance,
        )
        report = build_replay_execution_preconditions(
            enriched_episode,
            steps=steps_by_episode.get(episode.episode_id, []),
            windows=windows_by_episode.get(episode.episode_id, []),
        )
        enriched_episode = replace(
            enriched_episode,
            metadata={
                **metadata,
                "execution_preconditions": report.to_dict(),
            },
        )
        enriched_episodes.append(enriched_episode)
        reports.append(report)

    summary = summarize_replay_execution_preconditions(reports)
    enriched_manifest = replace(
        dataset.manifest,
        metadata={
            **dict(dataset.manifest.metadata),
            "execution_precondition_summary": summary,
            "training_run_future_training_artifacts": future_training_artifacts,
            "training_run_future_training_signals": future_training_signals,
            "inferential_learnability_summary": inferential_learnability_summary
            or dict(dataset.manifest.metadata.get("inferential_learnability_summary", {}) or {}),
            "inferential_work_order_summary": inferential_work_order_summary,
        },
    )
    return ReplayDatasetBundle(
        manifest=enriched_manifest,
        episodes=enriched_episodes,
        steps=list(dataset.steps),
        windows=list(dataset.windows),
        root_dir=dataset.root_dir,
    )


def build_training_run_receipt_label_bundle(
    training_run_root: str | Path,
    *,
    replay_dataset_dir: Optional[str | Path] = None,
    label_mode: str = "training_run",
) -> ReceiptLabelBundle:
    root = Path(training_run_root)
    bundle_path = root / "receipt_labels" / "receipt_label_bundle.json"
    if bundle_path.exists():
        return load_receipt_label_bundle(bundle_path.parent)

    manifest = None
    manifest_path = root / "training_runtime_manifest.json"
    if manifest_path.exists():
        manifest = load_training_runtime_manifest(manifest_path)
        receipt_bundle_path = _existing_path(
            manifest.artifact_paths.get("receipt_label_bundle"),
            manifest.artifact_paths.get("receipt_label_summary"),
        )
        if receipt_bundle_path and Path(receipt_bundle_path).name == "receipt_label_bundle.json":
            return load_receipt_label_bundle(Path(receipt_bundle_path).parent)

    dataset = None
    dataset_root = _resolve_training_run_dataset_root(root, manifest=manifest, replay_dataset_dir=replay_dataset_dir)
    if dataset_root is not None:
        dataset = load_replay_dataset(dataset_root)
    elif _resolve_episode_logs_dir(root, manifest=manifest) is not None:
        dataset = _build_dataset_from_episode_logs(
            _resolve_episode_logs_dir(root, manifest=manifest),
            run_id=(manifest.run_id if manifest is not None else None),
        )

    observed_outcomes = _load_observed_outcomes(
        _resolve_observed_episode_receipt_path(root, manifest=manifest),
    )
    if dataset is None:
        return ReceiptLabelBundle(
            schema_version=RECEIPT_LABEL_BUNDLE_SCHEMA_VERSION,
            label_mode="unavailable",
            deployment_outcomes=[],
            adaptation_outcomes=[],
            datapack_contributions=[],
            deployment_receipts=[],
            metadata={
                "reason": "training_run_receipt_dataset_unavailable",
                "training_run_root": str(root),
            },
        )
    if not observed_outcomes:
        observed_outcomes = _observed_outcomes_from_dataset(dataset, label_mode=label_mode)
    if manifest is not None:
        dataset = _enrich_training_run_dataset(
            dataset,
            root=root,
            manifest=manifest,
            observed_outcomes=observed_outcomes,
        )
    return _build_receipt_label_bundle(
        dataset=dataset,
        observed_outcomes=observed_outcomes,
        label_mode=label_mode,
        metadata={
            "training_run_root": str(root),
            "training_runtime_manifest": str(manifest_path) if manifest_path.exists() else None,
            "promotion_ledger_ref": (
                dataset.manifest.metadata.get("training_run_future_training_artifacts", {})
                .get("promotion_ledger_ref")
                if manifest is not None
                else None
            ),
            "budget_settlement_live": (
                dataset.manifest.metadata.get("training_run_future_training_signals", {})
                .get("budget_settlement_live")
                if manifest is not None
                else False
            ),
            "execution_precondition_summary": dict(
                dataset.manifest.metadata.get("execution_precondition_summary", {}) or {}
            ),
            "inferential_learnability_summary": dict(
                dataset.manifest.metadata.get("inferential_learnability_summary", {}) or {}
            ),
            "inferential_work_order_summary": dict(
                dataset.manifest.metadata.get("inferential_work_order_summary", {}) or {}
            ),
            "observation_source": (
                "online_episode_receipts"
                if _resolve_observed_episode_receipt_path(root, manifest=manifest) is not None
                else "training_run_dataset"
            ),
        },
    )


def _build_receipt_label_bundle(
    *,
    dataset: ReplayDatasetBundle,
    observed_outcomes: Mapping[str, Mapping[str, Any]],
    label_mode: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ReceiptLabelBundle:
    deployment_outcomes: list[DeploymentOutcomeLabel] = []
    adaptation_outcomes: list[AdaptationOutcomeLabel] = []
    datapack_contributions: list[DatapackContributionLabel] = []
    deployment_receipts: list[DeploymentReceiptRecord] = []

    for episode in dataset.episodes:
        defaults = _default_episode_outcome_payload(episode)
        observed = dict(observed_outcomes.get(episode.episode_id, {}) or {})
        source_domain = str(observed.get("source_domain", defaults["source_domain"]) or defaults["source_domain"])
        predicted_value = float(observed.get("predicted_value", defaults["predicted_value"]))
        realized_value = float(observed.get("realized_value", defaults["realized_value"]))
        quoted_rate = float(observed.get("quoted_rate", defaults["quoted_rate"]))
        billed_rate = float(observed.get("accepted_rate", observed.get("billed_rate", defaults["billed_rate"])))
        pricing_accepted = bool(observed.get("pricing_accepted", defaults["pricing_accepted"]))
        task_success = observed.get("task_success", defaults["task_success"])
        objective_satisfied = observed.get("objective_satisfied", defaults["objective_satisfied"])
        realized_reward = observed.get("realized_reward", defaults["realized_reward"])
        failure_events = [str(value) for value in observed.get("failure_events", defaults["failure_events"]) or []]
        risk_events = [str(value) for value in observed.get("risk_events", defaults["risk_events"]) or []]
        incident_events = [str(value) for value in observed.get("incident_events", defaults["incident_events"]) or []]
        human_review_label = observed.get("human_review_label", defaults["human_review_label"])
        override_label = observed.get("override_label", defaults["override_label"])
        datapack_id = str(observed.get("datapack_id", defaults["datapack_id"]) or defaults["datapack_id"])

        outcome = DeploymentOutcomeLabel(
            schema_version="deployment_outcome_label_v1",
            run_id=episode.run_id,
            episode_id=episode.episode_id,
            source_domain=source_domain,
            deployment_id=f"deploy_{episode.episode_id}",
            objective_profile_id=str(episode.metadata.get("objective_profile_id", "balanced_contract")),
            predicted_value=predicted_value,
            realized_value=realized_value,
            pricing_accepted=pricing_accepted,
            task_success=task_success,
            objective_satisfied=objective_satisfied,
            realized_reward=realized_reward,
            failure_events=failure_events,
            risk_events=risk_events,
            incident_events=incident_events,
            human_review_label=human_review_label,
            override_label=override_label,
            provenance={
                "source": label_mode,
                "dataset_digest": dataset.manifest.dataset_digest,
            },
            metadata={
                "quality_score": defaults["quality_score"],
                "frontier_gain": defaults["frontier_gain"],
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
            reasons=[
                str(value)
                for value in observed.get(
                    "pricing_reasons",
                    defaults["pricing_reasons"],
                ) or []
            ],
            metadata={"label_mode": label_mode},
        )
        adaptation = AdaptationOutcomeLabel(
            schema_version="adaptation_outcome_label_v1",
            run_id=episode.run_id,
            adaptation_id=f"adapt_{episode.episode_id}",
            source_domain=source_domain,
            recommended_mode=str(observed.get("recommended_mode", defaults["recommended_mode"])),
            realized_mode=str(observed.get("realized_mode", defaults["realized_mode"])),
            expected_gain=float(observed.get("expected_adaptation_benefit", defaults["expected_adaptation_benefit"])),
            realized_gain=float(observed.get("realized_adaptation_benefit", defaults["realized_adaptation_benefit"])),
            compute_cost=float(observed.get("adaptation_compute_cost", defaults["adaptation_compute_cost"])),
            risk_cost=float(observed.get("adaptation_risk_cost", defaults["adaptation_risk_cost"])),
            review_required=bool(observed.get("adaptation_review_required", defaults["adaptation_review_required"])),
            provenance={"source": label_mode},
            metadata={"episode_id": episode.episode_id},
        )
        datapack = DatapackContributionLabel(
            schema_version="datapack_contribution_label_v1",
            datapack_id=datapack_id,
            run_id=episode.run_id,
            source_domain=source_domain,
            marginal_frontier_gain_predicted=float(observed.get("marginal_frontier_gain_predicted", defaults["marginal_frontier_gain_predicted"])),
            marginal_frontier_gain_realized=float(observed.get("marginal_frontier_gain_realized", defaults["marginal_frontier_gain_realized"])),
            data_share_credit_predicted=float(observed.get("data_share_credit_predicted", defaults["data_share_credit_predicted"])),
            data_share_credit_realized=float(observed.get("data_share_credit_realized", defaults["data_share_credit_realized"])),
            downweight_recommended=bool(observed.get("downweight_recommended", defaults["downweight_recommended"])),
            provenance={"source": label_mode},
            metadata={"episode_id": episode.episode_id},
        )
        receipt = DeploymentReceiptRecord(
            schema_version="deployment_receipt_record_v1",
            run_id=episode.run_id,
            episode_id=episode.episode_id,
            deployment_id=outcome.deployment_id,
            source_domain=source_domain,
            objective_profile_id=outcome.objective_profile_id,
            predicted_value=predicted_value,
            realized_value=realized_value,
            quoted_rate=quoted_rate,
            billed_rate=billed_rate,
            pricing_acceptance=pricing,
            realized_reward=realized_reward,
            task_success=task_success,
            objective_satisfied=objective_satisfied,
            incident_events=incident_events,
            human_review_label=human_review_label,
            override_label=override_label,
            adaptation_outcome_ref=adaptation.label_id,
            datapack_label_ref=datapack.label_id,
            provenance={"source": label_mode},
            metadata={
                "quality_score": defaults["quality_score"],
                "frontier_gain": defaults["frontier_gain"],
            },
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
            **dict(metadata or {}),
        },
    )


def _default_episode_outcome_payload(episode) -> Dict[str, Any]:
    axes = dict(episode.econ_tensor_summary.get("axes", {}) or {})
    predicted_value = float(
        axes.get(
            "value_earned",
            episode.pricing_summary.get("net_customer_rate", 0.0),
        )
    )
    quality = float(episode.datapack_summary.get("quality_score", 0.0) or 0.0)
    frontier_gain = float(episode.datapack_summary.get("marginal_frontier_gain", 0.0) or 0.0)
    data_share_credit = float(episode.datapack_summary.get("data_share_credit", 0.0) or 0.0)
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
    incident_events = failure_events + risk_events
    task_success = episode.status.lower() not in {"failed", "error"} and hard_flags == 0
    datapack_id = str(
        episode.metadata.get("datapack_id")
        or episode.datapack_summary.get("datapack_id")
        or episode.episode_id
    )
    return {
        "source_domain": episode.source_domain,
        "predicted_value": predicted_value,
        "realized_value": realized_value,
        "quoted_rate": quoted_rate,
        "billed_rate": billed_rate,
        "pricing_accepted": pricing_accepted,
        "pricing_reasons": ["synthetic_shadow_accept"] if pricing_accepted else ["synthetic_shadow_reject"],
        "task_success": task_success,
        "objective_satisfied": objective_satisfied,
        "realized_reward": float(episode.total_reward),
        "failure_events": failure_events,
        "risk_events": risk_events,
        "incident_events": incident_events,
        "human_review_label": None,
        "override_label": None,
        "recommended_mode": "offline_td3_bc_shadow",
        "realized_mode": "offline_td3_bc_shadow" if realized_value >= predicted_value * 0.8 else "behavior_cloning_refresh",
        "expected_adaptation_benefit": max(0.0, frontier_gain + 0.15 * predicted_value),
        "realized_adaptation_benefit": max(0.0, frontier_gain + 0.15 * realized_value - 0.05 * hard_flags),
        "adaptation_compute_cost": max(0.01, 0.04 * max(1, episode.total_steps)),
        "adaptation_risk_cost": float(hard_flags) * 0.15,
        "adaptation_review_required": hard_flags > 0,
        "datapack_id": datapack_id,
        "marginal_frontier_gain_predicted": frontier_gain,
        "marginal_frontier_gain_realized": max(0.0, frontier_gain * realized_multiplier),
        "data_share_credit_predicted": data_share_credit,
        "data_share_credit_realized": max(0.0, data_share_credit * realized_multiplier),
        "downweight_recommended": quality < 0.45 or hard_flags > 0,
        "quality_score": quality,
        "frontier_gain": frontier_gain,
    }


def _observed_outcomes_from_dataset(
    dataset: ReplayDatasetBundle,
    *,
    label_mode: str,
) -> Dict[str, Dict[str, Any]]:
    observed: Dict[str, Dict[str, Any]] = {}
    for episode in dataset.episodes:
        defaults = _default_episode_outcome_payload(episode)
        axes = dict(episode.econ_tensor_summary.get("axes", {}) or {})
        observed[episode.episode_id] = {
            "source_domain": episode.source_domain or label_mode,
            "predicted_value": defaults["predicted_value"],
            "realized_value": float(axes.get("value_earned", episode.total_reward)),
            "quoted_rate": defaults["quoted_rate"],
            "billed_rate": defaults["billed_rate"],
            "pricing_accepted": defaults["pricing_accepted"],
            "task_success": defaults["task_success"],
            "objective_satisfied": defaults["objective_satisfied"],
            "realized_reward": float(episode.total_reward),
            "failure_events": list(defaults["failure_events"]),
            "risk_events": list(defaults["risk_events"]),
            "incident_events": list(defaults["incident_events"]),
            "expected_adaptation_benefit": defaults["expected_adaptation_benefit"],
            "realized_adaptation_benefit": max(0.0, float(episode.total_reward) * 0.1),
            "adaptation_compute_cost": defaults["adaptation_compute_cost"],
            "adaptation_risk_cost": defaults["adaptation_risk_cost"],
            "adaptation_review_required": defaults["adaptation_review_required"],
            "marginal_frontier_gain_predicted": defaults["marginal_frontier_gain_predicted"],
            "marginal_frontier_gain_realized": max(0.0, defaults["marginal_frontier_gain_predicted"]),
            "data_share_credit_predicted": defaults["data_share_credit_predicted"],
            "data_share_credit_realized": max(0.0, defaults["data_share_credit_predicted"]),
            "downweight_recommended": defaults["downweight_recommended"],
        }
    return observed


def _build_dataset_from_episode_logs(
    episode_logs_dir: Optional[str | Path],
    *,
    run_id: Optional[str] = None,
) -> Optional[ReplayDatasetBundle]:
    if episode_logs_dir is None:
        return None
    root = Path(episode_logs_dir)
    if not root.exists():
        return None
    builder = ReplayDatasetBuilder()
    for path in sorted(root.glob("*.json")):
        builder.add_workcell_episode_log(path, run_id=run_id, source_domain="training_run")
    try:
        return builder.build()
    except ValueError:
        return None


def _resolve_training_run_dataset_root(
    root: Path,
    *,
    manifest,
    replay_dataset_dir: Optional[str | Path],
) -> Optional[Path]:
    candidates: list[Optional[str | Path]] = [
        replay_dataset_dir,
        getattr(manifest, "replay_dataset_dir", None) if manifest is not None else None,
        root / "online_replay_dataset",
        root / "replay_dataset",
    ]
    if manifest is not None:
        candidates.extend(
            [
                _artifact_parent(manifest.artifact_paths.get("online_replay_dataset_manifest")),
                _artifact_parent(manifest.artifact_paths.get("replay_dataset_manifest")),
            ]
        )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() and path.name == "manifest.json":
            path = path.parent
        if (path / "manifest.json").exists():
            return path
    return None


def _resolve_episode_logs_dir(root: Path, *, manifest) -> Optional[Path]:
    candidates: list[Optional[str | Path]] = [
        root / "online_episode_logs",
        manifest.artifact_paths.get("online_episode_logs") if manifest is not None else None,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and path.is_dir():
            return path
    return None


def _resolve_observed_episode_receipt_path(root: Path, *, manifest) -> Optional[Path]:
    candidates: list[Optional[str | Path]] = [
        root / "online_episode_receipts.jsonl",
        manifest.artifact_paths.get("online_episode_receipts") if manifest is not None else None,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _load_observed_outcomes(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    observed: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            episode_id = str(payload.get("episode_id", ""))
            if episode_id:
                observed[episode_id] = dict(payload)
    return observed


def _existing_path(*candidates: Optional[str | Path]) -> Optional[Path]:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _artifact_parent(path: Optional[str | Path]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path)
    return candidate.parent if candidate.exists() else None


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
    "build_workcell_episode_log_receipt_label_bundle",
    "build_rollout_receipt_label_bundle",
    "build_training_run_receipt_label_bundle",
]
