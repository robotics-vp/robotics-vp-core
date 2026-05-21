"""Runbook templates for future Economic WM provider/GPU bring-up.

This module turns teacher/provider evidence contracts into manifest-shaped run
plans. It intentionally emits templates only: no provider, GPU training, or
promotion benchmark is considered run until a real manifest records execution
receipts, timestamps, costs, and artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.evidence_contracts import (
    EconomicWMEvidenceRequirement,
    EconomicWMTeacherProviderContract,
    load_economic_wm_teacher_provider_contract,
)

ECONOMIC_WM_PROVIDER_RUN_TEMPLATE_VERSION = "economic_wm_provider_run_template_v1"
ECONOMIC_WM_PROVIDER_RUNBOOK_VERSION = "economic_wm_provider_runbook_v1"


@dataclass(frozen=True)
class _RunTemplateSpec:
    title: str
    mode: str
    run_class: str
    pod_class: Optional[str]
    epistemic_status: str
    subsystem: str
    blocker: str
    command_templates: list[str]
    artifact_paths: list[str]
    config_paths: list[str] = field(default_factory=list)
    dependency_chain: list[str] = field(default_factory=list)
    expected_value: str = ""
    estimated_cost_usd: Optional[float] = None
    urgency: str = "medium"
    image: str = "template-only"
    template: str = "template-only"
    gpu_class: Optional[str] = None
    seeds: list[int] = field(default_factory=list)
    local_verification_available: bool = False


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


_TEMPLATE_GUARD = "echo 'TEMPLATE_ONLY: replace this guard with the real command before launch' && false"


_TEMPLATE_SPECS: Dict[str, _RunTemplateSpec] = {
    "non_stub_teacher_runtime_invocation": _RunTemplateSpec(
        title="Non-stub teacher runtime invocation proof-of-life",
        mode="runpod",
        run_class="provider",
        pod_class="provider",
        epistemic_status="proof_of_life",
        subsystem="teacher_runtime",
        blocker="non_stub_teacher_runtime_not_verified",
        command_templates=[
            "python3 scripts/economic_world_model/prepare_economic_wm_teacher_provider_contracts.py --output-dir artifacts/economic_world_model/economic_wm_teacher_provider_contracts",
            _TEMPLATE_GUARD,
        ],
        artifact_paths=[
            "artifacts/economic_world_model/economic_wm_teacher_provider_contracts/economic_wm_teacher_provider_contract_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/teacher_trace_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/external_provider_truth_v1.json",
        ],
        config_paths=[
            "docs/economic_world_model/economic_wm_teacher_provider_contracts.md",
            "docs/agent_ergonomics/run_manifest_schema.md",
        ],
        dependency_chain=[
            "economic_wm_teacher_provider_contract_v1",
            "teacher_adapter_contract_v1",
        ],
        expected_value=(
            "Burns down whether the Economic WM can trust non-stub teacher "
            "sidecars as supervision rather than treating teacher output as unavailable."
        ),
        urgency="high",
        image="runpod-provider-image-template",
        template="economic-wm-teacher-provider-template",
        gpu_class="provider-dependent",
    ),
    "provider_runtime_truth_receipts": _RunTemplateSpec(
        title="External provider runtime truth receipt proof-of-life",
        mode="runpod",
        run_class="provider",
        pod_class="provider",
        epistemic_status="proof_of_life",
        subsystem="external_provider_runtime",
        blocker="provider_bringup_not_run",
        command_templates=[
            "python3 scripts/economic_world_model/compile_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/economic_wm_provider_runbook",
            _TEMPLATE_GUARD,
        ],
        artifact_paths=[
            "artifacts/economic_world_model/provider_runs/{run_id}/runtime_provider_manifest_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/provider_invocation_receipt_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/external_provider_truth_v1.json",
        ],
        config_paths=[
            "docs/economic_world_model/economic_wm_provider_runbook.md",
            "docs/agent_ergonomics/run_manifest_schema.md",
        ],
        dependency_chain=[
            "economic_wm_teacher_provider_contract_v1",
            "runtime_provider_manifest_v1",
        ],
        expected_value=(
            "Separates real provider availability from planning-only fallback "
            "truth before any Economic WM training consumes provider fields."
        ),
        urgency="high",
        image="runpod-provider-image-template",
        template="economic-wm-provider-truth-template",
        gpu_class="provider-dependent",
    ),
    "promotion_grade_benchmark_evidence": _RunTemplateSpec(
        title="Promotion-grade Economic WM benchmark evidence candidate",
        mode="runpod",
        run_class="train",
        pod_class="train",
        epistemic_status="benchmark_candidate",
        subsystem="promotion_benchmark_evidence",
        blocker="promotion_grade_benchmark_evidence_missing",
        command_templates=[
            "python3 scripts/economic_world_model/compile_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/economic_wm_provider_runbook",
            _TEMPLATE_GUARD,
        ],
        artifact_paths=[
            "artifacts/economic_world_model/provider_runs/{run_id}/benchmark_gate_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/promotion_metric_report_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/benchmark_evidence_bundle_v1.json",
        ],
        config_paths=[
            "docs/economic_world_model/economic_wm_provider_runbook.md",
            "docs/agent_ergonomics/run_manifest_schema.md",
        ],
        dependency_chain=[
            "economic_wm_training_corpus_manifest_v1",
            "economic_wm_teacher_provider_contract_v1",
            "provider_runtime_truth_receipts",
        ],
        expected_value=(
            "Converts benchmark-ready local rows into decision-grade evidence "
            "without promoting a model from fixture-only metrics."
        ),
        urgency="medium",
        image="runpod-train-image-template",
        template="economic-wm-benchmark-template",
        gpu_class="train-dependent",
    ),
    "gpu_training_runtime_receipt": _RunTemplateSpec(
        title="GPU training runtime receipt proof-of-life",
        mode="runpod",
        run_class="train",
        pod_class="train",
        epistemic_status="proof_of_life",
        subsystem="gpu_training_runtime",
        blocker="gpu_training_not_run",
        command_templates=[
            "python3 scripts/economic_world_model/compile_economic_wm_provider_runbook.py --output-dir artifacts/economic_world_model/economic_wm_provider_runbook",
            _TEMPLATE_GUARD,
        ],
        artifact_paths=[
            "artifacts/economic_world_model/provider_runs/{run_id}/training_runtime_manifest_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/gpu_runtime_receipt_v1.json",
            "artifacts/economic_world_model/provider_runs/{run_id}/checkpoint_manifest_v1.json",
        ],
        config_paths=[
            "scripts/TRAINING_MIGRATION_BACKLOG.json",
            "docs/agent_ergonomics/run_manifest_schema.md",
        ],
        dependency_chain=[
            "economic_wm_training_corpus_manifest_v1",
            "provider_runtime_truth_receipts",
            "promotion_grade_benchmark_evidence",
        ],
        expected_value=(
            "Proves an Economic WM training loop can run on GPU and emit "
            "ledger-grade receipts before any training result is interpreted."
        ),
        urgency="medium",
        image="runpod-train-image-template",
        template="economic-wm-gpu-training-template",
        gpu_class="train-dependent",
    ),
    "replay_row_linkage_integrity": _RunTemplateSpec(
        title="Local replay-row linkage integrity check",
        mode="local",
        run_class="loop",
        pod_class=None,
        epistemic_status="proof_of_life",
        subsystem="local_replay_bridge",
        blocker="none",
        command_templates=[
            "python3 scripts/economic_world_model/materialize_economic_wm_training_rows.py --output-dir artifacts/economic_world_model/economic_wm_training_rows --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json",
            "python3 scripts/economic_world_model/evaluate_economic_wm_shadow_allocations.py --output-dir artifacts/economic_world_model/economic_wm_shadow_allocation_eval --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json --corpus-manifest artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json --rows artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl",
        ],
        artifact_paths=[
            "artifacts/economic_world_model/economic_wm_training_rows/economic_wm_training_corpus_manifest_v1.json",
            "artifacts/economic_world_model/economic_wm_training_rows/economic_wm_replay_feature_rows_v1.jsonl",
            "artifacts/economic_world_model/economic_wm_shadow_allocation_eval/economic_wm_shadow_allocation_eval_v1.json",
        ],
        config_paths=[
            "docs/economic_world_model/economic_wm_training_rows.md",
            "docs/economic_world_model/economic_wm_shadow_allocation_eval.md",
        ],
        dependency_chain=[
            "governed_video_stage1_bridge_export_v1",
            "economic_wm_scaffold_report_v1",
        ],
        expected_value=(
            "Keeps local replay/data-bridge truth intact while external GPU "
            "or provider work is unavailable."
        ),
        estimated_cost_usd=0.0,
        urgency="high",
        image="local-python",
        template="local-cli",
        local_verification_available=True,
    ),
}


@dataclass(frozen=True)
class EconomicWMProviderRunTemplate:
    """One manifest-shaped template for a future Economic WM run."""

    template_id: str
    requirement_id: str
    requirement_key: str
    title: str
    mode: str
    run_class: str
    pod_class: Optional[str]
    epistemic_status: str
    wm: str = "economic_world_model"
    subsystem: str = ""
    blocker: str = ""
    launch_allowed: bool = False
    blocked_by: list[str] = field(default_factory=list)
    command_templates: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)
    required_artifacts: list[str] = field(default_factory=list)
    config_paths: list[str] = field(default_factory=list)
    dependency_chain: list[str] = field(default_factory=list)
    expected_value: str = ""
    estimated_cost_usd: Optional[float] = None
    urgency: str = "medium"
    image: str = "template-only"
    template: str = "template-only"
    gpu_class: Optional[str] = None
    seeds: list[int] = field(default_factory=list)
    local_verification_available: bool = False
    current_status: str = "missing"
    promotion_gate: bool = True
    promotion_eligible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_PROVIDER_RUN_TEMPLATE_VERSION

    def to_manifest_stub(
        self, *, commit_sha: str = "", branch: str = ""
    ) -> Dict[str, Any]:
        """Return a run-manifest-shaped template, not a real run manifest."""

        digest = sha256_json(
            {
                "template_id": self.template_id,
                "requirement_key": self.requirement_key,
                "version": self.version,
            }
        )[:6]
        run_mode = self.mode if self.mode in {"local", "runpod"} else "runpod"
        return {
            "run_id": f"{run_mode}-19700101-000000-{digest}",
            "mode": run_mode,
            "pod_class": self.pod_class if run_mode == "runpod" else None,
            "run_class": self.run_class,
            "epistemic_status": self.epistemic_status,
            "commit_sha": commit_sha or "template_commit_sha_required_at_launch",
            "branch": branch or "template_branch_required_at_launch",
            "task": f"[TEMPLATE ONLY] {self.title}",
            "wm": self.wm,
            "subsystem": self.subsystem,
            "blocker": self.blocker,
            "config_paths": list(self.config_paths),
            "seeds": list(self.seeds),
            "image": self.image,
            "template": self.template,
            "pod_id": None,
            "volume_id": None,
            "commands": list(self.command_templates),
            "artifact_paths": list(self.artifact_paths),
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "cost_snapshot": None,
            "gpu_class": self.gpu_class,
            "wall_clock_seconds": None,
            "artifact_size_bytes": None,
            "storage_or_checkpoint_size_bytes": None,
            "expected_value": self.expected_value or None,
            "estimated_cost_usd": self.estimated_cost_usd,
            "dependency_chain": list(self.dependency_chain),
            "urgency": self.urgency,
            "justified_itself": "unclear",
            "rollback_notes": (
                "Template only. If this was launched without replacing guards, "
                "discard the run and do not treat artifacts as evidence."
            ),
            "replay_notes": (
                "Replace TEMPLATE_ONLY guards with real commands, set a fresh run_id, "
                "then record pod_id/timestamps/costs/artifacts after execution."
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "requirement_id": self.requirement_id,
            "requirement_key": self.requirement_key,
            "title": self.title,
            "mode": self.mode,
            "run_class": self.run_class,
            "pod_class": self.pod_class,
            "epistemic_status": self.epistemic_status,
            "wm": self.wm,
            "subsystem": self.subsystem,
            "blocker": self.blocker,
            "launch_allowed": bool(self.launch_allowed),
            "blocked_by": list(self.blocked_by),
            "command_templates": list(self.command_templates),
            "artifact_paths": list(self.artifact_paths),
            "required_artifacts": list(self.required_artifacts),
            "config_paths": list(self.config_paths),
            "dependency_chain": list(self.dependency_chain),
            "expected_value": self.expected_value,
            "estimated_cost_usd": self.estimated_cost_usd,
            "urgency": self.urgency,
            "image": self.image,
            "template": self.template,
            "gpu_class": self.gpu_class,
            "seeds": list(self.seeds),
            "local_verification_available": bool(self.local_verification_available),
            "current_status": self.current_status,
            "promotion_gate": bool(self.promotion_gate),
            "promotion_eligible": bool(self.promotion_eligible),
            "manifest_stub": self.to_manifest_stub(),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMProviderRunTemplate":
        return cls(
            template_id=str(payload.get("template_id", "")),
            requirement_id=str(payload.get("requirement_id", "")),
            requirement_key=str(payload.get("requirement_key", "")),
            title=str(payload.get("title", "")),
            mode=str(payload.get("mode", "runpod")),
            run_class=str(payload.get("run_class", "provider")),
            pod_class=(
                None
                if payload.get("pod_class") is None
                else str(payload.get("pod_class"))
            ),
            epistemic_status=str(payload.get("epistemic_status", "proof_of_life")),
            wm=str(payload.get("wm", "economic_world_model")),
            subsystem=str(payload.get("subsystem", "")),
            blocker=str(payload.get("blocker", "")),
            launch_allowed=bool(payload.get("launch_allowed", False)),
            blocked_by=[
                str(item) for item in list(payload.get("blocked_by", []) or [])
            ],
            command_templates=[
                str(item) for item in list(payload.get("command_templates", []) or [])
            ],
            artifact_paths=[
                str(item) for item in list(payload.get("artifact_paths", []) or [])
            ],
            required_artifacts=[
                str(item) for item in list(payload.get("required_artifacts", []) or [])
            ],
            config_paths=[
                str(item) for item in list(payload.get("config_paths", []) or [])
            ],
            dependency_chain=[
                str(item) for item in list(payload.get("dependency_chain", []) or [])
            ],
            expected_value=str(payload.get("expected_value", "")),
            estimated_cost_usd=_float_or_none(payload.get("estimated_cost_usd")),
            urgency=str(payload.get("urgency", "medium")),
            image=str(payload.get("image", "template-only")),
            template=str(payload.get("template", "template-only")),
            gpu_class=(
                None
                if payload.get("gpu_class") is None
                else str(payload.get("gpu_class"))
            ),
            seeds=[int(item) for item in list(payload.get("seeds", []) or [])],
            local_verification_available=bool(
                payload.get("local_verification_available", False)
            ),
            current_status=str(payload.get("current_status", "missing")),
            promotion_gate=bool(payload.get("promotion_gate", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_PROVIDER_RUN_TEMPLATE_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMProviderRunbook:
    """Template-only runbook compiled from an Economic WM evidence contract."""

    runbook_id: str
    contract_id: str
    templates: list[EconomicWMProviderRunTemplate] = field(default_factory=list)
    launch_allowed: bool = False
    provider_bringup_ready: bool = False
    gpu_training_ready: bool = False
    promotion_eligible: bool = False
    reward_math_mutation: bool = False
    authority_class: str = "runbook_template_only"
    training_blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_PROVIDER_RUNBOOK_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runbook_id": self.runbook_id,
            "version": self.version,
            "contract_id": self.contract_id,
            "templates": [template.to_dict() for template in self.templates],
            "launch_allowed": bool(self.launch_allowed),
            "provider_bringup_ready": bool(self.provider_bringup_ready),
            "gpu_training_ready": bool(self.gpu_training_ready),
            "promotion_eligible": bool(self.promotion_eligible),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "authority_class": self.authority_class,
            "training_blockers": list(self.training_blockers),
            "aggregate_counts": {
                str(key): float(value) for key, value in self.aggregate_counts.items()
            },
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMProviderRunbook":
        return cls(
            runbook_id=str(payload.get("runbook_id", "")),
            contract_id=str(payload.get("contract_id", "")),
            templates=[
                EconomicWMProviderRunTemplate.from_dict(item)
                for item in list(payload.get("templates", []) or [])
            ],
            launch_allowed=bool(payload.get("launch_allowed", False)),
            provider_bringup_ready=bool(payload.get("provider_bringup_ready", False)),
            gpu_training_ready=bool(payload.get("gpu_training_ready", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            authority_class=str(
                payload.get("authority_class", "runbook_template_only")
            ),
            training_blockers=[
                str(item) for item in list(payload.get("training_blockers", []) or [])
            ],
            aggregate_counts={
                str(key): float(value)
                for key, value in dict(
                    payload.get("aggregate_counts", {}) or {}
                ).items()
            },
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_WM_PROVIDER_RUNBOOK_VERSION)),
        )


def _generic_spec(requirement: EconomicWMEvidenceRequirement) -> _RunTemplateSpec:
    blocker = (
        requirement.blockers[0] if requirement.blockers else requirement.current_status
    )
    return _RunTemplateSpec(
        title=f"Evidence run for {requirement.requirement_key}",
        mode="runpod",
        run_class="provider",
        pod_class="provider",
        epistemic_status="proof_of_life",
        subsystem=requirement.provider_family or "external_provider_runtime",
        blocker=blocker,
        command_templates=[_TEMPLATE_GUARD],
        artifact_paths=[
            f"artifacts/economic_world_model/provider_runs/{{run_id}}/{requirement.requirement_key}.json"
        ],
        dependency_chain=["economic_wm_teacher_provider_contract_v1"],
        expected_value=f"Produce missing evidence for {requirement.requirement_key}.",
    )


def _build_template(
    *,
    contract_id: str,
    requirement: EconomicWMEvidenceRequirement,
) -> EconomicWMProviderRunTemplate:
    spec = _TEMPLATE_SPECS.get(requirement.requirement_key) or _generic_spec(
        requirement
    )
    blocker_set = _unique([*requirement.blockers, spec.blocker])
    payload = {
        "contract_id": contract_id,
        "requirement_id": requirement.requirement_id,
        "requirement_key": requirement.requirement_key,
        "mode": spec.mode,
        "run_class": spec.run_class,
        "pod_class": spec.pod_class,
        "epistemic_status": spec.epistemic_status,
        "blocker": spec.blocker,
        "version": ECONOMIC_WM_PROVIDER_RUN_TEMPLATE_VERSION,
    }
    return EconomicWMProviderRunTemplate(
        template_id=f"ewm_run_template_{sha256_json(payload)[:16]}",
        requirement_id=requirement.requirement_id,
        requirement_key=requirement.requirement_key,
        title=spec.title,
        mode=spec.mode,
        run_class=spec.run_class,
        pod_class=spec.pod_class,
        epistemic_status=spec.epistemic_status,
        subsystem=spec.subsystem,
        blocker=spec.blocker,
        launch_allowed=False,
        blocked_by=blocker_set,
        command_templates=list(spec.command_templates),
        artifact_paths=list(spec.artifact_paths),
        required_artifacts=list(requirement.required_artifacts),
        config_paths=list(spec.config_paths),
        dependency_chain=_unique(
            [*spec.dependency_chain, requirement.requirement_key, contract_id]
        ),
        expected_value=spec.expected_value,
        estimated_cost_usd=spec.estimated_cost_usd,
        urgency=spec.urgency,
        image=spec.image,
        template=spec.template,
        gpu_class=spec.gpu_class,
        seeds=list(spec.seeds),
        local_verification_available=spec.local_verification_available,
        current_status=requirement.current_status,
        promotion_gate=requirement.promotion_gate,
        promotion_eligible=False,
        metadata={
            "template_only": True,
            "requirement_satisfaction_score": requirement.satisfaction_score,
            "provider_family": requirement.provider_family,
            "evidence_kind": requirement.evidence_kind,
            "requirement_metadata": requirement.metadata,
        },
    )


def build_economic_wm_provider_runbook(
    *,
    contract: EconomicWMTeacherProviderContract,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMProviderRunbook:
    """Compile template-only run plans from an Economic WM evidence contract."""

    templates = [
        _build_template(contract_id=contract.contract_id, requirement=requirement)
        for requirement in contract.requirements
    ]
    blocker_set = _unique(
        [
            *contract.training_blockers,
            *[blocker for template in templates for blocker in template.blocked_by],
        ]
    )
    payload = {
        "contract_id": contract.contract_id,
        "template_ids": [template.template_id for template in templates],
        "version": ECONOMIC_WM_PROVIDER_RUNBOOK_VERSION,
    }
    aggregate_counts = {
        "template_count": float(len(templates)),
        "runpod_template_count": float(
            sum(1 for template in templates if template.mode == "runpod")
        ),
        "local_template_count": float(
            sum(1 for template in templates if template.mode == "local")
        ),
        "provider_run_class_count": float(
            sum(1 for template in templates if template.run_class == "provider")
        ),
        "train_run_class_count": float(
            sum(1 for template in templates if template.run_class == "train")
        ),
        "blocked_template_count": float(
            sum(1 for template in templates if not template.launch_allowed)
        ),
    }
    return EconomicWMProviderRunbook(
        runbook_id=f"ewm_provider_runbook_{sha256_json(payload)[:16]}",
        contract_id=contract.contract_id,
        templates=templates,
        launch_allowed=False,
        provider_bringup_ready=False,
        gpu_training_ready=False,
        promotion_eligible=False,
        reward_math_mutation=False,
        authority_class="runbook_template_only",
        training_blockers=blocker_set,
        aggregate_counts=aggregate_counts,
        artifact_refs={
            "economic_wm_teacher_provider_contract_id": contract.contract_id,
            **_mapping(contract.artifact_refs),
            **_mapping(artifact_refs),
        },
        metadata={
            "boundary": "template-only runbook; no provider/GPU/training/promotion claim",
            "template_only": True,
            "manifest_schema_ref": "docs/agent_ergonomics/run_manifest_schema.md",
            "source_authority_class": contract.authority_class,
            **_mapping(metadata),
        },
    )


def save_economic_wm_provider_runbook(
    path: str | Path, runbook: EconomicWMProviderRunbook
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(runbook.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_economic_wm_provider_runbook(path: str | Path) -> EconomicWMProviderRunbook:
    return EconomicWMProviderRunbook.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def build_economic_wm_provider_runbook_from_contract_path(
    *,
    contract_path: str | Path,
    output_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMProviderRunbook:
    contract = load_economic_wm_teacher_provider_contract(contract_path)
    runbook = build_economic_wm_provider_runbook(
        contract=contract,
        artifact_refs={"contract_path": str(contract_path), **_mapping(artifact_refs)},
        metadata=metadata,
    )
    save_economic_wm_provider_runbook(output_path, runbook)
    return runbook


__all__ = [
    "ECONOMIC_WM_PROVIDER_RUN_TEMPLATE_VERSION",
    "ECONOMIC_WM_PROVIDER_RUNBOOK_VERSION",
    "EconomicWMProviderRunTemplate",
    "EconomicWMProviderRunbook",
    "build_economic_wm_provider_runbook",
    "build_economic_wm_provider_runbook_from_contract_path",
    "load_economic_wm_provider_runbook",
    "save_economic_wm_provider_runbook",
]
