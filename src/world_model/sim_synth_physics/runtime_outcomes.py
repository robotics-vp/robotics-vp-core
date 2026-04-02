"""Upstream runtime output contracts and outcome harvesting for Phase-1 backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import mapping, stable_id, strings
from .receipts import BackendRuntimeLaunchReceipt, BackendRuntimeOutcomeReceipt
from .runtime_outcome_parsers import summarize_runtime_output_artifacts


ISAAC_OUTPUT_SOURCES = {
    "unitree_sim_isaaclab": [
        {
            "source_id": "runtime_root",
            "artifact_kind": "runtime_outputs",
            "root_kind": "launch_root",
            "patterns": [
                "logs/**/*.json",
                "logs/**/*.yaml",
                "logs/**/*.yml",
                "logs/**/*.pt",
                "logs/**/*.onnx",
                "logs/**/*.csv",
                "recordings/**/*",
                "generated/**/*.json",
            ],
            "upstream_hint": "Unitree IsaacLab launch outputs and logs.",
        },
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Runtime policies and deploy manifests.",
        },
        {
            "source_id": "robot_assets",
            "artifact_kind": "robot_assets",
            "root_kind": "target_ids",
            "target_ids": ["unitree_asset_root", "unitree_model_root"],
            "patterns": ["**/*.usd", "**/*.usda", "**/*.urdf", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Unitree robot descriptions and calibration manifests.",
        },
    ],
    "unitree_rl_gym": [
        {
            "source_id": "runtime_root",
            "artifact_kind": "runtime_outputs",
            "root_kind": "launch_root",
            "patterns": [
                "logs/**/*.pt",
                "logs/**/*.onnx",
                "logs/**/*.yaml",
                "logs/**/*.json",
                "exported/**/*",
                "deploy_real/**/*",
            ],
            "upstream_hint": "Unitree RL Gym checkpoints, deploy configs, and logs.",
        },
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Policy-bank checkpoints and deploy configs.",
        },
    ],
    "unitree_lerobot": [
        {
            "source_id": "runtime_root",
            "artifact_kind": "runtime_outputs",
            "root_kind": "launch_root",
            "patterns": [
                "outputs/**/*",
                "episodes/**/*",
                "replay/**/*",
                "metrics/**/*.json",
                "logs/**/*.json",
                "logs/**/*.yaml",
            ],
            "upstream_hint": "Unitree LeRobot eval outputs, replay captures, and metrics.",
        },
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "LeRobot checkpoints and deploy configs.",
        },
        {
            "source_id": "robot_assets",
            "artifact_kind": "robot_assets",
            "root_kind": "target_ids",
            "target_ids": ["unitree_asset_root", "unitree_model_root"],
            "patterns": ["**/*.usd", "**/*.usda", "**/*.urdf", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Unitree robot descriptions and calibration manifests for LeRobot eval.",
        },
    ],
    "humanoidverse": [
        {
            "source_id": "runtime_root",
            "artifact_kind": "runtime_outputs",
            "root_kind": "launch_root",
            "patterns": [
                "logs/**/*.pt",
                "logs/**/*.onnx",
                "logs/**/*.yaml",
                "logs/**/*.json",
                "outputs/**/*",
            ],
            "upstream_hint": "HumanoidVerse logs, checkpoints, and eval outputs.",
        },
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Humanoid policy bank.",
        },
    ],
    "isaaclab_core": [
        {
            "source_id": "runtime_root",
            "artifact_kind": "runtime_outputs",
            "root_kind": "launch_root",
            "patterns": ["logs/**/*.pt", "logs/**/*.onnx", "logs/**/*.yaml", "logs/**/*.json"],
            "upstream_hint": "Isaac Lab task outputs and logs.",
        },
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Isaac policy bank and deploy configs.",
        },
    ],
    "xr_teleoperate": [
        {
            "source_id": "runtime_root",
            "artifact_kind": "runtime_outputs",
            "root_kind": "launch_root",
            "patterns": [
                "teleop/utils/data/**/*",
                "teleop/**/*.json",
                "teleop/**/*.yaml",
                "teleop/**/*.csv",
            ],
            "upstream_hint": "XR teleop data captures and runtime metadata.",
        },
        {
            "source_id": "teleop_certs",
            "artifact_kind": "deploy_contracts",
            "root_kind": "launch_root",
            "patterns": ["teleop/televuer/*.pem", "teleop/televuer/*.key", "teleop/televuer/*.cnf"],
            "upstream_hint": "XR teleop certs and deploy-time config.",
        },
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Teleop/runtime policy bank.",
        },
    ],
}

HOLOSOMA_OUTPUT_SOURCES = {
    "holosoma_repo": [
        {
            "source_id": "runtime_root",
            "artifact_kind": "runtime_outputs",
            "root_kind": "launch_root",
            "patterns": [
                "logs/**/*.json",
                "logs/**/*.yaml",
                "logs/**/*.csv",
                "outputs/**/*",
                "runs/**/*",
                "checkpoints/**/*",
            ],
            "upstream_hint": "Holosoma eval/train outputs and logs.",
        },
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Holosoma policy bank.",
        },
    ],
    "holosoma_motion_bank": [
        {
            "source_id": "motion_bank",
            "artifact_kind": "motion_bank",
            "root_kind": "target_ids",
            "target_ids": ["holosoma_motion_root"],
            "patterns": ["**/*.npz", "**/*.npy", "**/*.bvh", "**/*.pkl", "**/*.json"],
            "upstream_hint": "Holosoma motion datasets and clips.",
        }
    ],
    "holosoma_policy_bank": [
        {
            "source_id": "policy_bank",
            "artifact_kind": "policy_bank",
            "root_kind": "policy_root",
            "patterns": ["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml", "**/*.json"],
            "upstream_hint": "Holosoma policy bank.",
        }
    ],
    "retargeting_bundle": [
        {
            "source_id": "retargeting_bundle",
            "artifact_kind": "retargeting_bundle",
            "root_kind": "target_ids",
            "target_ids": ["retargeting_root"],
            "patterns": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.npz", "**/*.pkl"],
            "upstream_hint": "Whole-body retargeting contracts and calibration.",
        }
    ],
}


def _target_ref(runtime_target_contract: Mapping[str, Any], target_ids: list[str]) -> str:
    for row in list(runtime_target_contract.get("targets", []) or []):
        row_mapping = mapping(row)
        target_id = str(row_mapping.get("target_id", "") or "")
        if target_id in target_ids:
            ref = str(row_mapping.get("ref", "") or "")
            if ref:
                return ref
    return ""


def _policy_root(runtime_bundle: Mapping[str, Any]) -> str:
    policy_contract = mapping(runtime_bundle.get("policy_contract"))
    return str(policy_contract.get("policy_root", "") or "")


def _policy_ref(runtime_bundle: Mapping[str, Any], launch_spec: Mapping[str, Any]) -> str:
    spec = mapping(launch_spec)
    policy_contract = mapping(runtime_bundle.get("policy_contract"))
    return str(spec.get("policy_ref", policy_contract.get("policy_ref", "")) or "")


def _selected_refs(
    runtime_bundle: Mapping[str, Any],
    launch_spec: Mapping[str, Any],
) -> dict[str, str]:
    bundle = mapping(runtime_bundle)
    spec = mapping(launch_spec)
    runtime_binding = mapping(bundle.get("runtime_binding")) or mapping(spec.get("runtime_binding"))
    return {
        "policy_ref": str(
            runtime_binding.get("selected_policy_ref")
            or _policy_ref(bundle, spec)
            or ""
        ),
        "deploy_config_ref": str(runtime_binding.get("selected_deploy_config", "") or ""),
        "runtime_report_ref": str(runtime_binding.get("selected_runtime_report", "") or ""),
        "policy_ref_source": str(runtime_binding.get("selected_policy_ref_source", "") or ""),
        "deploy_config_ref_source": str(
            runtime_binding.get("selected_deploy_config_source", "") or ""
        ),
        "runtime_report_ref_source": str(
            runtime_binding.get("selected_runtime_report_source", "") or ""
        ),
    }


def _source_root(
    runtime_bundle: Mapping[str, Any],
    launch_spec: Mapping[str, Any],
    source_spec: Mapping[str, Any],
) -> str:
    bundle = mapping(runtime_bundle)
    spec = mapping(launch_spec)
    root_kind = str(source_spec.get("root_kind", "") or "")
    if root_kind == "launch_root":
        return str(spec.get("root", "") or "")
    if root_kind == "policy_root":
        return _policy_root(bundle)
    if root_kind == "target_ids":
        return _target_ref(mapping(bundle.get("runtime_target_contract")), strings(source_spec.get("target_ids")))
    return ""


def _output_specs(backend: str, profile_id: str) -> list[dict[str, Any]]:
    if backend == "isaac":
        return [mapping(item) for item in ISAAC_OUTPUT_SOURCES.get(profile_id, [])]
    return [mapping(item) for item in HOLOSOMA_OUTPUT_SOURCES.get(profile_id, [])]


def build_backend_runtime_output_contract(
    runtime_bundle: Mapping[str, Any],
    launch_spec: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = mapping(runtime_bundle)
    spec = mapping(launch_spec)
    backend = str(bundle.get("backend", spec.get("backend", "")) or "")
    profile_id = str(spec.get("preferred_profile", bundle.get("preferred_profile", "")) or "")
    selected_refs = _selected_refs(bundle, spec)
    policy_ref = str(selected_refs.get("policy_ref", "") or "")
    sources: list[dict[str, Any]] = []
    for source_spec in _output_specs(backend, profile_id):
        root = _source_root(bundle, spec, source_spec)
        exact_refs = []
        artifact_kind = str(source_spec.get("artifact_kind", "") or "")
        if artifact_kind == "policy_bank" and policy_ref:
            policy_path = Path(policy_ref)
            if policy_path.exists():
                exact_refs.append(str(policy_path.resolve()))
        if artifact_kind in {"policy_bank", "deploy_contracts", "runtime_outputs"}:
            deploy_config_ref = str(selected_refs.get("deploy_config_ref", "") or "")
            if deploy_config_ref:
                deploy_path = Path(deploy_config_ref)
                if deploy_path.exists():
                    exact_refs.append(str(deploy_path.resolve()))
        if artifact_kind in {"runtime_outputs", "policy_bank"}:
            runtime_report_ref = str(selected_refs.get("runtime_report_ref", "") or "")
            if runtime_report_ref:
                runtime_report_path = Path(runtime_report_ref)
                if runtime_report_path.exists():
                    exact_refs.append(str(runtime_report_path.resolve()))
        sources.append(
            {
                "source_id": str(source_spec.get("source_id", "") or ""),
                "artifact_kind": str(source_spec.get("artifact_kind", "") or ""),
                "root": root,
                "root_exists": bool(root and Path(root).exists()),
                "patterns": strings(source_spec.get("patterns")),
                "exact_refs": exact_refs,
                "upstream_hint": str(source_spec.get("upstream_hint", "") or ""),
            }
        )
    payload = {
        "backend": backend,
        "profile_id": profile_id,
        "policy_ref": policy_ref,
        "expected_deploy_config_ref": str(selected_refs.get("deploy_config_ref", "") or ""),
        "expected_runtime_report_ref": str(selected_refs.get("runtime_report_ref", "") or ""),
        "source_count": len(sources),
    }
    return {
        "version": "backend_runtime_output_contract_v1",
        "contract_id": stable_id("backend_runtime_output_contract", payload),
        "backend": backend,
        "profile_id": profile_id,
        "policy_ref": policy_ref,
        "expected_policy_ref": policy_ref,
        "expected_policy_ref_source": str(selected_refs.get("policy_ref_source", "") or ""),
        "expected_deploy_config_ref": str(selected_refs.get("deploy_config_ref", "") or ""),
        "expected_deploy_config_ref_source": str(
            selected_refs.get("deploy_config_ref_source", "") or ""
        ),
        "expected_runtime_report_ref": str(selected_refs.get("runtime_report_ref", "") or ""),
        "expected_runtime_report_ref_source": str(
            selected_refs.get("runtime_report_ref_source", "") or ""
        ),
        "sources": sources,
    }


def _validate_selected_output_refs(
    output_contract: Mapping[str, Any],
    artifact_refs: Sequence[str],
    structured_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    contract = mapping(output_contract)
    structured = mapping(structured_outputs)
    harvested_ref_set = {str(ref) for ref in strings(artifact_refs)}
    policy_refs = set(strings(structured.get("policy_checkpoint_refs")))
    deploy_refs = set(strings(structured.get("deploy_config_refs")))
    expected_rows = [
        (
            "policy_ref",
            str(contract.get("expected_policy_ref", "") or ""),
            str(contract.get("expected_policy_ref_source", "") or ""),
        ),
        (
            "deploy_config_ref",
            str(contract.get("expected_deploy_config_ref", "") or ""),
            str(contract.get("expected_deploy_config_ref_source", "") or ""),
        ),
        (
            "runtime_report_ref",
            str(contract.get("expected_runtime_report_ref", "") or ""),
            str(contract.get("expected_runtime_report_ref_source", "") or ""),
        ),
    ]
    expected_components = [component for component, ref, _source in expected_rows if ref]
    matched_components: list[str] = []
    mismatched_components: list[str] = []
    missing_components: list[str] = []
    expected_refs: dict[str, str] = {}
    expected_sources: dict[str, str] = {}
    for component, ref, source in expected_rows:
        if not ref:
            continue
        expected_refs[component] = ref
        expected_sources[component] = source
        if component == "policy_ref":
            matched = ref in policy_refs or ref in harvested_ref_set
        elif component == "deploy_config_ref":
            matched = ref in deploy_refs or ref in harvested_ref_set
        else:
            matched = ref in harvested_ref_set
        if matched:
            matched_components.append(component)
        elif harvested_ref_set:
            mismatched_components.append(component)
        else:
            missing_components.append(component)
    if not expected_components:
        status = "no_expected_selected_refs"
    elif not mismatched_components and not missing_components:
        status = "selected_refs_matched"
    elif matched_components:
        status = "selected_refs_partial"
    elif mismatched_components:
        status = "selected_refs_mismatched"
    else:
        status = "selected_refs_missing"
    return {
        "status": status,
        "expected_components": expected_components,
        "matched_components": matched_components,
        "mismatched_components": mismatched_components,
        "missing_components": missing_components,
        "expected_refs": expected_refs,
        "expected_ref_sources": expected_sources,
    }


def _collect_matches(root: Path, patterns: list[str], *, limit: int) -> list[str]:
    matches: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(resolved)
            if len(matches) >= limit:
                return matches
    return matches


def harvest_backend_runtime_outcomes(
    output_contract: Mapping[str, Any],
    *,
    executed: bool,
    limit_per_source: int = 10,
    explicit_artifact_refs: Sequence[str] | None = None,
    explicit_policy_ref: str = "",
) -> dict[str, Any]:
    contract = mapping(output_contract)
    sources = [mapping(item) for item in list(contract.get("sources", []) or [])]
    harvested_artifacts: list[str] = []
    source_summaries: list[dict[str, Any]] = []
    kind_counts: dict[str, int] = {}
    any_ready_source = False
    for source in sources:
        root = str(source.get("root", "") or "")
        exact_refs = []
        if executed:
            exact_refs = [
                str(Path(ref).resolve())
                for ref in strings(source.get("exact_refs"))
                if ref and Path(ref).exists()
            ]
        root_exists = bool(root and Path(root).exists())
        if root_exists or exact_refs:
            any_ready_source = True
        matched_refs = list(exact_refs)
        if executed and root_exists:
            matched_refs.extend(
                _collect_matches(Path(root), strings(source.get("patterns")), limit=limit_per_source)
            )
        deduped_refs: list[str] = []
        seen: set[str] = set()
        for ref in matched_refs:
            if ref in seen:
                continue
            seen.add(ref)
            deduped_refs.append(ref)
        harvested_artifacts.extend(ref for ref in deduped_refs if ref not in harvested_artifacts)
        artifact_kind = str(source.get("artifact_kind", "") or "unknown")
        kind_counts[artifact_kind] = kind_counts.get(artifact_kind, 0) + len(deduped_refs)
        source_summaries.append(
            {
                "source_id": str(source.get("source_id", "") or ""),
                "artifact_kind": artifact_kind,
                "root": root,
                "root_exists": root_exists,
                "matched_refs": deduped_refs,
                "matched_count": len(deduped_refs),
                "upstream_hint": str(source.get("upstream_hint", "") or ""),
            }
        )
    explicit_refs: list[str] = []
    for ref in strings(explicit_artifact_refs):
        if ref and Path(ref).exists():
            resolved = str(Path(ref).resolve())
            if resolved not in explicit_refs:
                explicit_refs.append(resolved)
    if explicit_policy_ref and Path(explicit_policy_ref).exists():
        resolved_policy_ref = str(Path(explicit_policy_ref).resolve())
        if resolved_policy_ref not in explicit_refs:
            explicit_refs.append(resolved_policy_ref)
    if explicit_refs:
        any_ready_source = True
        for ref in explicit_refs:
            if ref not in harvested_artifacts:
                harvested_artifacts.append(ref)
        source_summaries.append(
            {
                "source_id": "explicit_runtime_artifacts",
                "artifact_kind": "runtime_execution_artifacts",
                "root": "",
                "root_exists": True,
                "matched_refs": explicit_refs,
                "matched_count": len(explicit_refs),
                "upstream_hint": "Artifacts emitted directly by concrete local runtime execution.",
            }
        )
    if not executed:
        outcome_status = "launch_not_executed"
    elif harvested_artifacts:
        outcome_status = "runtime_outputs_harvested"
    elif any_ready_source:
        outcome_status = "runtime_outputs_missing"
    else:
        outcome_status = "outcome_sources_missing"
    structured_outputs = summarize_runtime_output_artifacts(harvested_artifacts)
    selected_ref_validation = _validate_selected_output_refs(
        contract,
        harvested_artifacts,
        structured_outputs,
    )
    return {
        "version": "backend_runtime_output_summary_v1",
        "backend": str(contract.get("backend", "") or ""),
        "profile_id": str(contract.get("profile_id", "") or ""),
        "executed": bool(executed),
        "outcome_status": outcome_status,
        "harvested_output_count": len(harvested_artifacts),
        "artifact_kind_counts": kind_counts,
        "source_summaries": source_summaries,
        "artifact_refs": harvested_artifacts,
        "structured_outputs": structured_outputs,
        "selected_ref_validation": selected_ref_validation,
        "output_contract": contract,
    }


def build_backend_runtime_outcome_receipt(
    *,
    runtime_bundle: Mapping[str, Any],
    launch_receipt: BackendRuntimeLaunchReceipt | None,
    output_summary: Mapping[str, Any],
    artifact_refs: list[str] | None = None,
) -> BackendRuntimeOutcomeReceipt:
    bundle = mapping(runtime_bundle)
    summary = mapping(output_summary)
    executed = bool(summary.get("executed", False))
    launch_receipt_id = ""
    launch_status = ""
    if launch_receipt is not None:
        executed = bool(launch_receipt.executed)
        launch_receipt_id = launch_receipt.receipt_id
        launch_status = launch_receipt.launch_status
    payload = {
        "backend": str(bundle.get("backend", "") or ""),
        "profile_id": str(summary.get("profile_id", "") or ""),
        "outcome_status": str(summary.get("outcome_status", "") or ""),
        "executed": executed,
        "harvested_output_count": int(summary.get("harvested_output_count", 0) or 0),
        "launch_status": launch_status,
    }
    return BackendRuntimeOutcomeReceipt(
        receipt_id=stable_id("backend_runtime_outcome_receipt", payload),
        backend=str(bundle.get("backend", "") or ""),
        outcome_profile=str(summary.get("profile_id", "") or ""),
        outcome_status=str(summary.get("outcome_status", "") or ""),
        executed=executed,
        harvested_output_count=int(summary.get("harvested_output_count", 0) or 0),
        artifact_refs=strings(artifact_refs or summary.get("artifact_refs")),
        metadata={
            "launch_receipt_id": launch_receipt_id,
            "launch_status": launch_status,
            "harvest_mode": (
                "external_launch" if launch_receipt is not None else "local_runtime_execution"
            ),
            "artifact_kind_counts": mapping(summary.get("artifact_kind_counts")),
            "source_summaries": list(summary.get("source_summaries", []) or []),
            "structured_outputs": mapping(summary.get("structured_outputs")),
            "selected_ref_validation": mapping(summary.get("selected_ref_validation")),
            "output_contract": mapping(summary.get("output_contract")),
        },
    )


__all__ = [
    "build_backend_runtime_outcome_receipt",
    "build_backend_runtime_output_contract",
    "harvest_backend_runtime_outcomes",
]
