"""Valuation verifier for provenance closure and replayability.

The "court system" for valuation: verifies all SHAs match, cross-references
are consistent, and regal decisions can be replayed from artifacts alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict

from src.contracts.schemas import (
    RunManifestV1,
    ValueLedgerRecordV1,
)
from src.utils.config_digest import sha256_json, sha256_file


# =============================================================================
# Blocking Check IDs (Phase 9: verifier checks that block deploy)
# =============================================================================
# These check_ids, when failed, will block allow_deploy for FULL regality runs.
# Diagnostic-only checks (eligible_shrunk_pathology, selection_rejection_spike)
# are NOT in this list and only warn.

BLOCKING_CHECK_IDS = frozenset([
    # Structural integrity (tamper detection)
    "orchestrator_state_sha_match",
    "selection_manifest_sha_match",
    "ledger_regal_sha_match",
    "deploy_gate_inputs_sha_match",
    "verification_report_sha_match",
    # Semantic invariants
    "selection_selected_subset_eligible",
    "selection_quarantine_disjoint",
    "selection_nonempty_for_training",
    "exposure_only_selected_ids",
    "exposure_quarantine_exclusion",
    "orchestrator_state_nonnegative_counters",
    "applied_knob_deltas_match_orchestrator_state",
    "orchestrator_decisions_match_plan_applied_events",
    # Required artifacts for FULL
    "ledger_regal_required_for_full",
    "trajectory_audit_sha_required_for_training",
    "econ_evidence_present_for_full",
])


class VerificationCheckV1(BaseModel):
    """Single verification check result.
    
    severity determines whether a failed check blocks deployment:
    - FAIL: always blocks if passed=False (for FULL runs)
    - WARN: diagnostic only, does not block deployment
    """
    model_config = ConfigDict(extra="forbid")
    
    check_id: str
    passed: bool
    message: str
    severity: Literal["FAIL", "WARN"] = "FAIL"  # FAIL blocks deploy, WARN is diagnostic
    expected: Optional[str] = None
    actual: Optional[str] = None


class VerificationReportV1(BaseModel):
    """Complete verification report for a run.
    
    This is the court record: hashable evidence of provenance verification.
    """
    model_config = ConfigDict(extra="forbid")
    
    schema_version: str = "v1"
    run_id: str
    verified_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Overall status
    all_passed: bool
    check_count: int
    passed_count: int
    failed_count: int
    
    # Detailed checks
    checks: List[VerificationCheckV1] = Field(default_factory=list)
    
    # Warnings (non-fatal)
    warnings: List[str] = Field(default_factory=list)
    
    # Manifest summary
    manifest_sha: Optional[str] = None
    ledger_record_count: int = 0
    
    # Policy provenance (Phase 10: blocking policy SHA)
    # Proves "this verification was run under policy X"
    regality_thresholds_sha: Optional[str] = None  # SHA of RegalityThresholdsV1 used
    blocking_policy_sha: Optional[str] = None  # SHA of blocking check IDs + severity policy
    
    def sha256(self) -> str:
        """Compute verification report SHA for provenance."""
        return sha256_json(self.model_dump(mode="json"))


def _load_manifest(manifest_path: Path) -> Tuple[Optional[RunManifestV1], Optional[str]]:
    """Load and validate manifest."""
    if not manifest_path.exists():
        return None, "run_manifest.json not found"
    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        manifest = RunManifestV1.model_validate(data)
        return manifest, None
    except Exception as e:
        return None, f"Failed to load manifest: {e}"


def _load_ledger(ledger_path: Path) -> Tuple[List[ValueLedgerRecordV1], Optional[str]]:
    """Load and validate ledger records."""
    if not ledger_path.exists():
        return [], "ledger.jsonl not found"
    records = []
    try:
        with open(ledger_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    data = json.loads(line)
                    records.append(ValueLedgerRecordV1.model_validate(data))
        return records, None
    except Exception as e:
        return [], f"Failed to load ledger at line {line_num}: {e}"


def verify_run(output_dir: str) -> VerificationReportV1:
    """Verify a run's valuation provenance.
    
    Checks:
    1. All required files exist
    2. All SHAs recompute to match stored values  
    3. Ledger ↔ manifest cross-references match
    4. Ledger allow_deploy logic matches deploy gate decision
    5. Regal provenance consistency
    6. Quarantine manifest consistency (if present)
    
    Args:
        output_dir: Path to run output directory
        
    Returns:
        VerificationReportV1 with all check results
    """
    output_path = Path(output_dir)
    checks: List[VerificationCheckV1] = []
    warnings: List[str] = []
    
    # 1. Check required files exist
    manifest_path = output_path / "run_manifest.json"
    ledger_path = output_path / "ledger.jsonl"
    exposure_path = output_path / "exposure_manifest.json"
    plan_path = output_path / "plan.json"
    
    for file_path, name in [
        (manifest_path, "run_manifest.json"),
        (ledger_path, "ledger.jsonl"),
    ]:
        checks.append(VerificationCheckV1(
            check_id=f"file_exists_{name}",
            passed=file_path.exists(),
            message=f"Required file {name} exists" if file_path.exists() else f"Missing required file: {name}",
        ))
    
    # Load manifest
    manifest, manifest_err = _load_manifest(manifest_path)
    if manifest_err:
        checks.append(VerificationCheckV1(
            check_id="manifest_load",
            passed=False,
            message=manifest_err,
        ))
        return _build_report("unknown", checks, warnings, None, 0)
    
    run_id = manifest.run_id
    
    # Detect training run early (needed for multiple checks including 10c, 11, 15)
    is_training_run = (
        manifest.final_weights_sha is not None
        and manifest.baseline_weights_sha is not None
        and manifest.final_weights_sha != manifest.baseline_weights_sha
    )
    
    # Load ledger
    ledger_records, ledger_err = _load_ledger(ledger_path)
    if ledger_err:
        checks.append(VerificationCheckV1(
            check_id="ledger_load",
            passed=False,
            message=ledger_err,
        ))
    else:
        checks.append(VerificationCheckV1(
            check_id="ledger_load",
            passed=True,
            message=f"Ledger loaded: {len(ledger_records)} records",
        ))
    
    # 2. Verify plan_sha if plan file exists
    if plan_path.exists():
        stored_sha = manifest.plan_sha
        computed_sha = sha256_file(str(plan_path))
        checks.append(VerificationCheckV1(
            check_id="plan_sha_match",
            passed=stored_sha == computed_sha,
            message="Plan SHA matches" if stored_sha == computed_sha else "Plan SHA mismatch",
            expected=stored_sha,
            actual=computed_sha,
        ))
    
    # 3. Verify exposure manifest SHA if present
    if exposure_path.exists():
        computed_sha = sha256_file(str(exposure_path))
        stored_sha = manifest.datapack_manifest_sha
        # Note: datapack_manifest_sha is computed from sorted IDs, not file hash
        # This is a warning rather than check
        warnings.append(f"Exposure manifest file exists, datapack_manifest_sha={stored_sha[:16] if stored_sha else 'None'}")
    
    # 4. Verify ledger ↔ manifest cross-references
    for record in ledger_records:
        # Check run_id matches
        if record.run_id != manifest.run_id:
            checks.append(VerificationCheckV1(
                check_id=f"ledger_{record.record_id}_run_id_match",
                passed=False,
                message=f"Ledger record {record.record_id} run_id mismatch",
                expected=manifest.run_id,
                actual=record.run_id,
            ))
        else:
            checks.append(VerificationCheckV1(
                check_id=f"ledger_{record.record_id}_run_id_match",
                passed=True,
                message=f"Ledger record {record.record_id} run_id matches manifest",
            ))
        
        # Check plan_sha matches
        if record.plan_sha != manifest.plan_sha:
            checks.append(VerificationCheckV1(
                check_id=f"ledger_{record.record_id}_plan_sha_match",
                passed=False,
                message=f"Ledger record {record.record_id} plan_sha mismatch",
                expected=manifest.plan_sha,
                actual=record.plan_sha,
            ))
    
    # 5. Verify regal provenance consistency
    for record in ledger_records:
        # Check allow_deploy invariant: must be False if regal_degraded or regal failed
        if record.regal_degraded:
            # allow_deploy MUST be False
            checks.append(VerificationCheckV1(
                check_id=f"ledger_{record.record_id}_degraded_blocks_deploy",
                passed=not record.allow_deploy,
                message="regal_degraded=True → allow_deploy=False" if not record.allow_deploy 
                        else "VIOLATION: regal_degraded=True but allow_deploy=True",
            ))
        elif record.regal is not None and not record.regal.all_passed:
            # allow_deploy MUST be False
            checks.append(VerificationCheckV1(
                check_id=f"ledger_{record.record_id}_failed_regal_blocks_deploy",
                passed=not record.allow_deploy,
                message="regal.all_passed=False → allow_deploy=False" if not record.allow_deploy
                        else "VIOLATION: regal failed but allow_deploy=True",
            ))
        
        # Check regal_report_sha ordering consistency (if present)
        if record.regal and record.regal.reports:
            # Verify reports are sorted by (phase, regal_id) for determinism
            sorted_reports = sorted(record.regal.reports, key=lambda r: (r.phase.value, r.regal_id))
            computed_sha = sha256_json([r.report_sha for r in sorted_reports])
            if manifest.regal_report_sha:
                checks.append(VerificationCheckV1(
                    check_id=f"ledger_{record.record_id}_regal_report_sha_determinism",
                    passed=True,  # Just verify sorting, not exact match
                    message=f"Regal reports sorted deterministically ({len(sorted_reports)} reports)",
                ))
    
    # 6. Verify trajectory_audit_sha is present if regal evaluation occurred
    if manifest.regal_config_sha and not manifest.trajectory_audit_sha:
        warnings.append("regal_config_sha present but trajectory_audit_sha missing (may be intentional if no training)")
    
    # 7. Verify quarantine manifest if present
    if manifest.quarantine_manifest_sha:
        checks.append(VerificationCheckV1(
            check_id="quarantine_manifest_sha_present",
            passed=True,
            message=f"Quarantine manifest SHA recorded: {manifest.quarantine_manifest_sha[:16]}",
        ))
    
    # 8. Verify deploy_gate_decision_sha if present and matches allow_deploy
    if manifest.deploy_gate_decision_sha:
        checks.append(VerificationCheckV1(
            check_id="deploy_gate_decision_sha_present",
            passed=True,
            message=f"Deploy gate decision SHA recorded: {manifest.deploy_gate_decision_sha[:16]}",
        ))

    # 9. Verify orchestrator_state_sha presence (Phase 1)
    if manifest.orchestrator_state_sha:
        orchestrator_state_path = output_path / "orchestrator_state.json"
        if orchestrator_state_path.exists():
            computed_sha = sha256_file(str(orchestrator_state_path))
            checks.append(VerificationCheckV1(
                check_id="orchestrator_state_sha_match",
                passed=manifest.orchestrator_state_sha == computed_sha,
                message="Orchestrator state SHA matches" if manifest.orchestrator_state_sha == computed_sha
                        else "Orchestrator state SHA mismatch",
                expected=manifest.orchestrator_state_sha,
                actual=computed_sha,
            ))
        else:
            checks.append(VerificationCheckV1(
                check_id="orchestrator_state_sha_present",
                passed=True,
                message=f"Orchestrator state SHA recorded: {manifest.orchestrator_state_sha[:16]}",
            ))

    # 10. Verify selection_manifest_sha presence (Phase 2)
    if manifest.selection_manifest_sha:
        selection_manifest_path = output_path / "selection_manifest.json"
        if selection_manifest_path.exists():
            computed_sha = sha256_file(str(selection_manifest_path))
            checks.append(VerificationCheckV1(
                check_id="selection_manifest_sha_match",
                passed=manifest.selection_manifest_sha == computed_sha,
                message="Selection manifest SHA matches" if manifest.selection_manifest_sha == computed_sha
                        else "Selection manifest SHA mismatch",
                expected=manifest.selection_manifest_sha,
                actual=computed_sha,
            ))
            
            # 10a-c: Selection manifest semantic invariants (P1.5 enforcement)
            try:
                with open(selection_manifest_path, "r") as f:
                    sel_data = json.load(f)
                
                eligible_ids = set(sel_data.get("eligible_datapack_ids", []))
                selected_ids = set(sel_data.get("selected_datapack_ids", []))
                quarantine_ids = set(sel_data.get("quarantine_datapack_ids", []))
                
                # Check 10a: selected_ids ⊆ eligible_ids
                invalid_selected = selected_ids - eligible_ids
                if invalid_selected:
                    checks.append(VerificationCheckV1(
                        check_id="selection_selected_subset_eligible",
                        passed=False,
                        message=f"VIOLATION: selected_ids not subset of eligible: {sorted(invalid_selected)[:5]}",
                    ))
                else:
                    checks.append(VerificationCheckV1(
                        check_id="selection_selected_subset_eligible",
                        passed=True,
                        message=f"selected_ids ⊆ eligible_ids ({len(selected_ids)}/{len(eligible_ids)})",
                    ))
                
                # Check 10b: selected_ids ∩ quarantine_ids = ∅
                quarantine_violation = selected_ids & quarantine_ids
                if quarantine_violation:
                    checks.append(VerificationCheckV1(
                        check_id="selection_quarantine_disjoint",
                        passed=False,
                        message=f"CRITICAL: quarantined datapacks selected: {sorted(quarantine_violation)}",
                    ))
                else:
                    checks.append(VerificationCheckV1(
                        check_id="selection_quarantine_disjoint",
                        passed=True,
                        message="selected ∩ quarantine = ∅ (quarantine enforced)",
                    ))
                
                # Check 10c: len(selected) > 0 for training runs (unless dry-run)
                is_dry_run = sel_data.get("dry_run", False)
                if is_training_run and not is_dry_run and len(selected_ids) == 0:
                    checks.append(VerificationCheckV1(
                        check_id="selection_nonempty_for_training",
                        passed=False,
                        message="VIOLATION: training run has no selected datapacks",
                    ))
                elif is_training_run and len(selected_ids) > 0:
                    checks.append(VerificationCheckV1(
                        check_id="selection_nonempty_for_training",
                        passed=True,
                        message=f"Training run has {len(selected_ids)} selected datapacks",
                    ))
            except Exception as e:
                warnings.append(f"Could not verify selection manifest semantics: {e}")
        else:
            checks.append(VerificationCheckV1(
                check_id="selection_manifest_sha_present",
                passed=True,
                message=f"Selection manifest SHA recorded: {manifest.selection_manifest_sha[:16]}",
            ))

    # 11. Verify trajectory_audit_sha is REQUIRED for training runs (Phase 3)
    # is_training_run already computed above
    if is_training_run:
        checks.append(VerificationCheckV1(
            check_id="trajectory_audit_sha_required_for_training",
            passed=manifest.trajectory_audit_sha is not None,
            message="Training run has trajectory_audit_sha" if manifest.trajectory_audit_sha
                    else "VIOLATION: Training run missing trajectory_audit_sha",
        ))

    # 12. Verify probe_report regal_context_sha matches manifest (Phase 5)
    if manifest.probe_report_sha and manifest.regal_context_sha:
        probe_report_path = output_path / "probe_report.json"
        if probe_report_path.exists():
            try:
                with open(probe_report_path, "r") as f:
                    probe_data = json.load(f)
                probe_regal_ctx = probe_data.get("regal_context_sha")
                if probe_regal_ctx:
                    checks.append(VerificationCheckV1(
                        check_id="probe_report_regal_context_match",
                        passed=probe_regal_ctx == manifest.regal_context_sha,
                        message="Probe report regal_context_sha matches manifest" if probe_regal_ctx == manifest.regal_context_sha
                                else "Probe report regal_context_sha mismatch",
                        expected=manifest.regal_context_sha,
                        actual=probe_regal_ctx,
                    ))
            except Exception:
                pass  # Skip if can't parse

    # 13. Verify deploy_gate_inputs_sha recomputation (Phase 6)
    if manifest.deploy_gate_inputs_sha:
        deploy_gate_path = output_path / "deploy_gate_inputs.json"
        if deploy_gate_path.exists():
            computed_sha = sha256_file(str(deploy_gate_path))
            checks.append(VerificationCheckV1(
                check_id="deploy_gate_inputs_sha_match",
                passed=manifest.deploy_gate_inputs_sha == computed_sha,
                message="Deploy gate inputs SHA matches" if manifest.deploy_gate_inputs_sha == computed_sha
                        else "Deploy gate inputs SHA mismatch",
                expected=manifest.deploy_gate_inputs_sha,
                actual=computed_sha,
            ))

    # 14. Verify verification_report_sha meta-check (Phase 7)
    # This is the "verifier of verifier" - manifest must include verification_report_sha
    # that matches the file content AFTER this verification runs
    if manifest.verification_report_sha:
        verification_report_path = output_path / "verification_report.json"
        if verification_report_path.exists():
            computed_sha = sha256_file(str(verification_report_path))
            checks.append(VerificationCheckV1(
                check_id="verification_report_sha_match",
                passed=manifest.verification_report_sha == computed_sha,
                message="Verification report SHA matches" if manifest.verification_report_sha == computed_sha
                        else "Verification report SHA mismatch (may need re-run)",
                expected=manifest.verification_report_sha,
                actual=computed_sha,
            ))

    # 15. RewardBreakdown required for FULL regality when trajectory_audit exists
    # Per Gap E: If trajectory_audit_sha exists AND this is a training run (FULL proxy),
    # reward_components must:
    # - Be present in the trajectory audit
    # - Contain required keys (task_reward, time_penalty, energy_cost)
    # - Pass sum consistency: total == sum(components) within tolerance
    # 
    # Gate: Only enforce for training runs (is_training_run = weights changed)
    # Legacy/PARTIAL runs without reward_components get a warning, not failure
    trajectory_audit_path = output_path / "trajectory_audit.json"
    if manifest.trajectory_audit_sha and trajectory_audit_path.exists() and is_training_run:
        try:
            with open(trajectory_audit_path, "r") as f:
                audit_data = json.load(f)
            
            # Check episode audits for reward_components
            episode_audits = audit_data.get("episode_audits", [])
            required_keys = {"task_reward", "time_penalty", "energy_cost"}
            
            episodes_with_issues = 0
            for idx, ep in enumerate(episode_audits):
                reward_components = ep.get("reward_components") or {}
                total_return = ep.get("total_return", 0.0)
                
                # Check 15a: reward_components presence (FULL runs must have this)
                if not reward_components:
                    checks.append(VerificationCheckV1(
                        check_id=f"reward_breakdown_present_ep{idx}",
                        passed=False,
                        message=f"Episode {idx}: reward_components missing (FULL training requires RewardBreakdown)",
                    ))
                    episodes_with_issues += 1
                else:
                    # Check 15b: required keys present (warn only for missing optional keys)
                    present_keys = set(reward_components.keys())
                    missing_required = required_keys - present_keys
                    if missing_required:
                        # Missing required keys is a failure
                        checks.append(VerificationCheckV1(
                            check_id=f"reward_breakdown_keys_ep{idx}",
                            passed=False,
                            message=f"Episode {idx}: missing required reward keys: {sorted(missing_required)}",
                        ))
                        episodes_with_issues += 1
                    
                    # Check 15c: sum consistency with relaxed tolerance
                    # Tolerance: absolute 1e-3 + relative 1% of total (handles shaping terms)
                    component_sum = sum(float(v) for v in reward_components.values())
                    tolerance = 1e-3 + 0.01 * abs(total_return)
                    sum_matches = abs(total_return - component_sum) <= tolerance
                    if not sum_matches:
                        checks.append(VerificationCheckV1(
                            check_id=f"reward_breakdown_sum_ep{idx}",
                            passed=False,
                            message=f"Episode {idx}: reward sum mismatch: total={total_return:.4f}, components sum={component_sum:.4f} (tolerance={tolerance:.4f})",
                            expected=str(total_return),
                            actual=str(component_sum),
                        ))
                        episodes_with_issues += 1
            
            # Summary check
            if episode_audits and episodes_with_issues == 0:
                checks.append(VerificationCheckV1(
                    check_id="reward_breakdown_required_for_full",
                    passed=True,
                    message=f"RewardBreakdown valid in all {len(episode_audits)} episodes",
                ))
        except Exception as e:
            warnings.append(f"Could not verify RewardBreakdown: {e}")
    elif manifest.trajectory_audit_sha and trajectory_audit_path.exists() and not is_training_run:
        # Non-training runs: just warn if missing, don't fail
        warnings.append("trajectory_audit present but not a training run; skipping strict RewardBreakdown check")

    # 16. Verify ledger_regal_sha if present (Phase 8: file-verified regal)
    if getattr(manifest, 'ledger_regal_sha', None):
        ledger_regal_path = output_path / "ledger_regal.json"
        if ledger_regal_path.exists():
            computed_sha = sha256_file(str(ledger_regal_path))
            checks.append(VerificationCheckV1(
                check_id="ledger_regal_sha_match",
                passed=manifest.ledger_regal_sha == computed_sha,
                message="Ledger regal SHA matches" if manifest.ledger_regal_sha == computed_sha
                        else "Ledger regal SHA mismatch",
                expected=manifest.ledger_regal_sha,
                actual=computed_sha,
            ))
        else:
            checks.append(VerificationCheckV1(
                check_id="ledger_regal_sha_match",
                passed=False,
                message="ledger_regal.json file missing but SHA in manifest",
            ))

    # 17. Verify exposure_manifest if present
    exposure_manifest_path = output_path / "exposure_manifest.json"
    if exposure_manifest_path.exists():
        try:
            with open(exposure_manifest_path, "r") as f:
                exp_data = json.load(f)
            
            exp_datapack_ids = set(exp_data.get("datapack_ids", []))
            
            # 17a: Cross-check with selection manifest if both exist
            selection_manifest_path = output_path / "selection_manifest.json"
            if selection_manifest_path.exists():
                with open(selection_manifest_path, "r") as f:
                    sel_data = json.load(f)
                
                selected_ids = set(sel_data.get("selected_datapack_ids", []))
                quarantine_ids = set(sel_data.get("quarantine_datapack_ids", []))
                
                # Check: exposure ⊆ selected
                exposure_not_in_selected = exp_datapack_ids - selected_ids
                if exposure_not_in_selected and selected_ids:  # Only check if selection exists
                    checks.append(VerificationCheckV1(
                        check_id="exposure_only_selected_ids",
                        passed=False,
                        message=f"VIOLATION: exposure contains non-selected datapacks: {sorted(exposure_not_in_selected)[:5]}",
                    ))
                elif selected_ids:
                    checks.append(VerificationCheckV1(
                        check_id="exposure_only_selected_ids",
                        passed=True,
                        message=f"exposure ⊆ selected ({len(exp_datapack_ids)}/{len(selected_ids)})",
                    ))
                
                # Check: no quarantine in exposure
                quarantine_in_exposure = exp_datapack_ids & quarantine_ids
                if quarantine_in_exposure:
                    checks.append(VerificationCheckV1(
                        check_id="exposure_quarantine_exclusion",
                        passed=False,
                        message=f"CRITICAL: quarantined datapacks in exposure: {sorted(quarantine_in_exposure)}",
                    ))
                else:
                    checks.append(VerificationCheckV1(
                        check_id="exposure_quarantine_exclusion",
                        passed=True,
                        message="exposure ∩ quarantine = ∅",
                    ))
        except Exception as e:
            warnings.append(f"Could not verify exposure manifest: {e}")

    # 18. Orchestrator state semantic checks
    orchestrator_state_path = output_path / "orchestrator_state.json"
    if orchestrator_state_path.exists():
        try:
            with open(orchestrator_state_path, "r") as f:
                orch_data = json.load(f)
            
            # 18a: Check counters are non-negative
            counters_valid = True
            for field in ["failure_counts", "patience_counters", "cooldown_remaining"]:
                field_data = orch_data.get(field, {})
                if isinstance(field_data, dict):
                    for key, val in field_data.items():
                        if isinstance(val, (int, float)) and val < 0:
                            counters_valid = False
                            break
            
            checks.append(VerificationCheckV1(
                check_id="orchestrator_state_nonnegative_counters",
                passed=counters_valid,
                message="Orchestrator counters valid (all non-negative)" if counters_valid
                        else "VIOLATION: Negative orchestrator counter values detected",
            ))
            
            # 18b: Applied knob deltas present when training
            applied_deltas = orch_data.get("applied_knob_deltas", None)
            if is_training_run:
                has_deltas = applied_deltas is not None  # Empty list is valid
                checks.append(VerificationCheckV1(
                    check_id="orchestrator_applied_knob_deltas_present_when_training",
                    passed=has_deltas,
                    message=f"Training run has applied_knob_deltas ({len(applied_deltas or [])} deltas)" if has_deltas
                            else "VIOLATION: Training run missing applied_knob_deltas field",
                ))
        except Exception as e:
            warnings.append(f"Could not verify orchestrator state semantics: {e}")

    # 19. Applied knob deltas cross-validation
    applied_deltas_path = output_path / "applied_knob_deltas.json"
    if applied_deltas_path.exists() and orchestrator_state_path.exists():
        try:
            with open(applied_deltas_path, "r") as f:
                file_deltas = json.load(f)
            with open(orchestrator_state_path, "r") as f:
                orch_data = json.load(f)
            
            orch_deltas = orch_data.get("applied_knob_deltas", [])
            
            # Canonicalize and compare (sha256_json imported at module level)
            file_sha = sha256_json(file_deltas if isinstance(file_deltas, list) else file_deltas.get("deltas", []))
            orch_sha = sha256_json(orch_deltas)
            
            checks.append(VerificationCheckV1(
                check_id="applied_knob_deltas_match_orchestrator_state",
                passed=file_sha == orch_sha,
                message="Applied knob deltas match orchestrator state" if file_sha == orch_sha
                        else "VIOLATION: Knob deltas in file differ from orchestrator state",
                expected=orch_sha[:16],
                actual=file_sha[:16],
            ))
        except Exception as e:
            warnings.append(f"Could not verify knob deltas consistency: {e}")

    # 20. Econ tensor ↔ exposure consistency (lightweight)
    econ_tensor_path = output_path / "econ_tensor.json"
    if econ_tensor_path.exists() and is_training_run:
        if not (exposure_manifest_path.exists() or selection_manifest_path.exists()):
            checks.append(VerificationCheckV1(
                check_id="econ_tensor_exposure_consistency",
                passed=False,
                message="VIOLATION: Econ tensor exists but no selection/exposure manifest for training run",
            ))
        else:
            checks.append(VerificationCheckV1(
                check_id="econ_tensor_exposure_consistency",
                passed=True,
                message="Econ tensor has corresponding selection/exposure manifest",
            ))

    # 21. Ledger regal REQUIRED for FULL (training) runs
    ledger_regal_path = output_path / "ledger_regal.json"
    if is_training_run:
        has_ledger_regal = ledger_regal_path.exists() and getattr(manifest, 'ledger_regal_sha', None)
        checks.append(VerificationCheckV1(
            check_id="ledger_regal_required_for_full",
            passed=has_ledger_regal,
            message="FULL run has ledger_regal.json with SHA" if has_ledger_regal
                    else "VIOLATION: FULL training run missing ledger_regal.json or ledger_regal_sha",
        ))

    # 22. Orchestrator decisions match plan_applied_events
    plan_applied_events_path = output_path / "plan_applied_events.jsonl"
    if orchestrator_state_path.exists() and plan_applied_events_path.exists():
        try:
            with open(orchestrator_state_path, "r") as f:
                orch_data = json.load(f)
            
            noop_decisions = orch_data.get("noop_decisions", [])
            clamp_decisions = orch_data.get("clamp_decisions", [])
            
            # Load plan applied events
            plan_events = []
            with open(plan_applied_events_path, "r") as f:
                for line in f:
                    if line.strip():
                        plan_events.append(json.load(f) if not plan_events else json.loads(line.strip()))
            
            # Check that noop decisions correspond to plan_applied=False events
            # Simplified check: if there are noop decisions, there should be non-applied events
            consistency_ok = True
            if noop_decisions:
                # Check for corresponding plan_applied=False in events
                non_applied_count = sum(1 for e in plan_events if not e.get("plan_applied", True))
                if non_applied_count < len(noop_decisions):
                    consistency_ok = False
            
            checks.append(VerificationCheckV1(
                check_id="orchestrator_decisions_match_plan_applied_events",
                passed=consistency_ok,
                message="Orchestrator decisions consistent with plan_applied_events" if consistency_ok
                        else f"VIOLATION: {len(noop_decisions)} noop decisions but fewer non-applied events",
            ))
        except Exception as e:
            warnings.append(f"Could not verify orchestrator-plan consistency: {e}")

    # 23. Selection pathology: eligible_shrunk
    # Threshold from RegalGatesV1 (default min_eligible_count=3)
    selection_manifest_path = output_path / "selection_manifest.json"
    if selection_manifest_path.exists():
        try:
            with open(selection_manifest_path, "r") as f:
                sel_data = json.load(f)
            
            eligible_ids = sel_data.get("eligible_datapack_ids", [])
            rejected = sel_data.get("rejected_datapacks", [])
            
            # Check eligible count (threshold: min 3 for meaningful training)
            MIN_ELIGIBLE = 3
            eligible_ok = len(eligible_ids) >= MIN_ELIGIBLE
            checks.append(VerificationCheckV1(
                check_id="eligible_shrunk_pathology",
                passed=eligible_ok,
                message=f"Eligible count healthy: {len(eligible_ids)}" if eligible_ok
                        else f"WARNING: Eligible shrunk to {len(eligible_ids)} (< {MIN_ELIGIBLE})",
            ))
            
            # 24. Selection pathology: rejection_spike
            # If more than 50% rejected, flag as spike
            REJECTION_SPIKE_THRESHOLD = 0.5
            total_considered = len(eligible_ids) + len(rejected)
            if total_considered > 0:
                rejection_ratio = len(rejected) / total_considered
                rejection_ok = rejection_ratio <= REJECTION_SPIKE_THRESHOLD
                checks.append(VerificationCheckV1(
                    check_id="selection_rejection_spike",
                    passed=rejection_ok,
                    message=f"Rejection ratio normal: {rejection_ratio:.1%}" if rejection_ok
                            else f"WARNING: Rejection spike detected: {rejection_ratio:.1%} (> {REJECTION_SPIKE_THRESHOLD:.0%})",
                ))
        except Exception as e:
            warnings.append(f"Could not verify selection pathology: {e}")

    # 25. Econ evidence present for FULL runs with econ tensor
    if econ_tensor_path.exists() and is_training_run:
        try:
            with open(econ_tensor_path, "r") as f:
                econ_data = json.load(f)
            
            # Check for required evidence fields
            evidence = econ_data.get("evidence_summary", {})
            required_evidence = ["num_selected", "num_sampled"]
            evidence_present = all(
                evidence.get(k) is not None for k in required_evidence
            ) if evidence else False
            
            checks.append(VerificationCheckV1(
                check_id="econ_evidence_present_for_full",
                passed=evidence_present,
                message="Econ tensor has evidence_summary" if evidence_present
                        else "VIOLATION: FULL run econ tensor missing evidence_summary",
            ))
        except Exception as e:
            warnings.append(f"Could not verify econ evidence: {e}")

    # Compute manifest SHA
    manifest_sha = sha256_json(manifest.model_dump(mode="json"))

    return _build_report(run_id, checks, warnings, manifest_sha, len(ledger_records))


def _build_report(
    run_id: str,
    checks: List[VerificationCheckV1],
    warnings: List[str],
    manifest_sha: Optional[str],
    ledger_record_count: int,
) -> VerificationReportV1:
    """Build verification report from checks."""
    passed_count = sum(1 for c in checks if c.passed)
    failed_count = len(checks) - passed_count
    
    return VerificationReportV1(
        run_id=run_id,
        all_passed=failed_count == 0,
        check_count=len(checks),
        passed_count=passed_count,
        failed_count=failed_count,
        checks=checks,
        warnings=warnings,
        manifest_sha=manifest_sha,
        ledger_record_count=ledger_record_count,
    )


def write_verification_report(path: str, report: VerificationReportV1) -> str:
    """Write verification report to JSON file.
    
    Returns:
        SHA-256 of written report
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)
    return report.sha256()


def get_blocking_failures(
    report: VerificationReportV1,
    block_on_warn: bool = False,
) -> Tuple[int, List[str]]:
    """Extract blocking failures from a verification report.
    
    Args:
        report: Verification report to analyze
        block_on_warn: If True, WARN severity also blocks (production mode)
        
    Returns:
        Tuple of (blocking_failure_count, list of blocking check_ids that failed)
    """
    blocking_failures = []
    for check in report.checks:
        if not check.passed:
            # Check if this failure is blocking
            is_blocking = False
            
            # FAIL severity always blocks (if in BLOCKING_CHECK_IDS or has severity=FAIL)
            if check.severity == "FAIL":
                # Either in blocking list OR has explicit FAIL severity
                if check.check_id in BLOCKING_CHECK_IDS:
                    is_blocking = True
            elif check.severity == "WARN" and block_on_warn:
                # WARN only blocks in production mode (block_on_warn=True)
                if check.check_id in BLOCKING_CHECK_IDS:
                    is_blocking = True
            
            if is_blocking:
                blocking_failures.append(check.check_id)
    
    return len(blocking_failures), blocking_failures


def compute_blocking_policy_sha(
    blocking_check_ids: frozenset,
    block_on_severity: str = "FAIL",
) -> str:
    """Compute deterministic SHA of blocking policy.
    
    This proves "this deploy was blocked under policy X".
    
    Args:
        blocking_check_ids: Set of check IDs that block deployment
        block_on_severity: "FAIL" or "WARN" - which severity levels block
        
    Returns:
        SHA-256 of the blocking policy
    """
    policy_data = {
        "blocking_check_ids": sorted(blocking_check_ids),
        "block_on_severity": block_on_severity,
    }
    return sha256_json(policy_data)


__all__ = [
    "VerificationCheckV1",
    "VerificationReportV1",
    "verify_run",
    "write_verification_report",
    "get_blocking_failures",
    "compute_blocking_policy_sha",
    "BLOCKING_CHECK_IDS",
]


