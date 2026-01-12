"""Realized value ledger - append-only record of training value.

Links plan exposure, training windows, and audit deltas in an
immutable, append-only ledger for provenance and attribution.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import (
    ValueLedgerRecordV1,
    LedgerWindowV1,
    LedgerExposureV1,
    LedgerPolicyV1,
    LedgerAuditV1,
    LedgerDeltasV1,
    LedgerRegalV1,
    LedgerGraphV1,
    LedgerProbeV1,
    LedgerPlanPolicyV1,
    LedgerEconV1,
    AuditAggregateV1,
)
from src.utils.config_digest import sha256_json


class ValueLedger:
    """Append-only value ledger.

    Writes ValueLedgerRecordV1 entries to a JSONL file.
    Each record links plan exposure to realized audit deltas.
    """

    def __init__(self, ledger_path: str):
        """Initialize ledger.

        Args:
            ledger_path: Path to ledger JSONL file
        """
        self.ledger_path = Path(ledger_path)
        self._records: List[ValueLedgerRecordV1] = []

        # Load existing records if file exists
        if self.ledger_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        """Load existing records from file."""
        with open(self.ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._records.append(ValueLedgerRecordV1.model_validate(data))

    def append(self, record: ValueLedgerRecordV1) -> None:
        """Append a record to the ledger (append-only).

        Args:
            record: Ledger record to append
        """
        self._records.append(record)

        # Append to file
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def create_record(
        self,
        run_id: str,
        plan_id: str,
        plan_sha: str,
        audit_before: AuditAggregateV1,
        audit_after: AuditAggregateV1,
        window: LedgerWindowV1,
        exposure: LedgerExposureV1,
        policy: LedgerPolicyV1,
        notes: Optional[str] = None,
        probe: Optional[LedgerProbeV1] = None,
        plan_policy: Optional[LedgerPlanPolicyV1] = None,
        graph: Optional[LedgerGraphV1] = None,
        regal: Optional[LedgerRegalV1] = None,
        econ: Optional["LedgerEconV1"] = None,
        # P0: regal provenance status
        plan_applied: bool = True,
    ) -> ValueLedgerRecordV1:
        """Create a ledger record from audit results.

        Args:
            run_id: Run identifier
            plan_id: Plan identifier
            plan_sha: Plan SHA-256
            audit_before: Audit results before training
            audit_after: Audit results after training
            window: Training window specification
            exposure: Datapack exposure
            policy: Policy checkpoints
            notes: Optional notes
            probe: Optional probe harness results
            plan_policy: Optional plan policy details
            graph: Optional graph metrics
            regal: Optional regal evaluation results
            econ: Optional econ tensor provenance
            plan_applied: Whether plan was applied (False if halted)

        Returns:
            ValueLedgerRecordV1 record
        """
        # Compute deltas
        delta_success = None
        delta_error = None
        delta_energy_Wh = None
        delta_mpl_proxy = None

        if audit_before.success_rate is not None and audit_after.success_rate is not None:
            delta_success = audit_after.success_rate - audit_before.success_rate

        if audit_before.mean_error is not None and audit_after.mean_error is not None:
            delta_error = audit_after.mean_error - audit_before.mean_error

        if audit_before.mean_energy_Wh is not None and audit_after.mean_energy_Wh is not None:
            delta_energy_Wh = audit_after.mean_energy_Wh - audit_before.mean_energy_Wh

        if audit_before.mean_mpl_proxy is not None and audit_after.mean_mpl_proxy is not None:
            delta_mpl_proxy = audit_after.mean_mpl_proxy - audit_before.mean_mpl_proxy

        audit_ref = LedgerAuditV1(
            audit_suite_id=audit_before.audit_suite_id,
            audit_seed=audit_before.seed,
            audit_config_sha=audit_before.config_sha,
            audit_results_before_sha=audit_before.episodes_sha,
            audit_results_after_sha=audit_after.episodes_sha,
        )

        deltas = LedgerDeltasV1(
            delta_success=delta_success,
            delta_error=delta_error,
            delta_energy_Wh=delta_energy_Wh,
            delta_mpl_proxy=delta_mpl_proxy,
        )

        # P0: Compute regal provenance status
        # SAFETY INVARIANT: Missing regal → deploy blocked (never permissive)
        regal_degraded = regal is None
        allow_deploy = regal.all_passed if regal else False  # Must be False when regal missing

        return ValueLedgerRecordV1(
            record_id=str(uuid.uuid4())[:8],
            run_id=run_id,
            plan_id=plan_id,
            plan_sha=plan_sha,
            exposure=exposure,
            window=window,
            policy=policy,
            audit=audit_ref,
            deltas=deltas,
            probe=probe,
            plan_policy=plan_policy,
            graph=graph,
            regal=regal,
            econ=econ,
            notes=notes,
            # P0: regal provenance fields
            regal_degraded=regal_degraded,
            allow_deploy=allow_deploy,
            plan_applied=plan_applied,
        )

    @property
    def records(self) -> List[ValueLedgerRecordV1]:
        """All ledger records."""
        return self._records

    def last_record(self) -> Optional[ValueLedgerRecordV1]:
        """Get the most recent record."""
        return self._records[-1] if self._records else None


__all__ = [
    "ValueLedger",
]
