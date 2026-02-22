"""Fast/slow reconciliation primitives for ontology + economic ledger sync.

This module combines three ideas for millisecond control loops:
1. Geometric shadows: project fast policy actions onto cached hard bounds.
2. Hierarchical ledgering: write econ tensors locally, settle asynchronously.
3. Predictive ontology masking: prefetch likely constraint sets by context zone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from typing import Deque, Dict, List, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class ConstraintBound:
    """Per-dimension hard bound used by the edge projector."""

    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if self.min_value > self.max_value:
            raise ValueError("min_value must be <= max_value")

    def clamp(self, value: float) -> float:
        safe_value = float(value)
        if not isfinite(safe_value):
            safe_value = self.min_value
        return max(self.min_value, min(self.max_value, safe_value))


@dataclass(frozen=True)
class ConstraintShadow:
    """Locally cached action manifold approximation."""

    version: int
    bounds: Mapping[str, ConstraintBound]
    source_window_ms: int = 100

    def project(self, action: Mapping[str, float]) -> Dict[str, float]:
        projected: Dict[str, float] = {}
        for name, value in action.items():
            bound = self.bounds.get(name)
            projected[name] = bound.clamp(value) if bound else float(value)
        return projected


@dataclass(frozen=True)
class EconTensorSample:
    """High-rate economic telemetry emitted by the fast loop."""

    tick_id: int
    energy_delta: float
    error_delta: float
    time_delta_ms: float


@dataclass(frozen=True)
class SettlementRecord:
    """Aggregated record prepared for L2/global settlement."""

    start_tick: int
    end_tick: int
    sample_count: int
    totals: Dict[str, float]


class TransientLedger:
    """L1 circular ledger for high-frequency econ tensors.

    Settlement is two-phase:
    - `prepare_settlement`: build a batch without mutating ack state.
    - `ack_settlement`: mark the batch as durably persisted in L2.
    """

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = max(1, int(capacity))
        self._buffer: Deque[EconTensorSample] = deque(maxlen=self.capacity)
        self._last_acked_tick: Optional[int] = None

    def append(self, sample: EconTensorSample) -> None:
        self._buffer.append(sample)

    def pending(self) -> List[EconTensorSample]:
        if self._last_acked_tick is None:
            return list(self._buffer)
        return [s for s in self._buffer if s.tick_id > self._last_acked_tick]

    def prepare_settlement(self, max_batch: int = 256) -> Optional[SettlementRecord]:
        pending = self.pending()[: max(1, int(max_batch))]
        if not pending:
            return None
        totals = {
            "energy_delta": sum(s.energy_delta for s in pending),
            "error_delta": sum(s.error_delta for s in pending),
            "time_delta_ms": sum(s.time_delta_ms for s in pending),
        }
        return SettlementRecord(
            start_tick=pending[0].tick_id,
            end_tick=pending[-1].tick_id,
            sample_count=len(pending),
            totals=totals,
        )

    def ack_settlement(self, end_tick: int) -> None:
        if self._last_acked_tick is None or end_tick > self._last_acked_tick:
            self._last_acked_tick = int(end_tick)

    def settle(self, max_batch: int = 256) -> Optional[SettlementRecord]:
        """Compatibility helper for immediate local settlement + ack."""
        record = self.prepare_settlement(max_batch=max_batch)
        if record:
            self.ack_settlement(record.end_tick)
        return record

    def is_reconciled(self, l2_latest_tick: int, max_tick_drift: int = 512) -> bool:
        max_seen_tick = max((s.tick_id for s in self._buffer), default=l2_latest_tick)
        return (max_seen_tick - int(l2_latest_tick)) <= int(max_tick_drift)


@dataclass
class OntologyMask:
    """Context-indexed cache of likely constraint shadows."""

    by_zone: MutableMapping[str, ConstraintShadow] = field(default_factory=dict)

    def load(self, zone_id: str, shadow: ConstraintShadow) -> None:
        self.by_zone[str(zone_id)] = shadow

    def resolve(self, zone_id: str) -> Optional[ConstraintShadow]:
        return self.by_zone.get(str(zone_id))


@dataclass(frozen=True)
class FastSlowSyncDecision:
    """Deploy gate decision exposed to orchestrator/runtime."""

    allow_deploy: bool
    reason: str
    active_shadow_version: Optional[int]


class FastSlowEconBridge:
    """Hybrid strategy blending geometric shadow + L1/L2 ledger + ontology masks."""

    def __init__(
        self,
        ledger: Optional[TransientLedger] = None,
        mask: Optional[OntologyMask] = None,
    ) -> None:
        self.ledger = ledger or TransientLedger()
        self.mask = mask or OntologyMask()
        self._fallback_shadow: Optional[ConstraintShadow] = None

    def update_shadow(
        self, shadow: ConstraintShadow, zone_id: Optional[str] = None
    ) -> None:
        self._fallback_shadow = shadow
        if zone_id:
            self.mask.load(zone_id=zone_id, shadow=shadow)

    def project_action(
        self, action: Mapping[str, float], zone_id: Optional[str] = None
    ) -> Dict[str, float]:
        shadow = self.mask.resolve(zone_id) if zone_id else None
        shadow = shadow or self._fallback_shadow
        if shadow is None:
            return {k: float(v) for k, v in action.items()}
        return shadow.project(action)

    def ingest_tick(
        self,
        tick_id: int,
        energy_delta: float,
        error_delta: float,
        time_delta_ms: float,
    ) -> None:
        self.ledger.append(
            EconTensorSample(
                tick_id=int(tick_id),
                energy_delta=float(energy_delta),
                error_delta=float(error_delta),
                time_delta_ms=float(time_delta_ms),
            )
        )

    def settle_to_l2(
        self, max_batch: int = 256, acknowledge: bool = False
    ) -> Optional[SettlementRecord]:
        record = self.ledger.prepare_settlement(max_batch=max_batch)
        if acknowledge and record is not None:
            self.ledger.ack_settlement(record.end_tick)
        return record

    def deploy_gate(
        self,
        *,
        l2_latest_tick: int,
        max_tick_drift: int = 512,
        zone_id: Optional[str] = None,
    ) -> FastSlowSyncDecision:
        if not self.ledger.is_reconciled(
            l2_latest_tick=l2_latest_tick, max_tick_drift=max_tick_drift
        ):
            return FastSlowSyncDecision(
                allow_deploy=False,
                reason="l1_l2_drift_exceeded",
                active_shadow_version=None,
            )

        shadow = self.mask.resolve(zone_id) if zone_id else None
        shadow = shadow or self._fallback_shadow
        if shadow is None:
            return FastSlowSyncDecision(
                allow_deploy=False,
                reason="no_constraint_shadow_available",
                active_shadow_version=None,
            )

        return FastSlowSyncDecision(
            allow_deploy=True,
            reason="reconciled_and_shadowed",
            active_shadow_version=shadow.version,
        )
