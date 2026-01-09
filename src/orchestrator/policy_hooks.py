"""Scaffolding hooks for economic and reward integrity policies.

These interfaces allow future integration of complex economic logic and anti-hacking guards
without disrupting the core homeostatic loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING

from src.contracts.schemas import PlanGainScheduleV1

if TYPE_CHECKING:
    from src.representation.homeostasis import SignalBundle
    from src.valuation.exposure_manifest import ExposureManifestV1


class EconPlanPolicyProvider(ABC):
    """Interface for economic policy providers."""
    
    @abstractmethod
    def get_gain_schedule(
        self,
        signal_bundle: "SignalBundle",
        exposure_manifest: Optional["ExposureManifestV1"] = None,
    ) -> Optional[PlanGainScheduleV1]:
        """Get gain schedule override for the current context.
        
        Returns:
            Review-approved gain schedule, or None to use default config.
        """
        pass


class RewardIntegrityGuard(ABC):
    """Interface for reward integrity guards."""
    
    @abstractmethod
    def adjust_gain_schedule(
        self,
        schedule: PlanGainScheduleV1,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> PlanGainScheduleV1:
        """Adjust gain schedule to prevent reward hacking or instability.
        
        Args:
            schedule: Proposed gain schedule
            telemetry: Runtime telemetry
            
        Returns:
            Safe gain schedule
        """
        pass


class DefaultEconPolicyProvider(EconPlanPolicyProvider):
    """Default no-op provider."""
    
    def get_gain_schedule(
        self,
        signal_bundle: "SignalBundle",
        exposure_manifest: Optional["ExposureManifestV1"] = None,
    ) -> Optional[PlanGainScheduleV1]:
        return None


class DefaultRewardIntegrityGuard(RewardIntegrityGuard):
    """Default identity guard."""
    
    def adjust_gain_schedule(
        self,
        schedule: PlanGainScheduleV1,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> PlanGainScheduleV1:
        return schedule
