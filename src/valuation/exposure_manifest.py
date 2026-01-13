"""Exposure manifest for training windows.

Tracks what datapacks/slices were used during a training window.
Mandatory for ledger writes.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from src.utils.config_digest import sha256_json

from src.contracts.schemas import SelectionManifestV1


class ExposureManifestV1(BaseModel):
    """Exposure manifest for a training window.

    Records exactly what data was used during training for attribution.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"
    manifest_id: str

    # Window metadata
    step_start: int
    step_end: int
    ts_start: str
    ts_end: str

    # Task exposure
    task_family_counts: Dict[str, int] = Field(default_factory=dict)
    total_samples: int = 0

    # Datapack exposure
    datapack_ids: List[str] = Field(default_factory=list)
    slice_ids: Optional[List[str]] = None

    # Optional: repr slice label distribution
    slice_label_distribution: Optional[Dict[str, int]] = None

    # Plan reference
    plan_id: Optional[str] = None
    plan_sha: Optional[str] = None

    def sha256(self) -> str:
        """Compute SHA-256 of manifest."""
        return sha256_json(self.model_dump(mode="json"))


class ExposureTracker:
    """Tracks exposure during training for manifest generation."""

    def __init__(self, manifest_id: str, step_start: int = 0):
        self.manifest_id = manifest_id
        self.step_start = step_start
        self.step_end = step_start
        self.ts_start = datetime.now().isoformat()
        self.ts_end = self.ts_start

        self._task_counts: Counter = Counter()
        self._datapack_ids: set = set()
        self._slice_ids: set = set()
        self._slice_labels: Counter = Counter()

        self._plan_id: Optional[str] = None
        self._plan_sha: Optional[str] = None
        
        # Quarantine enforcement
        self._quarantine_datapack_ids: set = set()
        self._excluded_count: int = 0  # Count of samples excluded due to quarantine

        # Selection tracking (Phase 2)
        self._eligible_datapack_ids: set = set()  # All datapacks that were eligible
        self._rejected_datapacks: List[Dict[str, str]] = []  # Rejected with reasons
        self._rng_seed: int = 42
        self._sampler_config_sha: Optional[str] = None

    def record_sample(
        self,
        task_family: str,
        datapack_id: Optional[str] = None,
        slice_id: Optional[str] = None,
        slice_label: Optional[str] = None,
    ) -> bool:
        """Record a single training sample.
        
        Args:
            task_family: Task family name
            datapack_id: Optional datapack identifier
            slice_id: Optional slice identifier
            slice_label: Optional slice label for repr distribution
            
        Returns:
            True if sample was recorded, False if excluded due to quarantine
        """
        # Check quarantine BEFORE recording
        if datapack_id and datapack_id in self._quarantine_datapack_ids:
            self._excluded_count += 1
            self._rejected_datapacks.append({"id": datapack_id, "reason": "quarantine"})
            return False
        
        self._task_counts[task_family] += 1
        if datapack_id:
            self._datapack_ids.add(datapack_id)
        if slice_id:
            self._slice_ids.add(slice_id)
        if slice_label:
            self._slice_labels[slice_label] += 1
        return True

    def update_step(self, step: int) -> None:
        """Update current step."""
        self.step_end = step
        self.ts_end = datetime.now().isoformat()

    def set_plan(self, plan_id: str, plan_sha: str) -> None:
        """Set plan reference."""
        self._plan_id = plan_id
        self._plan_sha = plan_sha
    
    def set_quarantine(self, datapack_ids: List[str]) -> None:
        """Set list of quarantined datapack IDs to exclude.
        
        Args:
            datapack_ids: List of datapack IDs to quarantine (exclude from training)
        """
        self._quarantine_datapack_ids = set(datapack_ids)
    
    @property
    def excluded_count(self) -> int:
        """Number of samples excluded due to quarantine."""
        return self._excluded_count
    
    @property
    def quarantine_datapack_ids(self) -> List[str]:
        """List of quarantined datapack IDs."""
        return sorted(self._quarantine_datapack_ids)

    def set_eligible_datapacks(self, datapack_ids: List[str]) -> None:
        """Set the list of eligible datapacks (before filtering).

        Args:
            datapack_ids: List of all eligible datapack IDs at sampling start
        """
        self._eligible_datapack_ids = set(datapack_ids)

    def set_sampler_config(self, seed: int, config_sha: Optional[str] = None) -> None:
        """Set sampler configuration for reproducibility.

        Args:
            seed: RNG seed used for sampling
            config_sha: Optional SHA of sampler config
        """
        self._rng_seed = seed
        self._sampler_config_sha = config_sha

    def record_rejection(self, datapack_id: str, reason: str) -> None:
        """Record a datapack rejection with reason.

        Args:
            datapack_id: ID of rejected datapack
            reason: Reason for rejection (e.g., "missing_fields", "validation_failed")
        """
        self._rejected_datapacks.append({"id": datapack_id, "reason": reason})

    def build_selection_manifest(self) -> SelectionManifestV1:
        """Build selection manifest for deterministic replay.

        Returns:
            SelectionManifestV1 with complete selection provenance
        """
        return SelectionManifestV1(
            manifest_id=f"{self.manifest_id}_selection",
            eligible_datapack_ids=sorted(self._eligible_datapack_ids),
            quarantine_datapack_ids=self.quarantine_datapack_ids,
            selected_datapack_ids=sorted(self._datapack_ids),
            rejected_datapacks=self._rejected_datapacks,
            rng_seed=self._rng_seed,
            sampler_config_sha=self._sampler_config_sha,
        )

    def build_manifest(self) -> ExposureManifestV1:
        """Build the exposure manifest."""
        return ExposureManifestV1(
            manifest_id=self.manifest_id,
            step_start=self.step_start,
            step_end=self.step_end,
            ts_start=self.ts_start,
            ts_end=self.ts_end,
            task_family_counts=dict(self._task_counts),
            total_samples=sum(self._task_counts.values()),
            datapack_ids=sorted(self._datapack_ids),
            slice_ids=sorted(self._slice_ids) if self._slice_ids else None,
            slice_label_distribution=dict(self._slice_labels) if self._slice_labels else None,
            plan_id=self._plan_id,
            plan_sha=self._plan_sha,
        )


def write_exposure_manifest(path: str, manifest: ExposureManifestV1) -> str:
    """Write exposure manifest to JSON file.

    Args:
        path: Output path
        manifest: Manifest to write

    Returns:
        SHA-256 of written manifest
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2)
    return manifest.sha256()


def load_exposure_manifest(path: str) -> ExposureManifestV1:
    """Load exposure manifest from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return ExposureManifestV1.model_validate(data)


def write_selection_manifest(path: str, manifest: SelectionManifestV1) -> str:
    """Write selection manifest to JSON file.

    Args:
        path: Output path
        manifest: Selection manifest to write

    Returns:
        SHA-256 of written file content
    """
    from src.utils.config_digest import sha256_file
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2)
    return sha256_file(str(output_path))


__all__ = [
    "ExposureManifestV1",
    "ExposureTracker",
    "write_exposure_manifest",
    "load_exposure_manifest",
    "write_selection_manifest",
]
