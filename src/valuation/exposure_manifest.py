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


__all__ = [
    "ExposureManifestV1",
    "ExposureTracker",
    "write_exposure_manifest",
    "load_exposure_manifest",
]
