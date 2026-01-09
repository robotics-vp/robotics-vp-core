"""Schema versioning and capabilities.

Central source of truth for schema versions and feature detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set


class SchemaVersion(str, Enum):
    """Known schema versions."""

    V2_0_ENERGY = "2.0-energy"
    V2_1_PORTABLE = "2.1-portable"
    V2_2_REPR = "2.2-repr"
    V2_3_HOMEOSTATIC = "2.3-homeostatic"  # Current


# Current schema version
CURRENT_SCHEMA_VERSION = SchemaVersion.V2_3_HOMEOSTATIC


# Version ordering for compatibility checks
VERSION_ORDER = [
    SchemaVersion.V2_0_ENERGY,
    SchemaVersion.V2_1_PORTABLE,
    SchemaVersion.V2_2_REPR,
    SchemaVersion.V2_3_HOMEOSTATIC,
]


@dataclass(frozen=True)
class SchemaCapabilities:
    """Capabilities available at a given schema version."""

    # Basic fields
    has_condition_profile: bool = True
    has_attribution_profile: bool = True
    has_guidance_profile: bool = True

    # Portable artifacts (2.1+)
    has_scene_tracks: bool = False
    has_rgb_features: bool = False
    has_slice_labels: bool = False

    # Repr tokens (2.2+)
    has_repr_tokens: bool = False

    # Homeostatic (2.3+)
    has_signal_bundle: bool = False
    has_exposure_manifest: bool = False


# Capabilities by version
CAPABILITIES: Dict[SchemaVersion, SchemaCapabilities] = {
    SchemaVersion.V2_0_ENERGY: SchemaCapabilities(
        has_condition_profile=True,
        has_attribution_profile=True,
        has_guidance_profile=True,
    ),
    SchemaVersion.V2_1_PORTABLE: SchemaCapabilities(
        has_condition_profile=True,
        has_attribution_profile=True,
        has_guidance_profile=True,
        has_scene_tracks=True,
        has_rgb_features=True,
        has_slice_labels=True,
    ),
    SchemaVersion.V2_2_REPR: SchemaCapabilities(
        has_condition_profile=True,
        has_attribution_profile=True,
        has_guidance_profile=True,
        has_scene_tracks=True,
        has_rgb_features=True,
        has_slice_labels=True,
        has_repr_tokens=True,
    ),
    SchemaVersion.V2_3_HOMEOSTATIC: SchemaCapabilities(
        has_condition_profile=True,
        has_attribution_profile=True,
        has_guidance_profile=True,
        has_scene_tracks=True,
        has_rgb_features=True,
        has_slice_labels=True,
        has_repr_tokens=True,
        has_signal_bundle=True,
        has_exposure_manifest=True,
    ),
}


def get_capabilities(version: str) -> SchemaCapabilities:
    """Get capabilities for a schema version.

    Args:
        version: Schema version string

    Returns:
        SchemaCapabilities for that version
    """
    try:
        v = SchemaVersion(version)
        return CAPABILITIES.get(v, SchemaCapabilities())
    except ValueError:
        # Unknown version - return minimal capabilities
        return SchemaCapabilities()


def is_version_at_least(version: str, minimum: SchemaVersion) -> bool:
    """Check if version is at least a minimum version.

    Args:
        version: Schema version to check
        minimum: Minimum required version

    Returns:
        True if version >= minimum
    """
    try:
        v = SchemaVersion(version)
        v_idx = VERSION_ORDER.index(v)
        min_idx = VERSION_ORDER.index(minimum)
        return v_idx >= min_idx
    except (ValueError, IndexError):
        return False


def can_use_portable_artifacts(version: str) -> bool:
    """Check if version supports portable artifacts."""
    caps = get_capabilities(version)
    return caps.has_scene_tracks and caps.has_rgb_features


def can_use_repr_tokens(version: str) -> bool:
    """Check if version supports repr_tokens."""
    return get_capabilities(version).has_repr_tokens


def can_use_homeostatic(version: str) -> bool:
    """Check if version supports homeostatic features."""
    caps = get_capabilities(version)
    return caps.has_signal_bundle and caps.has_exposure_manifest


# Contract schema versions (separate from datapack versions)
CONTRACT_VERSIONS = {
    "SemanticUpdatePlanV1": "v1",
    "EpisodeInfoSummaryV1": "v1",
    "ValueLedgerRecordV1": "v1",
    "RunManifestV1": "v1",
    "ExposureManifestV1": "v1",
    "AuditSuiteDefinition": "v1",
}


def get_contract_version(contract_name: str) -> str:
    """Get version of a contract schema."""
    return CONTRACT_VERSIONS.get(contract_name, "unknown")


__all__ = [
    "SchemaVersion",
    "CURRENT_SCHEMA_VERSION",
    "VERSION_ORDER",
    "SchemaCapabilities",
    "CAPABILITIES",
    "get_capabilities",
    "is_version_at_least",
    "can_use_portable_artifacts",
    "can_use_repr_tokens",
    "can_use_homeostatic",
    "CONTRACT_VERSIONS",
    "get_contract_version",
]
