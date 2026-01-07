"""Configuration for Embodiment module."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmbodimentConfig:
    """Tunable thresholds and weights for embodiment extraction."""

    # Contact detection
    contact_base_radius: float = 0.05
    contact_scale_factor: float = 0.5
    visibility_threshold: float = 0.2
    occlusion_threshold: float = 0.8
    impossible_occlusion_threshold: float = 0.9
    low_visibility_threshold: float = 0.05
    min_contact_frequency: float = 0.05

    # Segmentation
    segment_min_length: int = 2

    # Embodiment weight blending
    w_semantic_weight: float = 0.4
    w_contact_weight: float = 0.3
    w_mhn_weight: float = 0.3
    drift_penalty_weight: float = 0.5
    impossible_penalty_weight: float = 0.5

    # Trust override heuristics
    trust_override_drift_threshold: float = 0.6
    trust_override_impossible_contacts: int = 3

    # Calibration heuristics
    expected_contact_rate_tolerance: float = 0.1
    friction_delta_step: float = 0.05
    damping_delta_step: float = 0.02
    mass_delta_step: float = 0.05
    restitution_delta_step: float = 0.1
