"""Epiplexity / prequential-MDL utilities."""

from .estimators import EpiplexityEstimator, PrequentialAUCLossEstimator, RequentialEstimator, ProbeModelConfig
from .tracker import EpiplexityRunKey, EpiplexityResult, EpiplexityTracker, ComputeBudget
from .harness import TokenizerAblationHarness, EpiplexityLeaderboard
from .metadata import (
    apply_epiplexity_overlay,
    attach_epiplexity_result,
    attach_epiplexity_summary,
    build_epiplexity_overlay_record,
    extract_epiplexity_summary_metric,
    extract_epiplexity_summary_confidence,
    load_epiplexity_overlay_map,
    select_default_epiplexity_summary,
    set_epiplexity_default_selector,
    write_epiplexity_overlays,
)
from .representations import build_default_representation_fns
from .transforms import transform_chain_hash

__all__ = [
    "EpiplexityEstimator",
    "PrequentialAUCLossEstimator",
    "RequentialEstimator",
    "ProbeModelConfig",
    "EpiplexityRunKey",
    "EpiplexityResult",
    "EpiplexityTracker",
    "ComputeBudget",
    "TokenizerAblationHarness",
    "EpiplexityLeaderboard",
    "apply_epiplexity_overlay",
    "attach_epiplexity_result",
    "attach_epiplexity_summary",
    "build_epiplexity_overlay_record",
    "extract_epiplexity_summary_metric",
    "extract_epiplexity_summary_confidence",
    "load_epiplexity_overlay_map",
    "select_default_epiplexity_summary",
    "set_epiplexity_default_selector",
    "write_epiplexity_overlays",
    "build_default_representation_fns",
    "transform_chain_hash",
]
