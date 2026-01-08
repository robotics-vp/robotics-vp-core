"""Epiplexity / prequential-MDL utilities."""

from .estimators import EpiplexityEstimator, PrequentialAUCLossEstimator, RequentialEstimator, ProbeModelConfig
from .tracker import EpiplexityRunKey, EpiplexityResult, EpiplexityTracker, ComputeBudget
from .harness import TokenizerAblationHarness, EpiplexityLeaderboard
from .metadata import (
    attach_epiplexity_result,
    attach_epiplexity_summary,
    extract_epiplexity_summary_metric,
    extract_epiplexity_summary_confidence,
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
    "attach_epiplexity_result",
    "attach_epiplexity_summary",
    "extract_epiplexity_summary_metric",
    "extract_epiplexity_summary_confidence",
    "build_default_representation_fns",
    "transform_chain_hash",
]
