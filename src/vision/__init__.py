"""
Vision Affordance and Fragility Heads.

Provides:
- RiskMapHead: Per-pixel fragility/collision risk
- AffordanceHead: Handle graspability/interaction affordance
- NoGoZoneHead: Binary unsafe region mask
- FragilityPriorHead: Draws heatmap around fragile objects
- VisionEncoderWithHeads: Combined encoder with all heads
"""

from .risk_map_head import RiskMapHead
from .affordance_head import AffordanceHead
from .no_go_head import NoGoZoneHead
from .fragility_prior_head import FragilityPriorHead
from .encoder_with_heads import VisionEncoderWithHeads

__all__ = [
    'RiskMapHead',
    'AffordanceHead',
    'NoGoZoneHead',
    'FragilityPriorHead',
    'VisionEncoderWithHeads',
]
