"""SemanticVLA — SCAFFOLDING ONLY.

Status: explicitly non-terminal placeholder.  This module is an insufficient
stand-in for the semantic analysis posture required by the multi-WM stack.

Successor
---------
The successor to SemanticVLA is NOT one monolithic semantic-analysis model.
It is the **distributed semantic bridge family** defined in
``src/world_model/perception_grounding/semantic_bridges.py``, built from:

1. **Perception/Grounding WM canonical semantic substrate**:
   ``SceneGraphState`` — the graph-transformer-produced object-relational-
   temporal representation that all downstream consumers read from.

2. **Per-WM semantic bridge layers** that transform the canonical substrate
   into WM-native semantic forms:
   - ``SimSynthSemanticBridgeState``: physics-relevant object preservation,
     branch comparison, diffusion conditioning
   - ``EmbodimentSemanticBridgeState``: affordance, action relevance,
     body-object pairwise scores
   - ``AnnotationSemanticBridgeState``: object-linked primitive/event
     labeling, failure interpretation, training dataset formation
   - ``EconomicSemanticBridgeState``: fixed-dim summaries for allocation

3. **Provider-backed evidence** from SAM 3/3.1, DINOv2/SigLIP, V-JEPA 2,
   and depth models, fused through evidence routing into the substrate.

4. **Teacher/runtime semantic proposals** as advisory evidence, not truth.

Until the distributed bridge family is load-bearing in the live loop,
this scaffolding class remains importable for backward compatibility.
It should NOT be extended.  New semantic work should go into the bridge
family and canonical substrate.

See also:
- ``docs/economic_world_model/neuralization_bridge_doctrine.md``
- ``docs/economic_world_model/doctrine_semantic_bridge_successor.md``
- ``src/world_model/perception_grounding/semantic_bridges.py``
"""

from typing import Any, Dict

# Successor imports — available once Phase 2 perception_grounding is wired.
# These are imported here to make the successor relationship explicit in code,
# not just in documentation.
try:
    from src.world_model.perception_grounding.semantic_bridges import (  # noqa: F401
        AnnotationSemanticBridgeState,
        EmbodimentSemanticBridgeState,
        EconomicSemanticBridgeState,
        SemanticBridgeRegistry,
        SimSynthSemanticBridgeState,
    )

    _SUCCESSOR_AVAILABLE = True
except ImportError:
    _SUCCESSOR_AVAILABLE = False

# Scaffolding status — consumed by downstream code to detect posture.
SEMANTIC_VLA_STATUS = "scaffolding_only"
SEMANTIC_VLA_SUCCESSOR = "distributed_semantic_bridge_family"
SEMANTIC_VLA_SUCCESSOR_MODULE = "src.world_model.perception_grounding.semantic_bridges"


class SemanticVLA:
    """Semantic VLA analyzer — scaffolding only.

    This class is an insufficient placeholder.  It extracts tags if present
    and returns empty structure otherwise.  It does NOT perform real semantic
    analysis.

    The replacement is the distributed semantic bridge family in
    ``src.world_model.perception_grounding.semantic_bridges``.

    Callers should check ``SEMANTIC_VLA_STATUS`` before treating outputs
    as semantically meaningful.
    """

    @property
    def status(self) -> str:
        return SEMANTIC_VLA_STATUS

    @property
    def successor_available(self) -> bool:
        return _SUCCESSOR_AVAILABLE

    def analyze_episode(self, datapack_or_episode: Any) -> Dict[str, Any]:
        # Stub: extract tags if present
        tags: list[Any] = []
        if hasattr(datapack_or_episode, "semantic_tags"):
            tags = getattr(datapack_or_episode, "semantic_tags", [])
        elif isinstance(datapack_or_episode, dict):
            tags = datapack_or_episode.get("semantic_tags", [])
        return {
            "task_graph_nodes": [],
            "object_tags": [],
            "semantic_tags": tags,
            "success_conditions": [],
            "attribution_hints": {},
            # Posture metadata — consumers can read these to understand
            # that this output is scaffolding, not real analysis.
            "_semantic_vla_status": SEMANTIC_VLA_STATUS,
            "_semantic_vla_successor": SEMANTIC_VLA_SUCCESSOR,
            "_successor_available": _SUCCESSOR_AVAILABLE,
        }
