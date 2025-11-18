#!/usr/bin/env python3
"""
Smoke test for Stage 2.4 SemanticTagPropagator.

Validates determinism, schema compliance, JSON safety, forbidden-field
constraints, and graceful handling of missing inputs. This is advisory-only and
does not mutate economics, rewards, or task graph state.
"""

import copy
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sima2.semantic_tag_propagator import SemanticTagPropagator
from src.sima2.tags import SemanticEnrichmentProposal


def _make_datapacks() -> List[Dict[str, Any]]:
    return [
        {
            "video_id": "drawer_001",
            "episode_id": "ep_001",
            "task": "open_drawer",
            "frames": [0, 1, 2, 3],
            "actions": [],
            "metadata": {
                "success": True,
                "objects_present": ["drawer", "handle"],
                "duration_sec": 10.0,
                "timestamp": 1700000000.0,
            },
        },
        {
            "video_id": "drawer_002",
            "episode_id": "ep_002",
            "task": "open_drawer",
            "frames": [0, 1, 2, 3, 4],
            "actions": [],
            "metadata": {
                "success": True,
                "objects_present": ["drawer", "handle", "vase_inside"],
                "duration_sec": 7.5,
                "timestamp": 1700000001.0,
            },
        },
    ]


def _make_ontology_props() -> List[Dict[str, Any]]:
    return [
        {
            "proposal_id": "onto_prop_45",
            "validation_status": "passed",
            "new_affordances": [
                {"name": "graspable", "object": "handle", "confidence": 0.92},
                {"name": "pullable", "object": "drawer", "confidence": 0.88},
            ],
            "fragility_updates": [{"object": "vase_inside", "level": "high", "damage_cost": 50.0}],
        }
    ]


def _make_task_graph_props() -> List[Dict[str, Any]]:
    return [
        {
            "proposal_id": "task_prop_78",
            "validation_status": "passed",
            "risk_annotations": [
                {"task": "open_drawer", "risk_type": "collision", "severity": "medium"}
            ],
            "efficiency_hints": [
                {"task": "open_drawer", "metric": "time", "suggestion": "slow down"}
            ],
        }
    ]


def _make_econ_outputs() -> Dict[str, Dict[str, Any]]:
    return {
        "ep_001": {"novelty_score": 0.2, "expected_mpl_gain": 0.5, "tier": 0},
        "ep_002": {"novelty_score": 0.73, "expected_mpl_gain": 4.2, "tier": 2},
    }


def _assert_no_forbidden_fields(enrichment: Dict[str, Any]) -> None:
    forbidden = {
        "rewards",
        "mpl_value",
        "wage_parity",
        "sampling_weight",
        "task_order",
        "affordance_definitions",
    }

    def _check(obj: Any) -> None:
        if isinstance(obj, dict):
            overlap = forbidden.intersection(obj.keys())
            assert not overlap, f"Forbidden fields found: {overlap}"
            for value in obj.values():
                _check(value)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _check(v)

    _check(enrichment)


def _assert_json_safe(proposal: SemanticEnrichmentProposal) -> None:
    enrichment = proposal.to_jsonl_enrichment()
    json_str = json.dumps(enrichment)
    assert json.loads(json_str) == enrichment, "JSON roundtrip failed"


def _assert_cross_consistency(proposal: SemanticEnrichmentProposal) -> None:
    """Ensure tags align with fixture proposals."""
    aff_names = {(t.object_name, t.affordance_name) for t in proposal.affordance_tags}
    assert ("handle", "graspable") in aff_names
    if any(t.object_name == "vase_inside" for t in proposal.fragility_tags):
        frag = [t for t in proposal.fragility_tags if t.object_name == "vase_inside"][0]
        assert frag.fragility_level == "high"

    for tag in proposal.risk_tags:
        assert tag.risk_type == "collision"
        assert tag.severity in {"medium", "high", "low"}

    for tag in proposal.novelty_tags:
        assert 0.0 <= tag.novelty_score <= 1.0


def main() -> None:
    print("[smoke_test_stage2_4_semantic_tag_propagation] Starting tests...")
    datapacks = _make_datapacks()
    ontology = _make_ontology_props()
    task_graph = _make_task_graph_props()
    economics = _make_econ_outputs()

    propagator = SemanticTagPropagator()

    # Determinism
    proposals_1 = propagator.generate_proposals(datapacks, ontology, task_graph, economics)
    proposals_2 = propagator.generate_proposals(datapacks, ontology, task_graph, economics)
    assert len(proposals_1) == len(proposals_2)
    assert json.dumps([p.to_jsonl_enrichment() for p in proposals_1], sort_keys=True) == json.dumps(
        [p.to_jsonl_enrichment() for p in proposals_2], sort_keys=True
    ), "Proposal outputs must be deterministic"
    print("[TEST 1 PASS] Deterministic proposal generation")

    # Required fields and schema checks
    for proposal in proposals_1:
        enrichment = proposal.to_jsonl_enrichment()["enrichment"]
        for required in ("coherence_score", "confidence", "validation_status"):
            assert required in enrichment, f"Missing required field {required}"
        assert isinstance(enrichment["coherence_score"], float)
        assert isinstance(enrichment["confidence"], float)
        assert enrichment["validation_status"] in {"pending", "passed", "failed"}
    print("[TEST 2 PASS] Required fields present and typed")

    # Value ranges
    for proposal in proposals_1:
        assert 0.0 <= proposal.coherence_score <= 1.0
        assert 0.0 <= proposal.confidence <= 1.0
        for eff in proposal.efficiency_tags:
            assert 0.0 <= eff.score <= 1.0
        for nov in proposal.novelty_tags:
            assert 0.0 <= nov.novelty_score <= 1.0
    print("[TEST 3 PASS] Numerical ranges valid")

    # JSON safety and forbidden fields
    for proposal in proposals_1:
        _assert_json_safe(proposal)
        _assert_no_forbidden_fields(proposal.to_jsonl_enrichment())
    print("[TEST 4 PASS] JSON safety and forbidden field checks")

    # Cross-consistency with fixture proposals
    for proposal in proposals_1:
        _assert_cross_consistency(proposal)
    print("[TEST 5 PASS] Cross-consistency validated")

    # No mutation of inputs
    dp_copy = copy.deepcopy(datapacks)
    ont_copy = copy.deepcopy(ontology)
    tg_copy = copy.deepcopy(task_graph)
    econ_copy = copy.deepcopy(economics)
    _ = propagator.generate_proposals(datapacks, ontology, task_graph, economics)
    assert dp_copy == datapacks and ont_copy == ontology and tg_copy == task_graph and econ_copy == economics
    print("[TEST 6 PASS] Inputs remain unchanged")

    # Stable ordering
    assert [p.episode_id for p in proposals_1] == ["ep_001", "ep_002"]
    if proposals_1:
        frag_orders = [t.object_name for t in proposals_1[-1].fragility_tags]
        assert frag_orders == sorted(frag_orders)
    print("[TEST 7 PASS] Stable ordering of proposals and tags")

    # Missing ontology graceful degradation
    missing_onto = propagator.generate_proposals(datapacks[:1], [], task_graph, economics)[0]
    assert missing_onto.validation_status == "passed"
    assert missing_onto.confidence < 0.8
    assert not missing_onto.affordance_tags
    print("[TEST 8 PASS] Missing ontology handled gracefully")

    # Missing task graph graceful degradation
    missing_task_graph = propagator.generate_proposals(datapacks[:1], ontology, [], economics)[0]
    assert missing_task_graph.validation_status == "passed"
    assert not missing_task_graph.risk_tags
    assert not missing_task_graph.efficiency_tags
    print("[TEST 9 PASS] Missing task graph handled gracefully")

    # Missing economics hard fail
    missing_econ = propagator.generate_proposals(datapacks[:1], ontology, task_graph, {})[0]
    assert missing_econ.validation_status == "failed"
    assert any("economics" in err.lower() for err in missing_econ.validation_errors)
    print("[TEST 10 PASS] Missing economics triggers validation failure")

    # Partial datapack resilience
    partial_dp = {
        "video_id": "drawer_partial",
        "episode_id": "ep_partial",
        "task": "open_drawer",
        "frames": [0, 1],
        "actions": [],
        "metadata": {"success": True},
    }
    econ_partial = {"ep_partial": {"novelty_score": 0.5, "expected_mpl_gain": 2.0, "tier": 1}}
    partial_result = propagator.generate_proposals([partial_dp], ontology, task_graph, econ_partial)[0]
    assert partial_result.validation_status == "passed"
    assert partial_result.confidence < 0.9
    print("[TEST 11 PASS] Partial datapack handled gracefully")

    # Rejected proposals filtered
    rejected_ontology = [
        {
            "proposal_id": "onto_prop_failed",
            "validation_status": "failed",
            "new_affordances": [{"name": "graspable", "object": "handle"}],
        }
    ]
    filtered = propagator.generate_proposals(datapacks[:1], rejected_ontology, task_graph, economics)[0]
    assert "onto_prop_failed" not in filtered.source_proposals
    assert not filtered.affordance_tags
    print("[TEST 12 PASS] Rejected proposals filtered out")

    # Aggregation sanity check
    aggregated = propagator.aggregate_for_task(proposals_1, task="open_drawer")
    assert aggregated is not None
    assert aggregated.task == "open_drawer"
    assert aggregated.source_proposals
    print("[TEST 13 PASS] Aggregation across videos works")

    print("[smoke_test_stage2_4_semantic_tag_propagation] All tests passed!")


if __name__ == "__main__":
    main()
