"""Tests for skill graph (B2).

Validates SkillGraph construction from the HRL registry, serialisation,
and missing-transition detection.
"""

from src.hrl.skill_graph import SkillGraph


def test_build_from_hrl_registry():
    sg = SkillGraph.build_from_registry(hrl_skills=True)
    # Should have all 6 HRL skills
    assert len(sg.nodes) == 6
    # Transition chain: 5 edges (6 skills → 5 transitions)
    assert len(sg.transitions) == 5
    # All should be missing (coverage_count=0 by default)
    assert len(sg.missing_transitions) == 5


def test_build_with_sima_sequences():
    sg = SkillGraph.build_from_registry(
        hrl_skills=False,
        sima_sequences=[{
            "task_id": "dishwash_demo",
            "skill_ids": ["pick_dish", "scrub", "rinse", "stow"],
        }],
    )
    assert len(sg.nodes) == 4
    # Chain: 3 transitions
    assert len(sg.transitions) == 3
    assert all(t.task_id == "dishwash_demo" for t in sg.transitions)


def test_build_with_vla_hints():
    sg = SkillGraph.build_from_registry(
        hrl_skills=False,
        vla_hints=[{
            "skill_id": "visual_grasp",
            "label": "Visual Grasp",
            "env_primitive_requirements": ["grasp_handle"],
        }],
    )
    assert len(sg.nodes) == 1
    assert sg.nodes[0].env_primitive_requirements == ["grasp_handle"]


def test_serialisation_round_trip():
    sg = SkillGraph.build_from_registry(hrl_skills=True)
    d = sg.to_dict()
    sg2 = SkillGraph.from_dict(d)
    assert len(sg2.nodes) == len(sg.nodes)
    assert len(sg2.transitions) == len(sg.transitions)


def test_node_lookup():
    sg = SkillGraph.build_from_registry(hrl_skills=True)
    node = sg.node_by_id("hrl:locate_drawer")
    assert node is not None
    assert node.skill_family == "hrl"
    assert sg.node_by_id("nonexistent") is None


def test_edges_for_task():
    sg = SkillGraph.build_from_registry(hrl_skills=True)
    edges = sg.edges_for_task("drawer_vase")
    assert len(edges) == 5
