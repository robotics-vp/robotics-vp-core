from pathlib import Path

from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics import VectorizedSimRunner, compile_sim_synth_physics_world_state


def _graph(label: str) -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode(f"task:{label}", "task", label),
            CoverageNode(f"skill:{label}", "skill", f"{label}_skill"),
            CoverageNode(f"risk:{label}", "risk_family", "collision"),
        ],
        edges=[
            CoverageEdge(
                f"skill:{label}",
                f"risk:{label}",
                "requires",
                evidence_count=0,
                economic_priority=0.6,
                trust_priority=0.4,
                promotion_readiness=0.2,
            )
        ],
    )


def test_vectorized_sim_runner_executes_cpu_local_batch(tmp_path: Path) -> None:
    world_states = [
        compile_sim_synth_physics_world_state(_graph("drawer")),
        compile_sim_synth_physics_world_state(_graph("vase")),
    ]

    result = VectorizedSimRunner().execute_world_states(
        world_states,
        output_root=tmp_path,
    )

    assert result.execution_mode == "sequential_batch"
    assert result.metadata["result_count"] == 2
    assert result.metadata["route_status_counts"] == {"ready": 2}
    assert len(result.results) == 2
    assert all(loop_result.physics_execution_contract.route_status == "ready" for loop_result in result.results)
    assert len(list(tmp_path.glob("window_*"))) == 2
