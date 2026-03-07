import numpy as np

from src.constraints.constraint_set import ConstraintSet
from src.economics.econ_tensor import EconTensor
from src.objectives.runtime_builder import (
    ObjectiveRuntimeBuilder,
    ObjectiveRuntimeRecord,
    SourceDomain,
)
from src.runtime.packets import (
    RuntimePacket,
    SchemaRef,
    build_contract_packet,
    runtime_packet_from_record,
)


def _record() -> ObjectiveRuntimeRecord:
    return ObjectiveRuntimeRecord(
        task_id="shadow_kitting",
        episode_id="ep_runtime_packet_001",
        env_id="workcell_simple",
        world_id="workcell_assembly_bench_simple",
        robot_id="shadow_sim_arm_v1",
        source_domain=SourceDomain.SYNTHETIC,
        seed=9,
        run_id="shadow_runtime_packet",
        timestamp="2026-03-07T00:00:00+00:00",
        episode_metrics={
            "items_completed": 4,
            "duration_s": 600.0,
            "error_rate": 0.03,
            "energy_wh": 1.1,
            "energy_wh_per_unit": 0.275,
            "safety_score": 0.95,
        },
        reward_components={
            "mpl_component": 0.8,
            "energy_penalty": 0.15,
            "error_penalty": 0.02,
        },
        telemetry={"trust_score": 0.9, "uncertainty": 0.1},
    )


def test_runtime_packet_round_trip_preserves_contract_and_tensors():
    record = _record()
    objective_tensor = ObjectiveRuntimeBuilder().build(record)
    econ_tensor = EconTensor(
        values=np.asarray([14.0, 31.0, 0.45, 0.1, 0.05], dtype=np.float32),
        context={"source": "shadow_runtime"},
    )
    constraint_set = ConstraintSet.from_runtime(
        hard_constraints={"semantic_disagreement_vla_vs_map": {"max": 0.25}},
        uncertainty={"semantic_disagreement": 0.12},
        metadata={"producer": "test_runtime_packets"},
    )
    packet = runtime_packet_from_record(
        record=record,
        contract_id="contract.shadow.kitting.v1",
        objective_profile_id="balanced_contract",
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
        observation_schema=SchemaRef(
            schema_id="observation_schema_v2",
            timing={"sample_hz": 10.0},
            shape={"obs_vector": 48},
        ),
        action_schema=SchemaRef(
            schema_id="action_schema_v2",
            timing={"apply_hz": 10.0},
            shape={"action_vector": 7},
        ),
        semantic_evidence={"teacher_trace_ref": "artifacts/teacher_trace.json", "vla_confidence": 0.72},
        uncertainty={"planner": 0.11, "semantic": 0.17},
        metadata={"path": "shadow_runtime"},
    )

    assert packet.contract.embodiment_id == "shadow_sim_arm_v1"
    assert packet.contract.observation_schema.schema_id == "observation_schema_v2"
    assert packet.contract.objective_schema_id == "objective_tensor_runtime_v1"
    assert packet.objective_tensor["context"]["episode_id"] == "ep_runtime_packet_001"
    assert packet.econ_tensor["schema_id"] == "econ_tensor_runtime_v1"
    assert packet.constraint_set["version"] == "constraint_set_v1"
    assert packet.summary()["contract_id"] == "contract.shadow.kitting.v1"
    assert packet.summary()["packet_hash"]

    restored = RuntimePacket.from_dict(packet.to_dict())
    assert restored.to_dict() == packet.to_dict()


def test_contract_packet_hash_changes_when_action_schema_changes():
    record = _record()
    objective_tensor = ObjectiveRuntimeBuilder().build(record)
    econ_tensor = EconTensor(values=np.asarray([10.0, 25.0, 0.2, 0.0, 0.02], dtype=np.float32))
    constraint_set = ConstraintSet.from_runtime()

    contract_a = build_contract_packet(
        contract_id="contract.shadow.kitting.v1",
        task_id=record.task_id,
        objective_profile_id="balanced_contract",
        embodiment_id=record.robot_id,
        source_domain=record.source_domain.value,
        observation_schema=SchemaRef(schema_id="observation_schema_v2"),
        action_schema=SchemaRef(schema_id="action_schema_v2"),
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
    )
    contract_b = build_contract_packet(
        contract_id="contract.shadow.kitting.v1",
        task_id=record.task_id,
        objective_profile_id="balanced_contract",
        embodiment_id=record.robot_id,
        source_domain=record.source_domain.value,
        observation_schema=SchemaRef(schema_id="observation_schema_v2"),
        action_schema=SchemaRef(schema_id="action_schema_v3"),
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
    )

    assert contract_a.contract_hash != contract_b.contract_hash
