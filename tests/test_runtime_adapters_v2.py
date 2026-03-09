import numpy as np

from src.constraints.constraint_set import ConstraintSet
from src.economics.econ_tensor import EconTensor
from src.objectives.runtime_builder import ObjectiveRuntimeBuilder
from src.runtime import ActionAdapterV2, ObservationAdapterV2, build_contract_packet

from tests.test_runtime_packets import _record


def test_runtime_adapters_v2_normalize_and_build_schema_refs() -> None:
    action_adapter = ActionAdapterV2(
        schema_id="action_schema.workcell.v2",
        control_hz=20.0,
        translator_ref="translator://workcell",
        embodiment_id="shadow_sim_arm_v1",
    )
    observation_adapter = ObservationAdapterV2(
        schema_id="observation_schema.workcell.v2",
        proprio_fields=["joint_0", "joint_1"],
        sensor_refs=["rgb_front", "scene_tracks", "belief_state"],
        sample_hz=10.0,
        translator_ref="translator://workcell_obs",
        embodiment_id="shadow_sim_arm_v1",
    )

    normalized_action = action_adapter.normalize({"dx": 0.1, "gripper": 1.0})
    normalized_observation = observation_adapter.normalize(
        {
            "proprio": {"joint_0": 0.5, "joint_1": -0.25},
            "quality_metrics": {"scene_ir_quality": 0.7},
            "belief_state_ref": "artifact://belief_state",
        }
    )

    assert normalized_action["vector"][0] == 0.1
    assert normalized_action["timing"]["apply_hz"] == 20.0
    assert normalized_observation["proprio_vector"] == [0.5, -0.25]
    assert normalized_observation["artifact_refs"]["belief_state_ref"] == "artifact://belief_state"

    record = _record()
    objective_tensor = ObjectiveRuntimeBuilder().build(record)
    econ_tensor = EconTensor(values=np.asarray([14.0, 31.0, 0.45, 0.1, 0.05], dtype=np.float32))
    constraint_set = ConstraintSet.from_runtime()

    contract = build_contract_packet(
        contract_id="contract.workcell.v2",
        task_id=record.task_id,
        objective_profile_id="balanced_contract",
        embodiment_id=record.robot_id,
        source_domain=record.source_domain.value,
        observation_schema=observation_adapter,
        action_schema=action_adapter,
        objective_tensor=objective_tensor,
        econ_tensor=econ_tensor,
        constraint_set=constraint_set,
    )

    assert contract.action_schema.schema_id == "action_schema.workcell.v2"
    assert contract.observation_schema.timing["sample_hz"] == 10.0
