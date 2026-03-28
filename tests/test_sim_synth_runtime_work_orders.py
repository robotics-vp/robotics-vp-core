from __future__ import annotations

from src.world_model.sim_synth_physics.receipts import (
    BackendRuntimeBridgeReceipt,
    BackendRuntimeExecutionReceipt,
    BackendRuntimeOutcomeReceipt,
    RobotAssetContractReceipt,
)
from src.world_model.sim_synth_physics.runtime_work_orders import (
    build_backend_runtime_work_orders,
)


def test_build_backend_runtime_work_orders_blocks_on_runtime_targets() -> None:
    bridge_receipt = BackendRuntimeBridgeReceipt(
        receipt_id="bridge_receipt_1",
        bridge_id="bridge_state_1",
        backend="isaac",
        bridge_status="runtime_targets_missing",
        execution_authority="shadow_runtime",
        transport_profile="isaac_shadow_bridge",
        planner_rate_hz=10.0,
        control_rate_hz=250.0,
        observation_rate_hz=60.0,
        action_decimation=4,
        latency_budget_ms=8.0,
        bridge_readiness_score=0.6,
        action_contracts=["whole_body_joint_command_v1"],
        observation_contracts=["imu_state_v1"],
        telemetry_contracts=["watchdog_state_v1"],
        safety_channels=["watchdog_v1"],
        metadata={
            "runtime_target_contract": {
                "runtime_targets_ready": False,
                "missing_required_target_ids": ["unitree_sdk2_root"],
            },
            "runtime_layout_contract": {
                "ready_profiles": ["unitree_sim_isaaclab"],
            },
            "policy_contract": {
                "policy_ready": False,
                "policy_root": "/tmp/policies",
            },
            "missing_assets": ["unitree_robot_description"],
        },
    )
    robot_asset_receipt = RobotAssetContractReceipt(
        receipt_id="asset_receipt_1",
        contract_id="asset_contract_1",
        asset_profile="unitree_humanoid_shadow_assets",
        target_hardware_class="unitree_g1_r1_class",
        readiness_score=0.4,
        required_assets=["unitree_robot_description"],
        missing_assets=["unitree_robot_description"],
    )
    runtime_receipt = BackendRuntimeExecutionReceipt(
        receipt_id="runtime_receipt_1",
        backend="isaac",
        execution_mode="workcell_isaaclab_evaluate_policy",
        execution_status="runtime_request_materialized_with_preconditions",
        metadata={
            "missing_preconditions": ["runtime_policy_id"],
            "launch_spec": {
                "command": "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py --task peg_in_hole --policy ${POLICY_REF} --headless"
            },
        },
    )

    work_orders = build_backend_runtime_work_orders(
        bridge_receipt=bridge_receipt,
        runtime_receipt=runtime_receipt,
        runtime_outcome_receipt=None,
        robot_asset_contract_receipt=robot_asset_receipt,
        world_state_id="world_state_1",
        physics_execution_contract_id="physics_contract_1",
    )

    assert len(work_orders) == 1
    assert work_orders[0].backend == "isaac"
    assert work_orders[0].status == "blocked_by_runtime_targets"
    assert "isaac_unitree_runtime_smoke" in work_orders[0].linked_backlog_ids
    assert work_orders[0].metadata["runtime_layout_ready_profiles"] == ["unitree_sim_isaaclab"]
    assert any("sim_main.py" in hint for hint in work_orders[0].command_hints)
    assert work_orders[0].metadata["policy_ready"] is False


def test_build_backend_runtime_work_orders_marks_concrete_runtime_complete() -> None:
    bridge_receipt = BackendRuntimeBridgeReceipt(
        receipt_id="bridge_receipt_2",
        bridge_id="bridge_state_2",
        backend="holosoma",
        bridge_status="runtime_bridge_ready",
        execution_authority="concrete_runtime",
        transport_profile="holosoma_motion_runtime_bridge",
        planner_rate_hz=60.0,
        control_rate_hz=120.0,
        observation_rate_hz=60.0,
        action_decimation=2,
        latency_budget_ms=0.0,
        bridge_readiness_score=1.0,
        metadata={"runtime_target_contract": {"runtime_targets_ready": True}},
    )

    work_orders = build_backend_runtime_work_orders(
        bridge_receipt=bridge_receipt,
        runtime_receipt=None,
        runtime_outcome_receipt=None,
        robot_asset_contract_receipt=None,
        world_state_id="world_state_2",
        physics_execution_contract_id="physics_contract_2",
    )

    assert len(work_orders) == 1
    assert work_orders[0].backend == "holosoma"
    assert work_orders[0].status == "satisfied_by_concrete_runtime"
    assert "holosoma_runtime_eval_smoke" in work_orders[0].linked_backlog_ids


def test_build_backend_runtime_work_orders_marks_external_runtime_outputs_complete() -> None:
    bridge_receipt = BackendRuntimeBridgeReceipt(
        receipt_id="bridge_receipt_3",
        bridge_id="bridge_state_3",
        backend="isaac",
        bridge_status="runtime_bridge_ready",
        execution_authority="shadow_runtime",
        transport_profile="isaaclab_unitree_dds_bridge",
        planner_rate_hz=10.0,
        control_rate_hz=250.0,
        observation_rate_hz=60.0,
        action_decimation=4,
        latency_budget_ms=8.0,
        bridge_readiness_score=0.85,
        metadata={
            "runtime_target_contract": {"runtime_targets_ready": True},
            "runtime_layout_contract": {"ready_profiles": ["unitree_sim_isaaclab"]},
            "policy_contract": {"policy_ready": True},
        },
    )
    runtime_receipt = BackendRuntimeExecutionReceipt(
        receipt_id="runtime_receipt_3",
        backend="isaac",
        execution_mode="workcell_isaaclab_evaluate_policy",
        execution_status="runtime_external_launch_completed",
    )
    runtime_outcome_receipt = BackendRuntimeOutcomeReceipt(
        receipt_id="runtime_outcome_receipt_3",
        backend="isaac",
        outcome_profile="unitree_sim_isaaclab",
        outcome_status="runtime_outputs_harvested",
        executed=True,
        harvested_output_count=3,
        artifact_refs=["/tmp/unitree_sim_isaaclab/logs/run_1/policy.onnx"],
    )

    work_orders = build_backend_runtime_work_orders(
        bridge_receipt=bridge_receipt,
        runtime_receipt=runtime_receipt,
        runtime_outcome_receipt=runtime_outcome_receipt,
        robot_asset_contract_receipt=None,
        world_state_id="world_state_3",
        physics_execution_contract_id="physics_contract_3",
    )

    assert len(work_orders) == 1
    assert work_orders[0].status == "satisfied_by_external_runtime_outcomes"
    assert (
        work_orders[0].metadata["backend_runtime_outcome_receipt_id"]
        == "runtime_outcome_receipt_3"
    )
    assert work_orders[0].metadata["backend_runtime_output_count"] == 3
