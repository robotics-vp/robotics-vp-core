from src.embodiment.registry import (
    CapabilityProfile,
    EmbodimentRegistry,
    EmbodimentRegistryEntry,
)


def _entry(
    embodiment_id: str = "workcell_arm_v1",
    action_schema_id: str = "action_schema_v2",
) -> EmbodimentRegistryEntry:
    profile = CapabilityProfile(
        profile_id=f"profile::{embodiment_id}",
        robot_family="workcell_arm",
        sensor_modalities=["rgb", "depth", "joint_state"],
        action_spaces=["cartesian_delta", "gripper_binary"],
        workspace_bounds={"x": {"min": -0.6, "max": 0.6}, "z": {"min": 0.0, "max": 1.2}},
        skill_capabilities={"pick": 1.0, "place": 1.0, "insert": 0.6},
        timing={"control_hz": 20.0, "observation_latency_ms": 80.0},
        safety_envelopes={"max_gripper_force": 20.0},
    )
    return EmbodimentRegistryEntry(
        embodiment_id=embodiment_id,
        robot_id=embodiment_id,
        robot_family="workcell_arm",
        capability_profile=profile,
        observation_schema_id="observation_schema_v2",
        action_schema_id=action_schema_id,
        translator_refs={
            "observation": "src.runtime.observation_adapter_v2:translate",
            "action": "src.runtime.action_adapter_v2:translate",
        },
        metadata={"site": "shadow_workcell"},
    )


def test_embodiment_registry_upserts_and_resolves_schema_ids():
    registry = EmbodimentRegistry()
    registry.register(_entry())

    assert registry.resolve_observation_schema("workcell_arm_v1") == "observation_schema_v2"
    assert registry.resolve_action_schema("workcell_arm_v1") == "action_schema_v2"
    assert registry.resolve_capability_profile("workcell_arm_v1").skill_capabilities["pick"] == 1.0

    registry.register(_entry(action_schema_id="action_schema_v3"))
    assert registry.resolve_action_schema("workcell_arm_v1") == "action_schema_v3"


def test_embodiment_registry_round_trip_is_stable():
    registry = EmbodimentRegistry(
        entries=[
            _entry(),
            _entry(embodiment_id="mobile_manipulator_v1", action_schema_id="action_schema_mobile_v2"),
        ]
    )

    restored = EmbodimentRegistry.from_dict(registry.to_dict())

    assert [entry.embodiment_id for entry in restored.list_entries()] == [
        "mobile_manipulator_v1",
        "workcell_arm_v1",
    ]
    assert restored.get("mobile_manipulator_v1").translator_refs["action"] == (
        "src.runtime.action_adapter_v2:translate"
    )
