from __future__ import annotations

from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.motor_backend.rollout_capture import RolloutBundle


def label_rollouts_with_vla(
    rollouts: RolloutBundle,
    base_datapack: DatapackConfig,
) -> list[DatapackConfig]:
    """
    Call into the VLA/vision stack to label rollouts and produce new datapacks.
    """
    if not rollouts.episodes:
        return []

    # TODO: replace this stub with real VLA integration:
    # - Call into our VLA model / service
    # - Use detection results to define specific motion segments & tags
    first = rollouts.episodes[0]
    derived_tags = list(base_datapack.tags)
    for tag in ("auto_labeled", "vla_stub"):
        if tag not in derived_tags:
            derived_tags.append(tag)

    derived = DatapackConfig(
        id=f"{base_datapack.id}_vla",
        description=f"{base_datapack.description} (auto-labeled)",
        motion_clips=[
            MotionClipSpec(path=str(first.trajectory_path), weight=1.0),
        ],
        domain_randomization=dict(base_datapack.domain_randomization),
        curriculum=dict(base_datapack.curriculum),
        tags=derived_tags,
        task_tags=list(base_datapack.task_tags),
        robot_families=list(base_datapack.robot_families),
        objective_hint=base_datapack.objective_hint or "auto-labeled",
    )

    return [derived]
