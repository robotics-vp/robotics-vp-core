from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from src.motor_backend.datapacks import DatapackConfig


class RolloutLabeler(Protocol):
    def label_rollouts(
        self,
        rollouts_dir: str | Path,
        base_datapack: DatapackConfig | None = None,
    ) -> Sequence[DatapackConfig]:
        ...


@dataclass(frozen=True)
class StubRolloutLabeler:
    """Placeholder labeler for future VLA/vision integration."""

    tag_suffix: str = "vla_stub"

    def label_rollouts(
        self,
        rollouts_dir: str | Path,
        base_datapack: DatapackConfig | None = None,
    ) -> Sequence[DatapackConfig]:
        if base_datapack is None:
            return []
        tags = list(base_datapack.tags)
        if self.tag_suffix and self.tag_suffix not in tags:
            tags.append(self.tag_suffix)
        return [
            DatapackConfig(
                id=f"{base_datapack.id}_vla",
                description=base_datapack.description,
                motion_clips=list(base_datapack.motion_clips),
                domain_randomization=dict(base_datapack.domain_randomization),
                curriculum=dict(base_datapack.curriculum),
                tags=tags,
                task_tags=list(base_datapack.task_tags),
                robot_families=list(base_datapack.robot_families),
                objective_hint=base_datapack.objective_hint,
            )
        ]
