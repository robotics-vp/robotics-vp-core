"""Datasets for Motion Hierarchy Node training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticChainDataset(Dataset):
    """
    Toy kinematic chain:
    - N nodes in 2D or 3D.
    - Single root, linear child chain.
    - Root follows a smooth random trajectory.
    - Each child is root + cumulative offset + small local noise.
    """

    def __init__(
        self,
        num_sequences: int = 1024,
        T: int = 32,
        N: int = 8,
        D: int = 2,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.num_sequences = int(num_sequences)
        self.T = int(T)
        self.N = int(N)
        self.D = int(D)
        self.device = torch.device(device)

        rng = np.random.default_rng(seed)
        self._positions: List[torch.Tensor] = []
        self._parent_index: List[torch.Tensor] = []

        for _ in range(self.num_sequences):
            root_vel = rng.normal(scale=0.05, size=(self.T, self.D)).astype(np.float32)
            root_pos = np.cumsum(root_vel, axis=0)

            offsets = rng.normal(scale=0.4, size=(self.N, self.D)).astype(np.float32)
            for i in range(1, self.N):
                offsets[i] = offsets[i - 1] + rng.normal(scale=0.2, size=(self.D,)).astype(np.float32)

            positions = np.zeros((self.T, self.N, self.D), dtype=np.float32)
            for i in range(self.N):
                noise = rng.normal(scale=0.02, size=(self.T, self.D)).astype(np.float32)
                positions[:, i, :] = root_pos + offsets[i] + noise

            parent_index = np.zeros((self.N,), dtype=np.int64)
            if self.N > 1:
                parent_index[1:] = np.arange(self.N - 1, dtype=np.int64)

            self._positions.append(torch.tensor(positions, device=self.device))
            self._parent_index.append(torch.tensor(parent_index, device=self.device))

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "positions": self._positions[idx],
            "parent_index": self._parent_index[idx],
        }


@dataclass
class MotionHierarchyBatch:
    positions: torch.Tensor  # (B, T, N, D)
    mask: Optional[torch.Tensor]
    metadata: Dict[str, Any]


class TrajectoryDatasetAdapter(Dataset):
    """
    Wraps existing LSD rollout / env logs and exposes a MotionHierarchyBatch.
    """

    def __init__(self, batches: Sequence[MotionHierarchyBatch]) -> None:
        self._batches = list(batches)

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> MotionHierarchyBatch:
        return self._batches[idx]

    @classmethod
    def from_positions(
        cls,
        positions: Sequence[torch.Tensor],
        masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "TrajectoryDatasetAdapter":
        batches: List[MotionHierarchyBatch] = []
        masks = masks or [None] * len(positions)
        metadata = metadata or [{} for _ in positions]
        for pos, mask, meta in zip(positions, masks, metadata):
            batches.append(MotionHierarchyBatch(positions=pos, mask=mask, metadata=dict(meta)))
        return cls(batches)
