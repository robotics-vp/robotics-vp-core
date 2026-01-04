"""Training utilities for Motion Hierarchy Node."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.vision.motion_hierarchy.config import MotionHierarchyConfig
from src.vision.motion_hierarchy.datasets import SyntheticChainDataset
from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode


@dataclass
class MotionHierarchyTrainer:
    model: MotionHierarchyNode
    config: MotionHierarchyConfig
    optimizer: torch.optim.Optimizer
    device: torch.device

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        positions = batch["positions"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)
        return {"positions": positions, "mask": mask, "parent_index": batch.get("parent_index")}

    def _ensure_optimizer_params(self) -> None:
        existing = {id(p) for group in self.optimizer.param_groups for p in group.get("params", [])}
        missing = [p for p in self.model.parameters() if id(p) not in existing]
        if missing:
            self.optimizer.add_param_group({"params": missing})

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "recon": 0.0, "resid": 0.0}
        count = 0
        for batch in dataloader:
            data = self._prepare_batch(batch)
            out = self.model(data["positions"], mask=data["mask"], return_losses=True)
            self._ensure_optimizer_params()
            losses = out["losses"]
            loss_total = losses["total"]

            self.optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            self.optimizer.step()

            totals["total"] += float(losses["total"].item())
            totals["recon"] += float(losses["recon"].item())
            totals["resid"] += float(losses["resid"].item())
            count += 1
        if count == 0:
            return {k: 0.0 for k in totals}
        return {k: v / count for k, v in totals.items()}

    def eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals = {"total": 0.0, "recon": 0.0, "resid": 0.0}
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                data = self._prepare_batch(batch)
                out = self.model(data["positions"], mask=data["mask"], return_losses=True)
                losses = out["losses"]
                totals["total"] += float(losses["total"].item())
                totals["recon"] += float(losses["recon"].item())
                totals["resid"] += float(losses["resid"].item())
                count += 1
        if count == 0:
            return {k: 0.0 for k in totals}
        return {k: v / count for k, v in totals.items()}


def _compute_parent_accuracy(
    hierarchy: torch.Tensor,
    parent_index: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    pred_parent = hierarchy.argmax(dim=-1)
    correct = pred_parent == parent_index
    if mask is not None:
        correct = correct & mask.bool()
        denom = mask.sum().clamp(min=1).float()
        return float(correct.float().sum() / denom)
    return float(correct.float().mean())


def run_synthetic_training(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Train MHN on a synthetic chain dataset and print losses."""
    torch.manual_seed(0)
    cfg = MotionHierarchyConfig(
        d_model=64,
        num_gnn_layers=2,
        k_neighbors=6,
        l_max=3,
        lambda_residual=0.5,
        use_batch_norm=True,
        gumbel_tau=0.7,
        gumbel_hard=False,
        device=device,
    )
    dataset = SyntheticChainDataset(num_sequences=256, T=24, N=8, D=2, device="cpu", seed=0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MotionHierarchyNode(cfg).to(device)
    warmup = next(iter(dataloader))
    _ = model(warmup["positions"].to(device), return_losses=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = MotionHierarchyTrainer(model=model, config=cfg, optimizer=optimizer, device=torch.device(device))

    for epoch in range(epochs):
        metrics = trainer.train_epoch(dataloader)
        sample = next(iter(dataloader))
        sample_positions = sample["positions"].to(device)
        sample_parents = sample["parent_index"].to(device)
        with torch.no_grad():
            out = model(sample_positions, return_losses=False)
        acc = _compute_parent_accuracy(out["hierarchy"], sample_parents)
        print(
            f"epoch={epoch + 1} total={metrics['total']:.4f} "
            f"recon={metrics['recon']:.4f} resid={metrics['resid']:.4f} "
            f"parent_acc={acc:.3f}"
        )
