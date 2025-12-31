import torch
from torch.utils.data import DataLoader

from src.vision.motion_hierarchy.config import MotionHierarchyConfig
from src.vision.motion_hierarchy.datasets import SyntheticChainDataset
from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode


def test_motion_hierarchy_training_reduces_loss():
    torch.manual_seed(0)
    dataset = SyntheticChainDataset(num_sequences=32, T=16, N=6, D=2, device="cpu", seed=0)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    positions = batch["positions"]

    config = MotionHierarchyConfig(
        d_model=32,
        num_gnn_layers=2,
        k_neighbors=4,
        l_max=2,
        lambda_residual=0.5,
        use_batch_norm=False,
        gumbel_tau=0.7,
        gumbel_hard=False,
        device="cpu",
    )
    model = MotionHierarchyNode(config)

    initial_loss = model(positions, return_losses=True)["losses"]["total"].item()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = [initial_loss]
    for _ in range(30):
        out = model(positions, return_losses=True)
        loss = out["losses"]["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert min(losses) < initial_loss
