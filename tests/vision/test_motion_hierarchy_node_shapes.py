import torch

from src.vision.motion_hierarchy.config import MotionHierarchyConfig
from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode


def test_motion_hierarchy_shapes():
    torch.manual_seed(0)
    positions = torch.randn(2, 10, 5, 3)
    config = MotionHierarchyConfig(
        d_model=32,
        num_gnn_layers=2,
        k_neighbors=3,
        l_max=2,
        lambda_residual=0.5,
        use_batch_norm=False,
        gumbel_tau=0.7,
        gumbel_hard=True,
        device="cpu",
    )
    model = MotionHierarchyNode(config)
    out = model(positions, return_losses=True)

    deltas = out["deltas"]
    delta_hat = out["delta_hat"]
    delta_resid = out["delta_resid"]
    hierarchy = out["hierarchy"]

    assert deltas.shape == (2, 9, 5, 3)
    assert delta_hat.shape == deltas.shape
    assert delta_resid.shape == deltas.shape
    assert hierarchy.shape == (2, 5, 5)

    row_sums = hierarchy.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)

    for key in ("deltas", "delta_hat", "delta_resid", "hierarchy", "parent_logits", "parent_probs"):
        assert not torch.isnan(out[key]).any()
