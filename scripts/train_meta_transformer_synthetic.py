#!/usr/bin/env python3
"""
Synthetic MetaTransformer pretraining runner.

Lightweight: generates random embeddings/labels and runs a few SGD steps.
No impact on other training loops.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.orchestrator.meta_transformer import MetaTransformer


class TinyMetaModel(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    d_in = 16
    model = TinyMetaModel(d_in=d_in)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Synthetic dataset
    X = torch.randn(64, d_in)
    y = torch.randn(64, 2)

    losses = []
    for step in range(10):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"Step {step}: loss={loss.item():.4f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/meta_transformer_synthetic.pt")
    with open("results/meta_transformer_synthetic_losses.txt", "w") as f:
        f.write("\n".join(str(l) for l in losses))
    print("Saved synthetic checkpoint and losses.")


if __name__ == "__main__":
    main()
