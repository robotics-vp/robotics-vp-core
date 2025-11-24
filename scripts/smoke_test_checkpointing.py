#!/usr/bin/env python3
"""
Smoke test for Activation Checkpointing.
Verifies that models run with checkpointing enabled.
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.training_env import checkpoint_if_enabled

class CheckpointedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 64)
    
    def forward(self, x):
        x = checkpoint_if_enabled(self.layer1, x, enabled=True)
        x = checkpoint_if_enabled(self.layer2, x, enabled=True)
        return x

def smoke_test_checkpointing():
    print("[SmokeTest] Starting Checkpointing smoke test...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CheckpointedModule().to(device)
    inputs = torch.randn(16, 64, device=device, requires_grad=True)
    
    # Forward
    try:
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        print(f"[SmokeTest] Forward/Backward successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"[SmokeTest] Failed: {e}")
        sys.exit(1)

    print("[SmokeTest] Checkpointing smoke test passed!")

if __name__ == "__main__":
    smoke_test_checkpointing()
