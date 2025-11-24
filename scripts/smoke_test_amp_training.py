#!/usr/bin/env python3
"""
Smoke test for AMP (Automatic Mixed Precision) training.
Runs a minimal training loop with AMP enabled to verify stability and logging.
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.training_env import should_use_amp, run_with_oom_recovery

def smoke_test_amp():
    print("[SmokeTest] Starting AMP training smoke test...")
    
    if not torch.cuda.is_available():
        print("[SmokeTest] CUDA not available. Skipping AMP test.")
        return

    device = torch.device("cuda")
    model = nn.Linear(128, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    # Dummy data
    inputs = torch.randn(16, 128, device=device)
    targets = torch.randn(16, 64, device=device)
    
    def train_step():
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    # Run wrapped step
    try:
        loss = run_with_oom_recovery(train_step)
        print(f"[SmokeTest] Step successful. Loss: {loss:.4f}")
    except Exception as e:
        print(f"[SmokeTest] Step failed: {e}")
        sys.exit(1)

    print("[SmokeTest] AMP smoke test passed!")

if __name__ == "__main__":
    smoke_test_amp()
