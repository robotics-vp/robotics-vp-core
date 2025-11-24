#!/usr/bin/env python3
"""
End-to-end smoke test for Stage 6 training pipeline.
Runs all training scripts with minimal data and epochs to verify connectivity and flags.
"""
import subprocess
import sys
from pathlib import Path

def run_smoke_test():
    print("[SmokeTest] Starting Stage 6 End-to-End Smoke Test...")
    
    base_cmd = [sys.executable]
    
    # 1. Vision Backbone (1 epoch, small batch)
    print("\n[SmokeTest] Testing Vision Backbone...")
    cmd = base_cmd + ["scripts/train_vision_backbone_real.py", 
                      "--epochs=1", "--batch-size=2", "--max-samples=4", "--use-mixed-precision"]
    subprocess.run(cmd, check=True)

    # 2. SIMA-2 Segmenter (1 epoch, small batch)
    print("\n[SmokeTest] Testing SIMA-2 Segmenter...")
    cmd = base_cmd + ["scripts/train_sima2_segmenter.py",
                      "--epochs=1", "--batch-size=2", "--use-mixed-precision"]
    subprocess.run(cmd, check=True)

    # 3. Spatial RNN (1 epoch)
    print("\n[SmokeTest] Testing Spatial RNN...")
    cmd = base_cmd + ["scripts/train_spatial_rnn.py",
                      "--epochs=1", "--sequence-length=4", "--use-mixed-precision"]
    subprocess.run(cmd, check=True)

    # 4. Hydra Policy (few steps)
    print("\n[SmokeTest] Testing Hydra Policy...")
    cmd = base_cmd + ["scripts/train_hydra_policy.py",
                      "--max-steps=10", "--max-samples=4", "--use-mixed-precision"]
    subprocess.run(cmd, check=True)

    print("\n[SmokeTest] All components passed smoke test!")

if __name__ == "__main__":
    try:
        run_smoke_test()
    except subprocess.CalledProcessError as e:
        print(f"\n[SmokeTest] FAILED: {e}")
        sys.exit(1)
