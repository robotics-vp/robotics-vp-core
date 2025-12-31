#!/usr/bin/env python3
"""
End-to-end smoke test for Stage 6 training pipeline.
Runs the master orchestrator with minimal epochs to verify connectivity and flags.
"""
import subprocess
import sys
from pathlib import Path

def run_smoke_test():
    print("[SmokeTest] Starting Stage 6 End-to-End Smoke Test...")
    
    # Run orchestrator with minimal epochs/samples
    cmd = [
        sys.executable, "scripts/run_stage6_train_all.py", 
        "--seed=0", 
        "--use-mixed-precision", 
        "--epochs=1"
    ]
    
    print(f"\n[SmokeTest] Running Orchestrator: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Check marker
    marker = Path("results/stage6/success.json")
    if marker.exists():
        print(f"\n[SmokeTest] Success marker found: {marker}")
        with open(marker) as f:
            print(f"  Content: {f.read()}")
    else:
        print(f"\n[SmokeTest] FAILED: Success marker not found at {marker}")
        sys.exit(1)

    print("\n[SmokeTest] Stage 6 Pipeline Passed!")

if __name__ == "__main__":
    try:
        run_smoke_test()
    except subprocess.CalledProcessError as e:
        print(f"\n[SmokeTest] FAILED: {e}")
        sys.exit(1)
