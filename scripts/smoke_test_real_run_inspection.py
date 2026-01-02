import subprocess
import json
import sys
import shutil
from pathlib import Path

def test_real_run():
    output_dir = Path("results/real_run_inspection")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Scene IR Tracker on synthetic LSD episode...")
    cmd = [
        sys.executable,
        "scripts/run_scene_ir_tracker_on_lsd.py",
        "--num-episodes", "1",
        "--num-frames", "20",
        "--save-overlays",
        "--output", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Tracker run failed:")
        print(result.stderr)
        sys.exit(1)
        
    print("Tracker run completed.")
    
    # Verify outputs
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        print("FAIL: metrics.json not found")
        sys.exit(1)
        
    overlay_dir = output_dir / "episode_0000" / "overlays"
    if not overlay_dir.exists():
        print("FAIL: Overlays directory not found")
        sys.exit(1)
        
    overlays = list(overlay_dir.glob("*.png"))
    if not overlays:
        print("FAIL: No overlay images found")
        sys.exit(1)
        
    print(f"Generated {len(overlays)} overlays.")
    print(f"SUCCESS. Please inspect overlays at: {overlay_dir}")
    print("[smoke_test_real_run_inspection] PASS")

if __name__ == "__main__":
    test_real_run()
