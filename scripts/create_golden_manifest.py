import numpy as np
import json
import hashlib
from pathlib import Path

def create_dummy_golden():
    output_dir = Path("data/golden_clips")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = output_dir / "dummy_golden.npz"
    
    # Create dummy data
    T, K = 100, 2
    ir_loss = np.zeros((T, K)) # Perfect loss
    converged = np.ones((T, K), dtype=bool)
    
    summary = {
        "id_switch_count": 0,
        "pct_diverged": 0.0
    }
    
    np.savez(
        npz_path,
        **{
            "scene_tracks_v1/ir_loss": ir_loss,
            "scene_tracks_v1/converged": converged,
            "scene_tracks_v1/summary_json": np.array([json.dumps(summary)])
        }
    )
    print(f"Created {npz_path}")
    
    # Compute hash
    sha256 = hashlib.sha256()
    with npz_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()
    
    # Create manifest
    manifest = {
        "version": "1.0",
        "entries": [
            {
                "path": "golden_clips/dummy_golden.npz",
                "sha256": file_hash,
                "tolerances": {
                    "max_id_switch_rate": 0.5,
                    "max_ir_loss_p90": 0.2,
                    "max_pct_diverged": 5.0,
                    "max_runtime_ms": 100.0
                }
            }
        ]
    }
    
    manifest_path = Path("data/golden_clips_manifest.json")
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Created {manifest_path}")

if __name__ == "__main__":
    create_dummy_golden()
