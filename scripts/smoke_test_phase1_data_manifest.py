"""
Smoke test for Phase I data manifest generator.

Ensures manifest builds, is JSON-safe, and lists required dataset entries.
"""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.json_safe import to_json_safe  # noqa: E402


MANIFEST_PATH = ROOT / "results" / "phase1" / "data_manifest.json"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> None:
    script_path = ROOT / "scripts" / "build_phase1_data_manifest.py"
    module = load_module(script_path)
    module.main(["--max-samples", "8", "--seed", "5", "--output", str(MANIFEST_PATH)])
    assert MANIFEST_PATH.exists(), "Manifest file was not created"

    manifest: Dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    assert manifest.get("phase") == "phase1"
    required_datasets = {"vision", "spatial_rnn", "sima2_segmenter", "hydra_policy"}
    assert required_datasets.issubset(manifest.get("datasets", {}).keys()), "Missing dataset entries in manifest"
    for name in required_datasets:
        entry = manifest["datasets"][name]
        assert entry.get("count", 0) > 0, f"Dataset {name} has zero samples"
        json.dumps(to_json_safe(entry), sort_keys=True)

    print("Phase I data manifest smoke test: PASSED")


if __name__ == "__main__":
    main()
