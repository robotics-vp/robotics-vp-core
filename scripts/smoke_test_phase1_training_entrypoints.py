"""
Smoke test for Phase I training entrypoints.

Validates:
- Entrypoints parse flags and run deterministic lightweight training
- Checkpoints are written and deterministic across runs
- JSON-safe logging
"""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.json_safe import to_json_safe  # noqa: E402


CHECKPOINT_DIR = ROOT / "checkpoints"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        import torch

        return to_json_safe(torch.load(path, map_location="cpu"))
    except Exception:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}


def run_entrypoint(script_path: Path, checkpoint_name: str, extra_args: List[str]) -> Dict[str, Any]:
    module = load_module(script_path)
    argv = ["--max-steps", "2", "--max-samples", "4", "--checkpoint-dir", str(CHECKPOINT_DIR)]
    argv.extend(extra_args)
    module.main(argv)
    ckpt_path = CHECKPOINT_DIR / checkpoint_name
    assert ckpt_path.exists(), f"Missing checkpoint {ckpt_path}"
    loaded = load_checkpoint(ckpt_path)
    payload = loaded.get("payload", loaded)
    assert payload.get("trained_steps", 0) >= 1, f"{checkpoint_name} should record training steps"
    return payload


def main() -> None:
    scripts = [
        ("train_vision_backbone.py", "vision_backbone_phase1.pt"),
        ("train_spatial_rnn.py", "spatial_rnn_phase1.pt"),
        ("train_sima2_segmenter.py", "sima2_segmenter_phase1.pt"),
        ("train_hydra_policy.py", "hydra_policy_phase1.pt"),
    ]
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for script_name, ckpt in scripts:
        script_path = ROOT / "scripts" / script_name
        first = run_entrypoint(script_path, ckpt, ["--seed", "42"])
        second = run_entrypoint(script_path, ckpt, ["--seed", "42"])
        assert first.get("deterministic_digest") == second.get("deterministic_digest"), f"{script_name} is nondeterministic"
        json.dumps(to_json_safe(first), sort_keys=True)
    print("Phase I training entrypoints smoke test: PASSED")


if __name__ == "__main__":
    main()
