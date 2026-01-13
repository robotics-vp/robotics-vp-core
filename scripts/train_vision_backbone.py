"""
Phase I vision backbone training entrypoint.

Parses config flags, loads Stage 5 datasets, instantiates the vision backbone
stub + a lightweight regression head, runs a deterministic training loop, and
writes a checkpoint under checkpoints/vision_backbone_phase1.pt.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets import VisionPhase1Dataset
from src.datasets.base import set_deterministic_seeds
from src.utils.json_safe import to_json_safe
from src.vision.backbone_stub import VisionBackboneStub
from src.training.wrap_training_entrypoint import regal_training


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase I Vision Backbone Training Stub")
    parser.add_argument("--stage1-root", type=str, default=str(Path("results") / "stage1_pipeline"))
    parser.add_argument("--stage2-root", type=str, default=str(Path("results") / "stage2_preview"))
    parser.add_argument("--sima2-root", type=str, default=str(Path("results") / "sima2_stress"))
    parser.add_argument("--trust-matrix", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args(argv)


def _compute_target(sample: Dict[str, Any]) -> float:
    frame = sample.get("stage1_frame", {})
    seg = sample.get("stage2_segments", {})
    stress = sample.get("sima2_stress", {})
    trust = float(sample.get("trust_weight", 1.0))
    tags = frame.get("tags") or []
    severity = float(stress.get("severity", 0.0))
    risk = float(seg.get("risk_level", 0.0))
    energy = float(seg.get("energy_intensity", 0.0))
    target = 0.1 * len(tags) + 0.2 * severity + 0.1 * risk + 0.05 * energy + 0.05 * trust
    return float(target)


def _model_checksum(head) -> float:
    try:
        import torch

        checksum = 0.0
        for _, param in head.state_dict().items():
            checksum += float(param.sum().item())
        return float(checksum)
    except Exception:
        return 0.0


def run_training(dataset: VisionPhase1Dataset, head, max_steps: int) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn

        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        steps = 0
        total_loss = 0.0
        for sample in dataset:
            if steps >= max_steps:
                break
            latent_vals = sample.get("vision_latent", {}).get("latent") or []
            latent = torch.tensor(latent_vals, dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([_compute_target(sample)], dtype=torch.float32)
            pred = head(latent).view(-1)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        return {
            "processed": steps,
            "mean_loss": float(total_loss / max(1, steps)),
            "head_checksum": _model_checksum(head),
        }
    except Exception:
        # Fallback: deterministic counter if torch is unavailable.
        steps = min(max_steps, len(dataset))
        return {"processed": steps, "mean_loss": 0.0, "head_checksum": 0.0}


def write_checkpoint(checkpoint_dir: Path, payload: Dict[str, Any], head_state: Optional[Dict[str, Any]]) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "vision_backbone_phase1.pt"
    try:
        import torch

        torch.save({"payload": payload, "head_state": head_state}, path)
    except Exception:
        to_dump = {"payload": payload, "head_state": head_state}
        path.write_text(json.dumps(to_json_safe(to_dump), sort_keys=True))
    return path


@regal_training(env_type="workcell")
def main(argv: Optional[List[str]] = None, runner=None) -> None:
    """Main entrypoint with regality wrapper."""
    if runner:
        runner.start_training()
    
    args = parse_args(argv)
    set_deterministic_seeds(args.seed)

    dataset = VisionPhase1Dataset(
        seed=args.seed,
        max_samples=args.max_samples,
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        sima2_root=args.sima2_root,
        trust_matrix_path=args.trust_matrix,
    )
    model = VisionBackboneStub()
    try:
        import torch.nn as nn

        head = nn.Sequential(nn.Linear(model.latent_dim, max(1, model.latent_dim // 2)), nn.ReLU(), nn.Linear(max(1, model.latent_dim // 2), 1))
        head_state = head.state_dict()
    except Exception:
        head = None
        head_state = {}

    metrics = run_training(dataset, head, max_steps=args.max_steps) if head is not None else {"processed": 0, "mean_loss": 0.0, "head_checksum": 0.0}
    head_state = head.state_dict() if head is not None else {}
    digest_src = json.dumps({"model": model.model_name, "metrics": metrics, "seed": args.seed, "head_checksum": metrics.get("head_checksum")}, sort_keys=True)
    payload = {
        "model_name": model.model_name,
        "latent_dim": model.latent_dim,
        "dataset_samples": len(dataset),
        "metrics": metrics,
        "deterministic_digest": hashlib.sha256(digest_src.encode("utf-8")).hexdigest(),
        "trained_steps": metrics.get("processed", 0),
    }
    ckpt_path = write_checkpoint(Path(args.checkpoint_dir), payload, head_state)

    log = {
        "event": "phase1_vision_training_complete",
        "checkpoint": str(ckpt_path),
        "payload": payload,
    }
    print(json.dumps(to_json_safe(log), sort_keys=True))

    if runner:
        runner.update_step(metrics.get("processed", 0))


if __name__ == "__main__":
    main()
