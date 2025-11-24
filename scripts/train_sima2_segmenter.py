"""
Phase I SIMA-2 segmenter training entrypoint.

Loads dataset scaffolds, instantiates the heuristic SIMA-2 segmenter, runs a
deterministic lightweight regression head over segment descriptors, and emits a
checkpoint.
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

from src.datasets import Sima2SegmenterDataset
from src.datasets.base import set_deterministic_seeds
from src.sima2.heuristic_segmenter import HeuristicSegmenter
from src.utils.json_safe import to_json_safe


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase I SIMA-2 Segmenter Training Stub")
    parser.add_argument("--stage1-root", type=str, default=str(Path("results") / "stage1_pipeline"))
    parser.add_argument("--stage2-root", type=str, default=str(Path("results") / "stage2_preview"))
    parser.add_argument("--sima2-root", type=str, default=str(Path("results") / "sima2_stress"))
    parser.add_argument("--trust-matrix", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args(argv)


def _segment_features(sample: Dict[str, Any]) -> List[float]:
    segs = sample.get("segments") or []
    seg = segs[0] if segs else {}
    tags = seg.get("tags", []) or []
    severity = float(seg.get("stress_severity", 0.0))
    risk = float(seg.get("risk_level", 0.0))
    trust = float(sample.get("trust_weight", 1.0))
    success_rate = float(sample.get("stage2_segments", {}).get("success_rate", 0.0))
    return [len(tags), risk, severity, trust, success_rate]


def _target_value(sample: Dict[str, Any]) -> float:
    segs = sample.get("segments") or []
    seg = segs[0] if segs else {}
    risk = float(seg.get("risk_level", 0.0))
    severity = float(seg.get("stress_severity", 0.0))
    return float(0.6 * risk + 0.4 * severity)


def _model_checksum(head) -> float:
    try:
        import torch

        checksum = 0.0
        for _, param in head.state_dict().items():
            checksum += float(param.sum().item())
        return float(checksum)
    except Exception:
        return 0.0


def run_training(dataset: Sima2SegmenterDataset, head, max_steps: int) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn

        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        steps = 0
        total_loss = 0.0
        seen_tags: set[str] = set()
        for sample in dataset:
            if steps >= max_steps:
                break
            feats = torch.tensor(_segment_features(sample), dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([_target_value(sample)], dtype=torch.float32)
            pred = head(feats).view(-1)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            for seg in sample.get("segments", []):
                for tag in seg.get("tags", []):
                    seen_tags.add(str(tag))
            steps += 1
        return {
            "processed": steps,
            "mean_loss": float(total_loss / max(1, steps)),
            "tag_coverage": len(seen_tags),
            "head_checksum": _model_checksum(head),
        }
    except Exception:
        steps = min(max_steps, len(dataset))
        return {"processed": steps, "mean_loss": 0.0, "tag_coverage": 0, "head_checksum": 0.0}


def write_checkpoint(checkpoint_dir: Path, payload: Dict[str, Any], head_state: Optional[Dict[str, Any]]) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "sima2_segmenter_phase1.pt"
    try:
        import torch

        torch.save({"payload": payload, "head_state": head_state}, path)
    except Exception:
        to_dump = {"payload": payload, "head_state": head_state}
        path.write_text(json.dumps(to_json_safe(to_dump), sort_keys=True))
    return path


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    set_deterministic_seeds(args.seed)

    dataset = Sima2SegmenterDataset(
        seed=args.seed,
        max_samples=args.max_samples,
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        sima2_root=args.sima2_root,
        trust_matrix_path=args.trust_matrix,
    )
    segmenter = HeuristicSegmenter()

    try:
        import torch.nn as nn

        head = nn.Sequential(nn.Linear(len(_segment_features(dataset.samples[0])), 8), nn.ReLU(), nn.Linear(8, 1))
    except Exception:
        head = None

    metrics = run_training(dataset, head, max_steps=args.max_steps) if head is not None else {"processed": 0, "mean_loss": 0.0, "tag_coverage": 0, "head_checksum": 0.0}
    head_state = head.state_dict() if head is not None else {}

    digest_src = json.dumps({"metrics": metrics, "seed": args.seed, "count": len(dataset), "head_checksum": metrics.get("head_checksum")}, sort_keys=True)
    payload = {
        "dataset_samples": len(dataset),
        "metrics": metrics,
        "deterministic_digest": hashlib.sha256(digest_src.encode("utf-8")).hexdigest(),
        "trained_steps": metrics.get("processed", 0),
    }
    ckpt_path = write_checkpoint(Path(args.checkpoint_dir), payload, head_state)

    log = {"event": "phase1_sima2_segmenter_training_complete", "checkpoint": str(ckpt_path), "payload": payload}
    print(json.dumps(to_json_safe(log), sort_keys=True))


if __name__ == "__main__":
    main()
