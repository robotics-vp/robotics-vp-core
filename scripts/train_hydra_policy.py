"""
Phase I Hydra policy training entrypoint.

Loads Stage 5 dataset slices, builds a minimal HydraActor skeleton, runs a
deterministic regression loop against condition features, and emits a checkpoint.
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

import torch
import torch.nn as nn

from src.datasets import HydraPolicyDataset
from src.datasets.base import set_deterministic_seeds
from src.observation.condition_vector import ConditionVector
from src.rl.hydra_heads import HydraActor
from src.utils.json_safe import to_json_safe


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase I Hydra Policy Training Stub")
    parser.add_argument("--stage1-root", type=str, default=str(Path("results") / "stage1_pipeline"))
    parser.add_argument("--stage2-root", type=str, default=str(Path("results") / "stage2_preview"))
    parser.add_argument("--sima2-root", type=str, default=str(Path("results") / "sima2_stress"))
    parser.add_argument("--trust-matrix", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args(argv)


def build_hydra_policy(skill_modes: List[str]) -> HydraActor:
    unique_modes = sorted(set(skill_modes or ["default"]))
    trunk = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    heads = {mode: nn.Linear(4, 2) for mode in unique_modes}
    return HydraActor(trunk=trunk, heads=heads, default_skill_mode=unique_modes[0])


def _condition_targets(sample: Dict[str, Any]) -> torch.Tensor:
    features = sample.get("condition_features", {}) or {}
    econ = features.get("econ_slice", {}) or {}
    ood = float(features.get("ood_severity", 0.0))
    recovery = float(features.get("recovery_priority", 0.0))
    novelty = float(features.get("novelty_tier", 0.0))
    # Targets encourage recovery/novelty awareness and OOD caution
    return torch.tensor([recovery + 0.5 * ood, novelty], dtype=torch.float32)


def _condition_obs(sample: Dict[str, Any]) -> torch.Tensor:
    features = sample.get("condition_features", {}) or {}
    econ = features.get("econ_slice", {}) or {}
    return torch.tensor(
        [
            float(econ.get("target_mpl", 0.0)),
            float(econ.get("energy_budget_wh", 0.0)),
            float(econ.get("current_wage_parity", 0.0)),
            float(features.get("ood_severity", 0.0)),
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def _model_checksum(policy: HydraActor) -> float:
    checksum = 0.0
    for _, param in policy.state_dict().items():
        checksum += float(param.sum().item())
    return float(checksum)


def run_training(dataset: HydraPolicyDataset, policy: HydraActor, max_steps: int) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    steps = 0
    total_loss = 0.0
    digests: List[float] = []
    for sample in dataset:
        if steps >= max_steps:
            break
        condition = None
        if sample.get("condition_vector"):
            condition = ConditionVector.from_dict(sample["condition_vector"])
        obs = _condition_obs(sample)
        target = _condition_targets(sample)
        pred = policy(obs, condition).view(-1)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        digests.append(float(pred.sum().item()))
        steps += 1
    return {
        "processed": steps,
        "mean_loss": float(total_loss / max(1, steps)),
        "policy_digest": float(sum(digests)),
        "policy_checksum": _model_checksum(policy),
    }


def write_checkpoint(checkpoint_dir: Path, payload: Dict[str, Any], policy_state: Dict[str, Any]) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "hydra_policy_phase1.pt"
    try:
        torch.save({"payload": payload, "policy_state": policy_state}, path)
    except Exception:
        to_dump = {"payload": payload, "policy_state": policy_state}
        path.write_text(json.dumps(to_json_safe(to_dump), sort_keys=True))
    return path


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    set_deterministic_seeds(args.seed)

    dataset = HydraPolicyDataset(
        seed=args.seed,
        max_samples=args.max_samples,
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        sima2_root=args.sima2_root,
        trust_matrix_path=args.trust_matrix,
    )

    skill_modes = [cv.get("skill_mode", "default") for cv in (s.get("condition_vector", {}) for s in dataset)]
    policy = build_hydra_policy(skill_modes)

    metrics = run_training(dataset, policy, max_steps=args.max_steps)
    digest_src = json.dumps({"metrics": metrics, "seed": args.seed, "count": len(dataset), "policy_checksum": metrics.get("policy_checksum")}, sort_keys=True)
    payload = {
        "dataset_samples": len(dataset),
        "metrics": metrics,
        "deterministic_digest": hashlib.sha256(digest_src.encode("utf-8")).hexdigest(),
        "trained_steps": metrics.get("processed", 0),
    }
    ckpt_path = write_checkpoint(Path(args.checkpoint_dir), payload, policy.state_dict())

    log = {"event": "phase1_hydra_policy_training_complete", "checkpoint": str(ckpt_path), "payload": payload}
    print(json.dumps(to_json_safe(log), sort_keys=True))


if __name__ == "__main__":
    main()
