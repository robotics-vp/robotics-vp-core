#!/usr/bin/env python3
"""
Inspect Phase C datapacks through valuation stack (w_econ + SyntheticWeightController).

Trust is mocked to stored value (default 1.0) for now.
"""
import argparse
import json
import numpy as np
import torch

from src.config.internal_profile import get_internal_experiment_profile
from src.controllers.synthetic_weight_controller import SyntheticWeightController
from src.valuation.w_econ_lattice import WEconLattice
from src.controllers.synth_lambda_controller import build_feature_vector  # noqa: F401 (unused but kept for context)


def load_w_econ(profile, device):
    path = profile.get("w_econ_lattice_path", "")
    if not path or not path.endswith(".pt"):
        return None
    try:
        checkpoint = torch.load(path, weights_only=False, map_location=device)
    except FileNotFoundError:
        return None

    model = WEconLattice(
        n_keypoints=checkpoint.get('n_keypoints', 16),
        n_bricks=checkpoint.get('n_bricks', 5),
        hidden_dim=checkpoint.get('hidden_dim', 32),
        objective_dim=checkpoint.get('objective_dim', 4)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Inspect Phase C datapacks through valuation stack")
    parser.add_argument("datapack", type=str, help="Path to datapack JSON/JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of datapacks to load")
    args = parser.parse_args()

    # Support JSON array or JSONL
    with open(args.datapack, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            datapacks = json.load(f)
        else:
            datapacks = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        datapacks = datapacks[:args.limit]

    if not datapacks:
        print("No datapacks found.")
        return

    profile = get_internal_experiment_profile("default")
    device = torch.device("cpu")
    w_econ = load_w_econ(profile, device)

    trust = np.array([p.get("attribution", {}).get("trust", 1.0) or 1.0 for p in datapacks], dtype=np.float32)
    delta_mpl = np.array([p.get("attribution", {}).get("delta_mpl", p["episode_metrics"].get("mpl_episode", 0.0)) for p in datapacks], dtype=np.float32)
    delta_error = np.array([p.get("attribution", {}).get("delta_error", p["episode_metrics"].get("error_rate_episode", 0.0)) for p in datapacks], dtype=np.float32)
    delta_ep = np.array([p.get("attribution", {}).get("delta_ep", p["episode_metrics"].get("ep_episode", 0.0)) for p in datapacks], dtype=np.float32)
    novelty = np.array([p.get("attribution", {}).get("novelty", 0.0) or 0.0 for p in datapacks], dtype=np.float32)
    brick_ids = np.array([p.get("brick_id", 0) or 0 for p in datapacks], dtype=np.int64)

    if w_econ is not None:
        with torch.no_grad():
            econ_weights = w_econ(
                torch.from_numpy(delta_mpl),
                torch.from_numpy(delta_error),
                torch.from_numpy(delta_ep),
                torch.from_numpy(novelty),
                torch.from_numpy(brick_ids),
                objective_vector=None
            ).cpu().numpy()
    else:
        econ_weights = np.ones_like(trust)

    controller = SyntheticWeightController(
        max_synth_share=profile.get("max_synth_share", 0.4),
        econ_weight_cap=profile.get("econ_weight_cap", 1.0),
        trust_floor=profile.get("min_trust_threshold", 0.0),
        default_lambda=profile.get("target_synth_share", 0.2),
    )

    res = controller.compute_weights(
        trust=trust,
        econ=econ_weights,
        n_real=len(datapacks),  # mock balance vs. real; adjust as needed
        mode="trust_econ_lambda",
        lambda_target=profile.get("target_synth_share", 0.2),
    )

    weights = res["weights"]
    print("=== Datapack Valuation ===")
    print(f"Count: {len(datapacks)}")
    print(f"Mean w_econ: {np.mean(econ_weights):.4f}")
    print(f"Mean trust: {np.mean(trust):.4f}")
    print(f"Mean weight: {np.mean(weights):.4f}")
    print(f"Effective synth share (approx): {res['debug']['effective_synth_share']:.4f}")

    # Energy stats
    energy = [p["episode_metrics"].get("energy_Wh", 0.0) for p in datapacks]
    print(f"Energy Wh mean: {np.mean(energy):.4f}")
    # Limb fractions
    limb_totals = {"shoulder": [], "elbow": [], "wrist": [], "gripper": []}
    for p in datapacks:
        limb = p.get("energy", {}).get("limb_energy_Wh", {}) or {}
        total = sum(limb.values()) if limb else 0.0
        for k in limb_totals.keys():
            val = limb.get(k, 0.0)
            limb_totals[k].append(val / total if total > 0 else 0.0)
    for k, vals in limb_totals.items():
        if vals:
            print(f"Limb {k} energy fraction mean: {np.mean(vals):.4f}")

    # Semantic drivers
    driver_counts = {}
    for p in datapacks:
        for d in p.get("semantic_energy_drivers", []):
            driver_counts[d] = driver_counts.get(d, 0) + 1
    if driver_counts:
        print("Semantic energy drivers frequency:")
        for k, v in driver_counts.items():
            print(f"  {k}: {v}")

    # Group by env + tags
    from collections import defaultdict
    groups = defaultdict(list)
    for p, w in zip(datapacks, weights):
        env = p.get("env_type", "unknown")
        tags = tuple(sorted(p.get("semantic_energy_drivers", [])))
        groups[(env, tags)].append((p, w))

    if groups:
        print("\nGrouped summaries (env + tags):")
        for (env, tags), items in groups.items():
            mpl = np.mean([i[0]["episode_metrics"].get("mpl_episode", 0.0) for i in items])
            err = np.mean([i[0]["episode_metrics"].get("error_rate_episode", 0.0) for i in items])
            enr = np.mean([i[0]["episode_metrics"].get("energy_Wh", 0.0) for i in items])
            per_unit = np.mean([i[0]["episode_metrics"].get("energy_Wh_per_unit", 0.0) for i in items])
            limb_frac = {}
            for limb in ["shoulder", "elbow", "wrist", "gripper"]:
                vals = []
                for i in items:
                    limb_energy = i[0].get("energy", {}).get("limb_energy_Wh", {}) or i[0]["episode_metrics"].get("limb_energy_Wh", {})
                    tot = sum(limb_energy.values()) if limb_energy else 0.0
                    vals.append((limb_energy.get(limb, 0.0) / tot) if tot > 0 else 0.0)
                limb_frac[limb] = np.mean(vals) if vals else 0.0
            print(f"- Env={env}, tags={list(tags)}, count={len(items)} "
                  f"MPL={mpl:.3f} err={err:.3f} energy_Wh={enr:.3f} Wh/unit={per_unit:.3f} "
                  f"limb_frac={{{', '.join([f'{k}:{v:.2f}' for k,v in limb_frac.items()])}}}")


if __name__ == "__main__":
    main()
