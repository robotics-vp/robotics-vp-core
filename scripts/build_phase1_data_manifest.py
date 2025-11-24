"""
Build Phase I data manifest.

Enumerates Phase I datasets, captures sample counts and source paths, and writes
results/phase1/data_manifest.json in a JSON-safe, deterministic format.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets import HydraPolicyDataset, Sima2SegmenterDataset, SpatialRNNDataset, VisionPhase1Dataset
from src.datasets.base import set_deterministic_seeds
from src.utils.json_safe import to_json_safe


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase I Data Manifest")
    parser.add_argument("--stage1-root", type=str, default=str(Path("results") / "stage1_pipeline"))
    parser.add_argument("--stage2-root", type=str, default=str(Path("results") / "stage2_preview"))
    parser.add_argument("--sima2-root", type=str, default=str(Path("results") / "sima2_stress"))
    parser.add_argument("--trust-matrix", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=str(Path("results") / "phase1" / "data_manifest.json"))
    return parser.parse_args(argv)


def build_manifest(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    shared_kwargs = {
        "seed": args.seed,
        "max_samples": args.max_samples,
        "stage1_root": args.stage1_root,
        "stage2_root": args.stage2_root,
        "sima2_root": args.sima2_root,
        "trust_matrix_path": args.trust_matrix,
    }

    datasets = {
        "vision": VisionPhase1Dataset(**shared_kwargs),
        "spatial_rnn": SpatialRNNDataset(**shared_kwargs),
        "sima2_segmenter": Sima2SegmenterDataset(**shared_kwargs),
        "hydra_policy": HydraPolicyDataset(**shared_kwargs),
    }

    manifest: Dict[str, Dict[str, Any]] = {
        "phase": "phase1",
        "seed": args.seed,
        "datasets": {},
    }
    for name, ds in datasets.items():
        entry = ds.manifest()
        entry["count"] = len(ds)
        manifest["datasets"][name] = entry
    return manifest


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    set_deterministic_seeds(args.seed)

    manifest = build_manifest(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_json_safe(manifest), sort_keys=True, indent=2))

    log = {"event": "phase1_data_manifest_built", "output": str(out_path), "dataset_counts": {k: v["count"] for k, v in manifest["datasets"].items()}}
    print(json.dumps(to_json_safe(log), sort_keys=True))


if __name__ == "__main__":
    main()
