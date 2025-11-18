#!/usr/bin/env python3
"""
Build a vision frame/latent dataset from ontology episodes.
"""
import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.vision.dataset_builder import build_frame_dataset_from_ontology


def main():
    parser = argparse.ArgumentParser(description="Build vision dataset from ontology.")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = build_frame_dataset_from_ontology(
        ontology_root=args.ontology_root,
        task_id=args.task_id,
        output_dir=str(out_dir),
        max_frames=args.max_frames,
        stride=args.stride,
    )
    print(f"[build_vision_dataset_from_ontology] Wrote {stats.get('frames',0)} frames to {out_dir}")


if __name__ == "__main__":
    main()
