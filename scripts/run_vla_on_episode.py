#!/usr/bin/env python3
"""
Run OpenVLAController on a single episode JSON (EpisodeInfoSummary with media_refs).
"""
import argparse
import json
import os
from PIL import Image

from src.vla.openvla_controller import OpenVLAController, OpenVLAConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-json", type=str, required=True)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/vla_actions")
    args = parser.parse_args()

    with open(args.episode_json, "r") as f:
        episode = json.load(f)
    episode_id = episode.get("episode_id") or episode.get("episode_metrics", {}).get("episode_id", "unknown")
    media_refs = episode.get("media_refs") or episode.get("episode_metrics", {}).get("media_refs", {})
    instr = args.instruction or episode.get("instruction") or "Follow the task safely."

    img_path = None
    if isinstance(media_refs, dict):
        if "rgb_frame" in media_refs:
            img_path = media_refs["rgb_frame"]
        elif "rgb_frames" in media_refs and media_refs["rgb_frames"]:
            img_path = media_refs["rgb_frames"][0]
    if img_path is None or not os.path.exists(img_path):
        print("No media available; emitting vla_available=false.")
        os.makedirs(args.out_dir, exist_ok=True)
        out = {
            "episode_id": episode_id,
            "instruction": instr,
            "action": {"vla_available": False},
        }
        with open(os.path.join(args.out_dir, f"{episode_id}.json"), "w") as f:
            json.dump(out, f, indent=2)
        return

    img = Image.open(img_path).convert("RGB")

    controller = OpenVLAController(OpenVLAConfig())
    controller.load_model()
    action = controller.predict_action(img, instr)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "episode_id": episode_id,
        "instruction": instr,
        "action": action,
        "vla_available": action.get("vla_available", False),
    }
    with open(os.path.join(args.out_dir, f"{episode_id}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("VLA available:", action.get("vla_available", False))
    print("Action:", action)


if __name__ == "__main__":
    main()
