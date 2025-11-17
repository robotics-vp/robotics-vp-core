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
    parser.add_argument("--model-name", type=str, default="openvla/openvla-7b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--use-vision-backbone", action="store_true", help="Enable vision backbone for embeddings")
    parser.add_argument("--backbone-type", type=str, default="dummy", choices=["dummy", "dino"])
    args = parser.parse_args()

    with open(args.episode_json, "r") as f:
        episode = json.load(f)
    episode_id = episode.get("episode_id") or episode.get("episode_metrics", {}).get("episode_id", "unknown")
    media_refs = episode.get("media_refs") or episode.get("episode_metrics", {}).get("media_refs", {})
    instr = args.instruction or episode.get("instruction") or "Follow the task safely."

    img_path = None
    img_paths_list = []
    if isinstance(media_refs, dict):
        if "rgb_frame" in media_refs:
            img_path = media_refs["rgb_frame"]
            img_paths_list = [img_path]
        elif "rgb_frames" in media_refs and media_refs["rgb_frames"]:
            img_path = media_refs["rgb_frames"][0]
            img_paths_list = media_refs["rgb_frames"]

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

    # Create config with optional vision backbone
    cfg = OpenVLAConfig(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        use_vision_backbone=args.use_vision_backbone,
        vision_backbone_type=args.backbone_type,
    )
    controller = OpenVLAController(cfg)
    controller.load_model()

    # Start episode for embedding collection
    if controller.has_vision_backbone():
        controller.start_episode()

    # Process first frame for action prediction
    action = controller.predict_action(img, instr)

    # If vision backbone is enabled and multiple frames available, process all frames
    episode_embedding = None
    if controller.has_vision_backbone():
        # Load and process additional frames if available
        for frame_path in img_paths_list[1:]:  # Skip first frame (already processed)
            if os.path.exists(frame_path):
                try:
                    frame_img = Image.open(frame_path).convert("RGB")
                    # Just add to buffer for embedding (don't predict action)
                    controller._frame_buffer.append(frame_img)
                except Exception as e:
                    print(f"Warning: Failed to load frame {frame_path}: {e}")

        # Compute episode embedding
        episode_embedding = controller.end_episode()
        if episode_embedding is not None:
            print(f"Episode embedding computed: dim={len(episode_embedding)}, norm={sum(x**2 for x in episode_embedding)**0.5:.4f}")
        else:
            print("Episode embedding: None (backbone unavailable or no frames)")

    # Add semantic tags based on action magnitudes
    semantic_tags = []
    if action.get("vla_available", False):
        raw_action = action.get("raw_action", [0] * 7)

        if len(raw_action) >= 7:
            gripper_val = abs(raw_action[6])
            pos_magnitude = sum(abs(x) for x in raw_action[:3])
            rot_magnitude = sum(abs(x) for x in raw_action[3:6])

            # Grasp confidence tags
            if gripper_val > 0.5:
                semantic_tags.append("vla:grasp_confident")
            elif gripper_val > 0.1:
                semantic_tags.append("vla:grasp_tentative")
            else:
                semantic_tags.append("vla:grasp_uncertain")

            # Motion magnitude tags
            if pos_magnitude > 0.3:
                semantic_tags.append("vla:large_position_delta")
            if rot_magnitude > 0.3:
                semantic_tags.append("vla:large_rotation_delta")

            # Scene complexity (heuristic based on action variance)
            action_variance = sum((x - sum(raw_action) / len(raw_action)) ** 2 for x in raw_action)
            if action_variance > 0.5:
                semantic_tags.append("vla:scene_complex")
            elif action_variance < 0.1:
                semantic_tags.append("vla:scene_simple")

            # Coordinated motion tag
            if pos_magnitude > 0.2 and rot_magnitude > 0.2:
                semantic_tags.append("vla:coordinated_motion")

            # Low confidence / uncertain scene detection
            if pos_magnitude < 0.05 and rot_magnitude < 0.05 and gripper_val < 0.05:
                semantic_tags.append("vla:scene_confusing")

    # Generate vla_hint text
    vla_hint_text = None
    if action.get("vla_available", False):
        raw = action.get("raw_action", [0] * 7)
        if len(raw) >= 7:
            hints = []
            if raw[0] > 0.1:
                hints.append("moving forward")
            elif raw[0] < -0.1:
                hints.append("moving backward")
            if raw[2] < -0.1:
                hints.append("reaching down")
            elif raw[2] > 0.1:
                hints.append("lifting up")
            if raw[6] > 0.3:
                hints.append("closing gripper")
            elif raw[6] < -0.3:
                hints.append("opening gripper")

            if hints:
                vla_hint_text = f"robot {', '.join(hints)}"
            else:
                vla_hint_text = "robot executing subtle motion"

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "episode_id": episode_id,
        "instruction": instr,
        "action": action,
        "vla_available": action.get("vla_available", False),
        "semantic_tags": semantic_tags,
    }
    if vla_hint_text:
        out["vla_hint_text"] = vla_hint_text

    # Add episode embedding if computed (logging only, does not affect training)
    if episode_embedding is not None:
        out["episode_embedding"] = episode_embedding.tolist()
        out["backbone_type"] = args.backbone_type
        out["embedding_dim"] = len(episode_embedding)

    with open(os.path.join(args.out_dir, f"{episode_id}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("VLA available:", action.get("vla_available", False))
    print("Action:", action)
    print("Semantic tags:", semantic_tags)
    if vla_hint_text:
        print("VLA hint:", vla_hint_text)
    if episode_embedding is not None:
        print(f"Episode embedding saved (dim={len(episode_embedding)})")


if __name__ == "__main__":
    main()
