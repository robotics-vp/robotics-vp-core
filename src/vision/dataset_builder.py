"""
Dataset builder for vision frames/latents from ontology episodes.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.ontology.store import OntologyStore
from src.vision.backbone_stub import VisionBackboneStub
from src.vision.config import load_vision_config
from src.vision.interfaces import VisionFrame


def _deterministic_frame_array(width: int, height: int, channels: int, episode_id: str, timestep: int, dtype: str):
    digest = hashlib.sha256(f"{episode_id}|{timestep}".encode("utf-8")).hexdigest()
    fill_val = int(digest[:2], 16)  # 0-255
    if dtype.startswith("float"):
        val = fill_val / 255.0
        arr = np.full((height, width, channels), val, dtype=np.float32)
    else:
        arr = np.full((height, width, channels), fill_val, dtype=np.uint8)
    return arr


def build_frame_dataset_from_ontology(
    ontology_root: str,
    task_id: str,
    output_dir: str,
    max_frames: int = 100,
    stride: int = 1,
) -> Dict[str, int]:
    store = OntologyStore(root_dir=ontology_root)
    episodes = store.list_episodes(task_id=task_id)
    cfg = load_vision_config()
    width, height = cfg.get("input_resolution", [224, 224])
    channels = int(cfg.get("channels", 3))
    dtype = str(cfg.get("dtype", "uint8"))
    encoder = VisionBackboneStub(model_name=cfg.get("model_name"), latent_dim=cfg.get("latent_dim"))

    out_dir = Path(output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    count = 0
    with meta_path.open("w") as mf:
        for ep in sorted(episodes, key=lambda e: e.episode_id):
            events = sorted(store.get_events(ep.episode_id), key=lambda ev: ev.timestep)
            for ev in events[:: max(1, stride)]:
                if count >= max_frames:
                    return {"frames": count}
                frame = VisionFrame(
                    backend="dataset_builder",
                    task_id=task_id,
                    episode_id=ep.episode_id,
                    timestep=ev.timestep,
                    width=int(width),
                    height=int(height),
                    channels=channels,
                    dtype=dtype,
                    camera_pose={"pose": "synthetic"},
                    camera_intrinsics={"resolution": [width, height]},
                    rgb_path=str(frames_dir / f"frame_{count:06d}.npy"),
                    metadata={"event_digest": hashlib.sha256(json.dumps(ev.state_summary, sort_keys=True, default=str).encode("utf-8")).hexdigest()},
                )
                array = _deterministic_frame_array(int(width), int(height), channels, ep.episode_id, ev.timestep, dtype)
                np.save(frame.rgb_path, array)
                latent = encoder.encode_frame(frame)
                mf.write(
                    json.dumps(
                        {
                            "frame_path": frame.rgb_path,
                            "frame": frame.to_dict(),
                            "latent": latent.to_dict(),
                            "episode_id": ep.episode_id,
                            "timestep": ev.timestep,
                            "reward_components": ev.reward_components,
                            "state_summary": ev.state_summary,
                        },
                        sort_keys=True,
                    )
                )
                mf.write("\n")
                count += 1
    return {"frames": count}
