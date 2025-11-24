"""
Phase I vision backbone dataset loader.

Wraps Stage 1/2/SIMA-2 artifacts into deterministic samples and precomputes
VisionBackboneStub latents so training scaffolds stay light-weight.
"""
import hashlib
from typing import Any, Dict

from src.datasets.base import Phase1DatasetBase, set_deterministic_seeds
from src.vision.backbone_stub import VisionBackboneStub
from src.vision.interfaces import VisionFrame


class VisionPhase1Dataset(Phase1DatasetBase):
    name = "vision_phase1"

    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        set_deterministic_seeds(seed)
        self.encoder = VisionBackboneStub()
        super().__init__(*args, seed=seed, **kwargs)

    def _augment_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        frame_meta = sample.get("stage1_frame", {})
        pack_id = frame_meta.get("pack_id", f"episode_{idx}")
        task = frame_meta.get("task", "unknown_task")
        digest = hashlib.sha256(f"{pack_id}:{idx}:{self.seed}".encode("utf-8")).hexdigest()

        frame = VisionFrame(
            backend="phase1_dataset",
            backend_id="phase1_dataset",
            task_id=task,
            episode_id=pack_id,
            timestep=idx,
            width=64,
            height=64,
            channels=3,
            dtype="uint8",
            rgb_path=frame_meta.get("frame_path"),
            camera_pose={"pose": "synthetic"},
            camera_intrinsics={"resolution": [64, 64]},
            camera_extrinsics={"frame": "world"},
            state_digest=digest,
            metadata={"tags": frame_meta.get("tags", []), "bucket": frame_meta.get("bucket")},
        )
        latent = self.encoder.encode_frame(frame)

        sample["vision_frame"] = frame.to_dict()
        sample["vision_latent"] = latent.to_dict()
        sample["latent_digest"] = digest
        return sample

