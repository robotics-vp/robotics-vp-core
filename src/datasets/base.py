"""
Common helpers for Phase I dataset loaders.

The goal is to surface consistent Stage 5-ready samples by stitching together:
- Stage 1 raw frames (task metadata + deterministic frame placeholders)
- Stage 2 segmented trajectories
- SIMA-2 stress artifacts
- IsaacAdapter rollouts
- ROS → Stage2 pipeline outputs (stubbed deterministically if missing)

Sampling uses TrustMatrix weights with deterministic shuffling keyed by seed.
"""
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analytics.econ_correlator import load_trust_matrix
from src.utils.json_safe import to_json_safe

DEFAULT_STAGE1_ROOT = Path("results") / "stage1_pipeline"
DEFAULT_STAGE2_ROOT = Path("results") / "stage2_preview"
DEFAULT_SIMA2_ROOT = Path("results") / "sima2_stress"
DEFAULT_ROS_STAGE2_PATH = Path("results") / "phase1" / "ros_to_stage2_stub.jsonl"
DEFAULT_TRUST_MATRIX_PATH = None  # use analytics default


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional for these loaders; ignore if unavailable.
        pass


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


class Phase1DatasetBase:
    """
    Deterministic sampler over Stage 1/2/SIMA-2/Isaac/ROS artifacts.
    """

    name: str = "phase1"

    def __init__(
        self,
        *,
        seed: int = 0,
        max_samples: Optional[int] = None,
        stage1_root: Optional[str] = None,
        stage2_root: Optional[str] = None,
        sima2_root: Optional[str] = None,
        isaac_rollouts_path: Optional[str] = None,
        ros_stage2_path: Optional[str] = None,
        trust_matrix_path: Optional[str] = DEFAULT_TRUST_MATRIX_PATH,
        shuffle: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        set_deterministic_seeds(self.seed)

        self.stage1_root = Path(stage1_root) if stage1_root else DEFAULT_STAGE1_ROOT
        self.stage2_root = Path(stage2_root) if stage2_root else DEFAULT_STAGE2_ROOT
        self.sima2_root = Path(sima2_root) if sima2_root else DEFAULT_SIMA2_ROOT
        self.isaac_rollouts_path = Path(isaac_rollouts_path) if isaac_rollouts_path else Path("data") / "physics_zv_rollouts.npz"
        self.ros_stage2_path = Path(ros_stage2_path) if ros_stage2_path else DEFAULT_ROS_STAGE2_PATH

        self.trust_matrix = load_trust_matrix(trust_matrix_path)

        self.stage1_frames = self._load_stage1_raw_frames()
        self.stage2_segments = self._load_stage2_segments()
        self.sima2_stress = self._load_sima2_stress_outputs()
        self.isaac_rollouts = self._load_isaac_rollouts()
        self.ros_outputs = self._load_ros_stage2_outputs()

        base_samples = self._assemble_base_samples()
        weighted = self._apply_trust_weighting(base_samples)
        if self.shuffle:
            self._shuffle(weighted)
        if max_samples is not None:
            weighted = weighted[: int(max_samples)]
        self.samples = weighted

    # --- public API ----------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def __iter__(self):
        for sample in self.samples:
            yield sample

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": len(self),
            "stage1_root": str(self.stage1_root),
            "stage2_root": str(self.stage2_root),
            "sima2_root": str(self.sima2_root),
            "isaac_rollouts_path": str(self.isaac_rollouts_path),
            "ros_stage2_path": str(self.ros_stage2_path),
        }

    # --- loaders -------------------------------------------------------- #
    def _load_stage1_raw_frames(self) -> List[Dict[str, Any]]:
        datapacks_path = self.stage1_root / "datapacks.json"
        frames: List[Dict[str, Any]] = []
        if datapacks_path.exists():
            try:
                packs = json.loads(datapacks_path.read_text())
            except Exception:
                packs = []
            for idx, pack in enumerate(sorted(packs, key=lambda p: str(p.get("pack_id", "")))):
                frames.append(
                    {
                        "pack_id": str(pack.get("pack_id", idx)),
                        "task": str(pack.get("task_name", "")),
                        "bucket": str(pack.get("bucket", "unknown")),
                        "tags": list(pack.get("semantic_tags") or []),
                        "frame_path": str(self.stage1_root / "frames" / f"{pack.get('pack_id', idx)}.npy"),
                        "stage": "stage1_raw",
                    }
                )
        if not frames:
            frames.append(self._build_synthetic_frame("stage1_fallback", 0))
        return frames

    def _load_stage2_segments(self) -> List[Dict[str, Any]]:
        primitives_path = self.stage2_root / "stage2_primitives.json"
        segments: List[Dict[str, Any]] = []
        if primitives_path.exists():
            try:
                payload = json.loads(primitives_path.read_text())
            except Exception:
                payload = []
            for idx, item in enumerate(sorted(payload, key=lambda v: str(v.get("primitive_id", "")))):
                segments.append(
                    {
                        "segment_id": str(item.get("primitive_id", idx)),
                        "tags": list(item.get("tags") or []),
                        "risk_level": _to_float(item.get("risk_level", 0.0)),
                        "energy_intensity": _to_float(item.get("energy_intensity", 0.0)),
                        "success_rate": _to_float(item.get("success_rate", 0.0)),
                        "source": str(item.get("source", "stage2")),
                        "stage": "stage2_segments",
                    }
                )
        if not segments:
            segments.append(
                {
                    "segment_id": "stage2_fallback",
                    "tags": ["default"],
                    "risk_level": 0.1,
                    "energy_intensity": 0.1,
                    "success_rate": 1.0,
                    "source": "stage2_stub",
                    "stage": "stage2_segments",
                }
            )
        return segments

    def _load_sima2_stress_outputs(self) -> List[Dict[str, Any]]:
        stress_dir = self.sima2_root
        candidates = [
            stress_dir / "primitives.jsonl",
            stress_dir / "semantic_tags.jsonl",
            stress_dir / "task_graph_proposals.jsonl",
        ]
        entries: List[Dict[str, Any]] = []
        for path in candidates:
            entries.extend(_read_jsonl(path, limit=64))
        if not entries and stress_dir.name != "sima2_stress_smoke":
            smoke_dir = stress_dir.parent / "sima2_stress_smoke"
            for path in [
                smoke_dir / "primitives.jsonl",
                smoke_dir / "semantic_tags.jsonl",
            ]:
                entries.extend(_read_jsonl(path, limit=32))

        stress: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            tag = entry.get("tag") or entry.get("label") or entry.get("primitive_id") or f"stress_{idx}"
            stress.append(
                {
                    "id": str(entry.get("primitive_id", tag)),
                    "tag": str(tag),
                    "severity": _to_float(entry.get("severity", entry.get("ood_score", 0.0) or 0.0)),
                    "metadata": entry.get("metadata") or {},
                    "stage": "sima2_stress",
                }
            )
        if not stress:
            stress.append({"id": "sima2_fallback", "tag": "sima2_default", "severity": 0.0, "metadata": {}, "stage": "sima2_stress"})
        return stress

    def _load_isaac_rollouts(self) -> List[Dict[str, Any]]:
        rollouts: List[Dict[str, Any]] = []
        candidate_paths: List[Path] = [
            Path("data") / "physics_zv_rollouts_trust.npz",
            Path("data") / "physics_zv_rollouts.npz",
            Path("data") / "synthetic_zv_rollouts.npz",
            self.isaac_rollouts_path,
        ]
        chosen: Optional[Path] = None
        for path in candidate_paths:
            if path.exists():
                chosen = path
                break
        if chosen and chosen.exists():
            try:
                with np.load(chosen, allow_pickle=True, mmap_mode="r") as data:
                    for idx, key in enumerate(sorted(data.files)):
                        arr = data[key]
                        rollouts.append(
                            {
                                "rollout_key": str(key),
                                "shape": list(arr.shape) if hasattr(arr, "shape") else [],
                                "source": "isaac_adapter",
                                "path": str(chosen),
                                "stage": "isaac_rollout",
                            }
                        )
                        if idx >= 15:
                            break
            except Exception:
                pass
        if not rollouts:
            rollouts.append({"rollout_key": "isaac_stub", "shape": [1, 1], "source": "isaac_adapter_stub", "path": "none", "stage": "isaac_rollout"})
        return rollouts

    def _load_ros_stage2_outputs(self) -> List[Dict[str, Any]]:
        chosen_path: Optional[Path] = None
        for cand in self._ros_candidate_paths():
            if cand.exists():
                chosen_path = cand
                break

        if chosen_path is None:
            chosen_path = self.ros_stage2_path
            if not chosen_path.exists():
                self._materialize_ros_stub(chosen_path)

        rows = _read_jsonl(chosen_path, limit=64)
        outputs: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            outputs.append(
                {
                    "id": str(row.get("id", f"ros_stage2_{idx}")),
                    "stage2_tags": list(row.get("tags") or row.get("stage2_tags") or []),
                    "task": str(row.get("task", row.get("task_id", "unknown"))),
                    "source": str(row.get("source", "ros_to_stage2")),
                    "stage": "ros_stage2",
                }
            )
        if not outputs:
            if chosen_path != self.ros_stage2_path:
                # Fallback to stub if preferred path has no records
                if not self.ros_stage2_path.exists():
                    self._materialize_ros_stub(self.ros_stage2_path)
                return self._load_ros_stage2_outputs()
            outputs.append({"id": "ros_stage2_stub", "stage2_tags": ["bridge"], "task": "unknown", "source": "ros_to_stage2_stub", "stage": "ros_stage2"})
        return outputs

    # --- sampling ------------------------------------------------------- #
    def _assemble_base_samples(self) -> List[Dict[str, Any]]:
        max_count = max(len(self.stage1_frames), len(self.stage2_segments), len(self.sima2_stress), len(self.isaac_rollouts), len(self.ros_outputs))
        samples: List[Dict[str, Any]] = []
        for idx in range(max_count):
            sample = {
                "sample_id": f"{self.name}_{idx}",
                "stage1_frame": self.stage1_frames[idx % len(self.stage1_frames)],
                "stage2_segments": self.stage2_segments[idx % len(self.stage2_segments)],
                "sima2_stress": self.sima2_stress[idx % len(self.sima2_stress)],
                "isaac_rollout": self.isaac_rollouts[idx % len(self.isaac_rollouts)],
                "ros_stage2": self.ros_outputs[idx % len(self.ros_outputs)],
            }
            sample = self._augment_sample(sample, idx)
            sample["trust_tag"] = sample.get("trust_tag") or self._resolve_trust_tag(sample)
            sample["trust_weight"] = float(sample.get("trust_weight", self._resolve_trust_weight(sample["trust_tag"])))
            samples.append(sample)
        return samples

    def _augment_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Hook for subclasses to attach task-specific payloads."""
        return sample

    def _apply_trust_weighting(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        weighted: List[Dict[str, Any]] = []
        for sample in samples:
            repeats = int(round(float(sample.get("trust_weight", 1.0)) * 5))
            repeats = min(10, max(1, repeats))
            for r in range(repeats):
                weighted.append({**sample, "weight_slot": r})
        return weighted

    def _shuffle(self, samples: List[Dict[str, Any]]) -> None:
        rng = random.Random(self.seed)
        rng.shuffle(samples)

    # --- helpers -------------------------------------------------------- #
    def _build_synthetic_frame(self, pack_id: str, idx: int) -> Dict[str, Any]:
        digest = hashlib.sha256(f"{pack_id}:{idx}:{self.seed}".encode("utf-8")).hexdigest()
        return {
            "pack_id": f"{pack_id}_{idx}",
            "task": "synthetic_task",
            "bucket": "synthetic",
            "tags": ["synthetic", "deterministic"],
            "frame_path": f"synthetic://frame/{digest[:8]}",
            "stage": "stage1_raw",
        }

    def _resolve_trust_tag(self, sample: Dict[str, Any]) -> str:
        for candidate in (
            sample.get("sima2_stress", {}).get("tag"),
            sample.get("stage2_segments", {}).get("tags", []),
            sample.get("stage1_frame", {}).get("tags", []),
            sample.get("ros_stage2", {}).get("stage2_tags", []),
        ):
            if isinstance(candidate, str) and candidate:
                return candidate
            if isinstance(candidate, Sequence):
                for val in candidate:
                    if val:
                        return str(val)
        return "default"

    def _resolve_trust_weight(self, tag: str) -> float:
        if not tag:
            return 1.0
        entry = self.trust_matrix.get(tag) if isinstance(self.trust_matrix, dict) else None
        if not entry and isinstance(tag, str):
            entry = self.trust_matrix.get(tag.lower(), {}) if isinstance(self.trust_matrix, dict) else None
        if entry:
            try:
                return float(entry.get("trust_score", entry.get("trust_level", 1.0)))
            except Exception:
                return 1.0
        return 1.0

    def _materialize_ros_stub(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        seeds = [s.get("pack_id", f"seed_{idx}") for idx, s in enumerate(self.stage1_frames[:3])]
        rows = []
        for idx, pack_id in enumerate(sorted(seeds)):
            rows.append(
                {
                    "id": f"ros_stub_{idx}",
                    "task": f"ros_task_{idx}",
                    "tags": ["ros_bridge", "stage2_alignment"],
                    "pack_id": pack_id,
                    "source": "ros_bridge_stub",
                }
            )
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True))
                f.write("\n")

    def _ros_candidate_paths(self) -> List[Path]:
        candidates: List[Path] = []
        if self.ros_stage2_path:
            candidates.append(self.ros_stage2_path)
        default_root = Path("results")
        candidates.extend(
            [
                default_root / "ros_to_stage2.jsonl",
                default_root / "ros_stage2.jsonl",
                default_root / "phase1" / "ros_to_stage2.jsonl",
                default_root / "phase1" / "ros_stage2.jsonl",
                default_root / "ros_to_stage2" / "outputs.jsonl",
                default_root / "ros_to_stage2" / "ros_to_stage2.jsonl",
                default_root / "ros_to_stage2" / "stage2_outputs.jsonl",
            ]
        )
        deduped: List[Path] = []
        seen = set()
        for cand in candidates:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cand)
        return deduped

    def to_json_safe(self) -> List[Dict[str, Any]]:
        return [to_json_safe(s) for s in self.samples]
