"""
ROS/JSON bridge to ingest real robot logs into RawRollout objects.

Supports ROS bag exports or simple JSON logs with the following topics:
- /camera/rgb
- /camera/depth
- /joint_states
- /tf
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.ingestion.rollout_types import ActionFrame, EnvStateDigest, ProprioFrame, RawRollout
from src.utils.json_safe import to_json_safe
from src.vision.interfaces import VisionFrame, compute_state_digest


class ROSBridgeIngestor:
    """
    Convert ROS bag / JSON robot logs into RawRollout objects for SIMA-2.
    """

    def __init__(self, output_root: Optional[str] = None, run_timestamp: Optional[float] = None):
        self.output_root = Path(output_root or Path("results") / "ingestion")
        self.run_timestamp = run_timestamp if run_timestamp is not None else time.time()

    def ingest(self, log_path: str, *, task_id: str = "real_robot", backend_id: str = "ros_bridge") -> RawRollout:
        raw_messages, metadata = self._load_log(log_path)
        messages = self._normalize_messages(raw_messages)
        episode_id = self._deterministic_episode_id(log_path, messages)
        output_dir = self._output_dir()
        frame_dir = output_dir / episode_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        timeline = self._timeline(messages)
        ts_to_step = {ts: idx for idx, ts in enumerate(timeline)}

        vision_frames: List[VisionFrame] = []
        proprio_frames: List[ProprioFrame] = []
        action_frames: List[ActionFrame] = []
        env_digests: List[EnvStateDigest] = []

        for topic, ts, payload in sorted(messages, key=lambda e: (e[1], e[0])):
            timestep = ts_to_step[ts]
            if "camera" in topic and ("rgb" in topic or "image" in topic):
                frame = self._vision_frame_from_payload(
                    payload, task_id=task_id, episode_id=episode_id, timestep=timestep, backend_id=backend_id, frame_dir=frame_dir
                )
                vision_frames.append(frame)
            elif "depth" in topic:
                frame = self._vision_frame_from_payload(
                    payload,
                    task_id=task_id,
                    episode_id=episode_id,
                    timestep=timestep,
                    backend_id=backend_id,
                    frame_dir=frame_dir,
                    is_depth=True,
                )
                vision_frames.append(frame)
            elif "joint_states" in topic or "joint_state" in topic:
                proprio_frames.append(self._proprio_from_payload(payload, timestep))
            elif "action" in topic or "command" in topic:
                action_frames.append(self._action_from_payload(payload, timestep))
            elif topic == "/tf" or "tf" in topic:
                env_digests.append(self._env_state_from_tf(payload, timestep))

        # Deduplicate and sort for determinism
        vision_frames = sorted(vision_frames, key=lambda vf: (vf.timestep, vf.camera_name or "cam"))
        proprio_frames = sorted(proprio_frames, key=lambda pf: pf.timestep)
        action_frames = sorted(action_frames, key=lambda af: af.timestep)
        env_digests = sorted(env_digests, key=lambda ed: ed.timestep)

        rollout = RawRollout(
            episode_id=episode_id,
            task_id=task_id,
            vision_frames=vision_frames,
            proprio_frames=proprio_frames,
            action_frames=action_frames,
            env_digests=env_digests,
            metadata={"source": backend_id, "log_path": str(log_path), **metadata},
        )

        self._write_datapack(output_dir, rollout)
        return rollout

    # ---------- Internal helpers ----------
    def _load_log(self, log_path: str) -> Tuple[Any, Dict[str, Any]]:
        path = Path(log_path)
        if not path.exists():
            raise FileNotFoundError(f"Log not found: {log_path}")

        if path.suffix.lower() == ".json":
            with open(path, "r") as f:
                return json.load(f), {"format": "json"}

        if path.suffix.lower() == ".bag":
            # Lazy import to avoid hard dependency; users can export to JSON if unavailable.
            try:
                import rosbag  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("rosbag is required for .bag ingestion; export to JSON or install rosbag") from exc

            messages = []
            with rosbag.Bag(str(path), "r") as bag:  # pragma: no cover
                for topic, msg, t in bag.read_messages():
                    payload = {"topic": topic, "stamp": float(t.to_sec())}
                    if hasattr(msg, "data"):
                        payload["data"] = getattr(msg, "data")
                    if hasattr(msg, "position"):
                        payload["position"] = getattr(msg, "position")
                    if hasattr(msg, "velocity"):
                        payload["velocity"] = getattr(msg, "velocity")
                    if hasattr(msg, "effort"):
                        payload["effort"] = getattr(msg, "effort")
                    messages.append(payload)
            return {"messages": messages}, {"format": "rosbag"}

        raise ValueError(f"Unsupported log format for {log_path}")

    def _normalize_messages(self, raw: Any) -> List[Tuple[str, float, Dict[str, Any]]]:
        messages: List[Tuple[str, float, Dict[str, Any]]] = []
        if isinstance(raw, dict):
            if "topic_messages" in raw:
                for topic, entries in (raw.get("topic_messages") or {}).items():
                    for idx, msg in enumerate(entries or []):
                        ts = self._extract_timestamp(msg, default=idx)
                        messages.append((str(topic), ts, dict(msg)))
            if "messages" in raw:
                for idx, msg in enumerate(raw.get("messages") or []):
                    topic = str(msg.get("topic") or msg.get("name") or f"topic_{idx}")
                    ts = self._extract_timestamp(msg, default=idx)
                    payload = dict(msg)
                    payload.pop("topic", None)
                    messages.append((topic, ts, payload))
        elif isinstance(raw, Sequence):
            for idx, msg in enumerate(raw):
                topic = str(getattr(msg, "topic", None) or msg.get("topic") if isinstance(msg, dict) else f"topic_{idx}")
                ts = self._extract_timestamp(msg, default=idx)
                payload = dict(msg) if isinstance(msg, dict) else {"data": msg}
                payload.pop("topic", None)
                messages.append((topic, ts, payload))
        return messages

    def _timeline(self, messages: List[Tuple[str, float, Dict[str, Any]]]) -> List[float]:
        if not messages:
            return []
        unique = sorted({float(ts) for _, ts, _ in messages})
        return unique

    def _extract_timestamp(self, msg: Any, default: int) -> float:
        if isinstance(msg, dict):
            for key in ("stamp", "timestamp", "time", "t"):
                if key in msg:
                    try:
                        return float(msg[key])
                    except Exception:
                        continue
        try:
            return float(getattr(msg, "stamp", default))
        except Exception:
            return float(default)

    def _deterministic_episode_id(self, log_path: str, messages: List[Tuple[str, float, Dict[str, Any]]]) -> str:
        payload = {
            "log": str(Path(log_path).name),
            "count": len(messages),
            "first_ts": messages[0][1] if messages else 0.0,
            "last_ts": messages[-1][1] if messages else 0.0,
        }
        serialized = json.dumps(to_json_safe(payload), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        return f"ros_episode_{digest}"

    def _output_dir(self) -> Path:
        ts_label = time.strftime("%Y%m%dT%H%M%S", time.gmtime(self.run_timestamp))
        out = self.output_root / ts_label
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _vision_frame_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        task_id: str,
        episode_id: str,
        timestep: int,
        backend_id: str,
        frame_dir: Path,
        is_depth: bool = False,
    ) -> VisionFrame:
        data = payload.get("data") if isinstance(payload, dict) else None
        np_data = np.asarray(data) if data is not None else None
        height, width, channels = (0, 0, 1)
        if np_data is not None and np_data.size > 0:
            shape = np_data.shape
            if np_data.ndim == 2:
                height, width = shape
                channels = 1
            elif np_data.ndim == 3:
                height, width, channels = shape
        rgb_path = None
        depth_path = None
        segmentation_path = None
        suffix = "depth" if is_depth else "rgb"
        target_path = frame_dir / f"{episode_id}_{suffix}_{timestep:04d}.json"
        with open(target_path, "w") as f:
            json.dump(to_json_safe(data), f, sort_keys=True)
        if is_depth:
            depth_path = str(target_path)
        else:
            rgb_path = str(target_path)
        if payload.get("segmentation") is not None:
            seg_path = frame_dir / f"{episode_id}_seg_{timestep:04d}.json"
            with open(seg_path, "w") as f:
                json.dump(to_json_safe(payload.get("segmentation")), f, sort_keys=True)
            segmentation_path = str(seg_path)
        intrinsics = payload.get("camera_intrinsics") or {"resolution": [width, height]}
        extrinsics = payload.get("camera_extrinsics") or payload.get("camera_pose") or {}
        tf_state = payload.get("tf") or {}
        state_digest = compute_state_digest({"tf": tf_state, "timestep": timestep})

        return VisionFrame(
            backend=backend_id,
            backend_id=backend_id,
            task_id=task_id,
            episode_id=episode_id,
            timestep=timestep,
            width=int(width),
            height=int(height),
            channels=int(channels),
            camera_intrinsics=to_json_safe(intrinsics),
            camera_extrinsics=to_json_safe(extrinsics),
            camera_pose=to_json_safe(extrinsics or intrinsics),
            rgb_path=rgb_path,
            depth_path=depth_path,
            segmentation_path=segmentation_path,
            camera_name=str(payload.get("camera_name") or "ros_camera"),
            state_digest=state_digest,
            metadata={"source_topic": payload.get("topic", "/camera/rgb"), "state": tf_state},
        )

    def _proprio_from_payload(self, payload: Dict[str, Any], timestep: int) -> ProprioFrame:
        positions = payload.get("position") or payload.get("positions") or payload.get("joint_positions") or []
        velocities = payload.get("velocity") or payload.get("velocities") or payload.get("joint_velocities") or []
        efforts = payload.get("effort") or payload.get("joint_effort") or payload.get("joint_torques") or []
        contacts = payload.get("contacts") or payload.get("contact_sensors") or []
        ee_pose = payload.get("end_effector_pose") or payload.get("pose")
        energy = self._estimate_energy(efforts, velocities, payload.get("dt"))
        return ProprioFrame(
            timestep=int(timestep),
            joint_positions=positions,
            joint_velocities=velocities,
            joint_torques=efforts,
            contact_sensors=contacts,
            end_effector_pose=ee_pose,
            energy_estimate_Wh=energy,
            metadata={"source_topic": payload.get("topic", "/joint_states")},
        )

    def _action_from_payload(self, payload: Dict[str, Any], timestep: int) -> ActionFrame:
        command = payload.get("action") or payload.get("command") or {k: v for k, v in payload.items() if k not in ("stamp", "timestamp")}
        return ActionFrame(timestep=int(timestep), command=command, metadata={"source_topic": payload.get("topic", "action")})

    def _env_state_from_tf(self, payload: Dict[str, Any], timestep: int) -> EnvStateDigest:
        transforms = payload.get("transforms") or payload.get("tf") or payload.get("frames") or payload
        tf_tree = {}
        if isinstance(transforms, list):
            for tf in transforms:
                child = str(tf.get("child_frame_id") or tf.get("child") or len(tf_tree))
                tf_tree[child] = {
                    "translation": to_json_safe(tf.get("translation") or tf.get("trans")),
                    "rotation": to_json_safe(tf.get("rotation") or tf.get("rot")),
                    "timestamp": self._extract_timestamp(tf, default=timestep),
                }
        elif isinstance(transforms, dict):
            tf_tree = to_json_safe(transforms)
        return EnvStateDigest(timestep=int(timestep), tf_tree=tf_tree, metadata={"source_topic": payload.get("topic", "/tf")})

    def _estimate_energy(self, torques: Any, velocities: Any, dt: Optional[float]) -> float:
        try:
            tau = np.asarray(torques, dtype=float)
            vel = np.asarray(velocities, dtype=float)
            if tau.shape != vel.shape:
                vel = np.broadcast_to(vel, tau.shape)
            dt_val = float(dt) if dt is not None else 0.02
            power = np.abs(tau * vel)
            energy_Wh = float(np.sum(power) * dt_val / 3600.0)
            return energy_Wh
        except Exception:
            return 0.0

    def _write_datapack(self, output_dir: Path, rollout: RawRollout) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        datapack_path = output_dir / "datapacks.jsonl"
        record = rollout.to_datapack_record(source="real_robot")
        with open(datapack_path, "w") as f:
            f.write(json.dumps(to_json_safe(record), sort_keys=True))
            f.write("\n")
        return datapack_path
