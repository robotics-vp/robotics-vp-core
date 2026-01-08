"""Environment sanity checks for local development."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.epiplexity.tracker import EpiplexityTracker, EpiplexityRunKey, ComputeBudget
from src.representation.channel_groups import load_channel_groups
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderConfig
from src.representation.channel_set_pipeline import ChannelSetPipeline, ChannelSetPipelineConfig
from src.representation.token_providers import RGBVisionTokenProvider, EmbodimentTokenProvider, SceneGraphTokenProvider
from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, NodeType, ObjectClass
from src.utils.determinism import maybe_enable_determinism_from_env


def _print_env_info() -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.getenv("VIRTUAL_ENV"))
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    print("Venv:", "yes" if in_venv else "no")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch device:", device)


def _check_imports() -> None:
    try:
        import src  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"Failed to import src package: {exc}") from exc
    print("Import:", "src OK")


def _check_determinism_flag() -> None:
    flag = os.getenv("VPE_DETERMINISTIC", "")
    seed = os.getenv("VPE_DETERMINISTIC_SEED", "")
    strict = os.getenv("VPE_DETERMINISTIC_STRICT", "")
    print(f"VPE_DETERMINISTIC={flag or '0'}")
    if flag:
        print(f"Determinism: enabled (seed={seed or '0'}, strict={strict or '0'})")
    else:
        print("Determinism: disabled (set VPE_DETERMINISTIC=1 to enable)")


def _check_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = cache_dir / "_write_test.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    print("Cache dir:", f"{cache_dir} writable")


def _smoke_channel_set() -> None:
    spec = load_channel_groups("configs/channel_groups_robotics.json")
    rng = np.random.default_rng(0)
    T = 3

    rgb_frames = rng.integers(0, 255, size=(T, 32, 32, 3), dtype=np.uint8)
    nodes = [SceneNode(id=0, polyline=np.array([[0.0, 0.0], [1.0, 0.0]]), node_type=NodeType.UNKNOWN)]
    objects = [SceneObject(id=0, class_id=ObjectClass.UNKNOWN, x=0.2, y=0.1, z=0.0)]
    graphs: List[SceneGraph] = [SceneGraph(nodes=nodes, edges=[], objects=objects, metadata={"t": t}) for t in range(T)]

    episode: Dict[str, Any] = {
        "rgb_frames": rgb_frames,
        "scene_graphs": graphs,
    }

    encoder = ChannelSetEncoder(
        channel_names=list(spec.channels.keys()),
        config=ChannelSetEncoderConfig(d_model=32, num_heads=4, dropout=0.0, pma_k=1),
    )
    encoder.eval()

    pipeline = ChannelSetPipeline(
        channel_spec=spec,
        providers=[
            RGBVisionTokenProvider(seed=0, allow_synthetic=True, token_dim=32),
            EmbodimentTokenProvider(allow_synthetic=True),
            SceneGraphTokenProvider(hidden_dim=32, num_layers=2, num_heads=4),
        ],
        encoder=encoder,
        config=ChannelSetPipelineConfig(use_channel_set_encoder=True, use_loo_cl_pretrain=True, target_len=T),
    )

    output = pipeline.encode(episode, mode="eval")
    if output.canonical_tokens is None:
        raise RuntimeError("Canonical tokens missing")

    print("Channel-set smoke:", f"canonical={list(output.canonical_tokens.shape)}")


def _smoke_epiplexity(cache_dir: Path) -> None:
    tokens = torch.randn(1, 4, 8)
    budget = ComputeBudget(max_steps=2, batch_size=4)
    key = EpiplexityRunKey(
        repr_id="doctor",
        repr_version_hash="v1",
        tokenizer_version="v1",
        transform_chain_hash="v1",
        dataset_slice_id="doctor",
        probe_model_id="probe",
        compute_budget_id=budget.budget_id(),
        seed=0,
    )
    tracker = EpiplexityTracker(cache_dir=str(cache_dir))
    result = tracker.evaluate_tokens(tokens, key, budget)
    print("Epiplexity smoke:", f"S_T={result.S_T_proxy:.4f} H_T={result.H_T_proxy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo doctor checks")
    parser.add_argument("--no-smoke", action="store_true", help="Skip smoke checks")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache dir for epiplexity smoke (default: temp)")
    args = parser.parse_args()

    maybe_enable_determinism_from_env(default_seed=0)
    _print_env_info()
    _check_imports()
    _check_determinism_flag()
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        _check_cache_dir(cache_dir)
        if not args.no_smoke:
            _smoke_channel_set()
            _smoke_epiplexity(cache_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="vpe_doctor_cache_") as tmpdir:
            cache_dir = Path(tmpdir)
            _check_cache_dir(cache_dir)
            if not args.no_smoke:
                _smoke_channel_set()
                _smoke_epiplexity(cache_dir)


if __name__ == "__main__":
    main()
