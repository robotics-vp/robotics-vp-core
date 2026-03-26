"""Host-readiness helpers for real SAM3D grounded-data lanes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.evidence.preconditions import ExecutionPreconditionsReport, build_execution_preconditions


REPO_ROOT = Path(__file__).resolve().parents[2]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def collect_grounded_data_host_capabilities(
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    capabilities = {
        "gpu_available": _torch_cuda_available(),
        "cuda_available": _torch_cuda_available(),
        "torch_available": _has_module("torch"),
        "opencv_available": _has_module("cv2"),
        "sam3d_objects_repo_available": (root / "third_party" / "sam3d_objects").exists(),
        "sam3d_body_repo_available": (root / "third_party" / "sam3d_body").exists(),
        "sam3d_objects_checkpoint_available": (
            root / "checkpoints" / "sam3d_objects" / "checkpoint.pth"
        ).exists(),
        "sam3d_body_checkpoint_available": (
            root / "checkpoints" / "sam3d_body" / "checkpoint.pth"
        ).exists(),
    }
    capabilities["real_sam3d_grounding_ready"] = all(
        bool(capabilities.get(key, False))
        for key in (
            "gpu_available",
            "opencv_available",
            "sam3d_objects_repo_available",
            "sam3d_body_repo_available",
            "sam3d_objects_checkpoint_available",
            "sam3d_body_checkpoint_available",
        )
    )
    return capabilities


def build_grounded_data_host_report(
    *,
    subject_id: str,
    subject_kind: str,
    host_capabilities: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> ExecutionPreconditionsReport:
    capabilities = dict(
        host_capabilities or collect_grounded_data_host_capabilities(repo_root=repo_root)
    )
    return build_execution_preconditions(
        subject_id=subject_id,
        subject_kind=subject_kind,
        signal_values=capabilities,
        required_boolean_signals={
            "gpu_available": True,
            "opencv_available": True,
            "sam3d_objects_repo_available": True,
            "sam3d_body_repo_available": True,
            "sam3d_objects_checkpoint_available": True,
            "sam3d_body_checkpoint_available": True,
            "real_sam3d_grounding_ready": True,
        },
        metadata={"requires_real_sam3d": True},
    )


__all__ = [
    "build_grounded_data_host_report",
    "collect_grounded_data_host_capabilities",
]
