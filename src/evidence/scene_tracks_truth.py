"""Truth helpers for distinguishing real SceneTracks from fallback lanes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.evidence.provider_truth import (
    build_external_provider_truth,
    coerce_external_provider_truth,
)


_FALLBACK_BACKENDS = {"passthrough", "stub", "auto"}
_ARTIFACT_HINT_KEYS = (
    "scene_tracks_v1",
    "scene_tracks_path",
    "scene_tracks_npz",
)


def normalize_scene_tracks_backend(value: Any) -> str:
    backend = str(value or "").strip().lower()
    if backend in {"wrapper_fallback", "load_failure_stub", "import_failure_stub", "stub_requested"}:
        return "stub"
    if backend == "zero_inference_passthrough":
        return "passthrough"
    return backend


def resolve_scene_tracks_backend(
    payload: Optional[Mapping[str, Any]] = None,
    *,
    explicit_backend: Any = None,
) -> str:
    """Resolve backend identity from explicit fields or nested runner metadata."""

    if explicit_backend not in (None, ""):
        return normalize_scene_tracks_backend(explicit_backend)

    metadata = dict(payload or {})
    scene_tracks_metadata = metadata.get("scene_tracks_metadata")
    if isinstance(scene_tracks_metadata, Mapping):
        nested = dict(scene_tracks_metadata)
    else:
        nested = {}

    for candidate in (
        metadata.get("scene_tracks_backend"),
        metadata.get("backend_selected"),
        nested.get("scene_tracks_backend"),
        nested.get("backend_selected"),
    ):
        backend = normalize_scene_tracks_backend(candidate)
        if backend:
            return backend

    runner = nested.get("runner")
    if not isinstance(runner, Mapping):
        runner = metadata.get("runner")
    if isinstance(runner, Mapping):
        run_config = runner.get("run_config")
        if not isinstance(run_config, Mapping):
            run_config = {}
        backend = normalize_scene_tracks_backend(run_config.get("backend_selected"))
        if backend:
            return backend
        if run_config.get("zero_inference_passthrough") is True:
            return "passthrough"
        if run_config.get("use_stub_adapters") is True:
            return "stub"
        if run_config.get("use_stub_adapters") is False and not run_config.get("real_backend_failure"):
            return "real"

    adapter_status = nested.get("adapter_status")
    if not isinstance(adapter_status, Mapping):
        adapter_status = metadata.get("adapter_status")
    if isinstance(adapter_status, Mapping):
        backend = normalize_scene_tracks_backend(adapter_status.get("overall_mode"))
        if backend:
            return backend

    if any(metadata.get(key) for key in _ARTIFACT_HINT_KEYS):
        return "artifact_present_unknown"
    if any(nested.get(key) for key in _ARTIFACT_HINT_KEYS):
        return "artifact_present_unknown"
    return ""


def normalize_scene_tracks_truth(
    *,
    backend: Any,
    explicit_non_stub: bool = False,
    semantic_grounding_ready: bool = False,
    training_eligible: bool = False,
    explicit_non_heuristic: bool = False,
) -> Dict[str, Any]:
    backend_selected = normalize_scene_tracks_backend(backend)
    backend_real = backend_selected == "real"
    backend_passthrough = backend_selected == "passthrough"
    backend_stub = backend_selected == "stub"
    backend_auto = backend_selected == "auto"
    fallback_backend = backend_passthrough or backend_stub or backend_auto

    scene_tracks_non_stub = bool(
        backend_real or (bool(explicit_non_stub) and not fallback_backend)
    )
    semantic_grounding_non_heuristic = bool(
        (backend_real and scene_tracks_non_stub)
        or (bool(explicit_non_heuristic) and scene_tracks_non_stub and not fallback_backend)
    )

    return {
        "scene_tracks_backend": backend_selected,
        "scene_tracks_backend_real": backend_real,
        "scene_tracks_backend_passthrough": backend_passthrough,
        "scene_tracks_backend_stub": backend_stub,
        "scene_tracks_backend_auto": backend_auto,
        "scene_tracks_non_stub": scene_tracks_non_stub,
        "semantic_grounding_non_heuristic": semantic_grounding_non_heuristic,
        "semantic_grounding_ready": bool(semantic_grounding_ready) and semantic_grounding_non_heuristic,
        "scene_tracks_training_eligible": bool(training_eligible) and backend_real,
    }


def build_scene_tracks_provider_truth(
    payload: Optional[Mapping[str, Any]] = None,
    *,
    explicit_backend: Any = None,
) -> Dict[str, Any]:
    metadata = dict(payload or {})
    truth = scene_tracks_truth_from_metadata(metadata, explicit_backend=explicit_backend)
    backend = str(truth.get("scene_tracks_backend", "") or "")
    if truth.get("semantic_grounding_non_heuristic", False):
        grounding_class = "non_heuristic_grounded"
    elif truth.get("scene_tracks_non_stub", False):
        grounding_class = "real_backend_unqualified"
    elif backend == "passthrough":
        grounding_class = "passthrough"
    elif backend == "stub":
        grounding_class = "stub"
    elif backend == "artifact_present_unknown":
        grounding_class = "artifact_present_unknown"
    else:
        grounding_class = "unavailable"
    calibration_class = (
        "camera_params_present"
        if bool(metadata.get("camera_params_present", metadata.get("camera_name", "")))
        else "camera_params_missing"
    )
    return build_external_provider_truth(
        provider_id="scene_tracks",
        provider_kind="scene_tracks_runtime",
        provider_name="scene_tracks",
        advisory_only=True,
        available=bool(backend and backend not in {"", "unavailable"}),
        backend_selected=backend,
        fallback_mode="" if backend == "real" else backend,
        calibration_class=calibration_class,
        grounding_class=grounding_class,
        confidence=float(metadata.get("scene_ir_quality", 0.0) or 0.0),
        metadata={
            "scene_tracks_non_stub": bool(truth.get("scene_tracks_non_stub", False)),
            "semantic_grounding_non_heuristic": bool(truth.get("semantic_grounding_non_heuristic", False)),
            "semantic_grounding_ready": bool(truth.get("semantic_grounding_ready", False)),
            "scene_tracks_training_eligible": bool(truth.get("scene_tracks_training_eligible", False)),
        },
    )


def scene_tracks_truth_from_metadata(
    payload: Optional[Mapping[str, Any]] = None,
    *,
    explicit_backend: Any = None,
) -> Dict[str, Any]:
    """Normalize scene-track truth semantics from metadata payloads."""

    metadata = dict(payload or {})
    explicit_truth = coerce_external_provider_truth(metadata.get("scene_tracks_provider_truth"))
    if explicit_truth:
        backend = normalize_scene_tracks_backend(explicit_truth.get("backend_selected"))
        grounding_class = str(explicit_truth.get("grounding_class", "") or "")
        return normalize_scene_tracks_truth(
            backend=backend,
            explicit_non_stub=backend == "real" or grounding_class in {"non_heuristic_grounded", "real_backend_unqualified"},
            semantic_grounding_ready=grounding_class == "non_heuristic_grounded",
            training_eligible=bool(
                explicit_truth.get("metadata", {}).get("scene_tracks_training_eligible", False)
            ),
            explicit_non_heuristic=grounding_class == "non_heuristic_grounded",
        )
    future_signals = metadata.get("future_training_signals")
    if not isinstance(future_signals, Mapping):
        future_signals = {}
    backend = resolve_scene_tracks_backend(metadata, explicit_backend=explicit_backend)
    return normalize_scene_tracks_truth(
        backend=backend,
        explicit_non_stub=bool(
            future_signals.get("scene_tracks_non_stub", metadata.get("scene_tracks_non_stub", False))
        ),
        semantic_grounding_ready=bool(
            future_signals.get("semantic_grounding_ready", metadata.get("semantic_grounding_ready", False))
        ),
        training_eligible=bool(
            future_signals.get(
                "scene_tracks_training_eligible",
                metadata.get(
                    "scene_tracks_training_eligible",
                    metadata.get("training_eligible", False),
                ),
            )
        ),
        explicit_non_heuristic=bool(
            future_signals.get(
                "semantic_grounding_non_heuristic",
                metadata.get("semantic_grounding_non_heuristic", False),
            )
        ),
    )


__all__ = [
    "build_scene_tracks_provider_truth",
    "normalize_scene_tracks_backend",
    "normalize_scene_tracks_truth",
    "resolve_scene_tracks_backend",
    "scene_tracks_truth_from_metadata",
]
