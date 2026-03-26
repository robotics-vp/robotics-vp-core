"""Truth helpers for distinguishing real SceneTracks from fallback lanes."""

from __future__ import annotations

from typing import Any, Dict


def normalize_scene_tracks_truth(
    *,
    backend: Any,
    explicit_non_stub: bool = False,
    semantic_grounding_ready: bool = False,
    training_eligible: bool = False,
    explicit_non_heuristic: bool = False,
) -> Dict[str, Any]:
    backend_selected = str(backend or "")
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


__all__ = ["normalize_scene_tracks_truth"]
