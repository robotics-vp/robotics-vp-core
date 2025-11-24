"""
Pipeline configuration helpers for Stage 6+ training.

Provides access to canonical task configuration and pipeline settings.
"""
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline.yaml"


def load_pipeline_config() -> Dict[str, Any]:
    """
    Load pipeline configuration from config/pipeline.yaml.

    Returns:
        Dictionary with pipeline configuration.
        Falls back to defaults if YAML not available or file not found.
    """
    if not YAML_AVAILABLE:
        return _get_default_config()

    if not PIPELINE_CONFIG_PATH.exists():
        return _get_default_config()

    try:
        with open(PIPELINE_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config or _get_default_config()
    except Exception:
        return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """
    Get default pipeline configuration.

    Returns:
        Default configuration dictionary with canonical task and settings.
    """
    return {
        "canonical_task_id": "drawer_open",
        "phase1": {
            "sima2_stress": {
                "enabled": True,
                "output_dir": "results/sima2_phase1",
                "num_episodes": 100,
            },
            "ros_stage2": {
                "enabled": True,
                "auto_detect": True,
                "output_dir": "results/stage2_phase1",
            },
            "isaac_adapter": {
                "enabled": True,
                "num_rollouts": 1,
                "output_dir": "results/isaac_phase1",
            },
            "manifest": {
                "output_path": "results/phase1_manifest.json",
            },
        },
        "training": {
            "vision_backbone": {
                "use_neural": False,
                "checkpoint_path": "checkpoints/vision_backbone.pt",
                "freeze_after_training": True,
            },
            "spatial_rnn": {
                "use_neural": False,
                "checkpoint_path": "checkpoints/spatial_rnn.pt",
            },
            "sima2_segmenter": {
                "use_neural": False,
                "checkpoint_path": "checkpoints/sima2_segmenter.pt",
                "f1_activation_threshold": 0.85,
            },
            "hydra_policy": {
                "use_neural": False,
                "checkpoint_path": "checkpoints/hydra_policy.pt",
            },
        },
        "determinism": {
            "default_seed": 0,
            "enforce_cuda_determinism": True,
            "warn_on_nondeterminism": True,
        },
        "safety": {
            "clip_gradients": True,
            "max_gradient_norm": 1.0,
            "check_for_nans": True,
            "fail_on_inf": True,
        },
    }


def get_canonical_task() -> str:
    """
    Get the canonical task ID for Phase I training.

    This task ID is used across:
    - SIMA-2 stress generator
    - Phase I manifest builder
    - All training entrypoints (vision, spatial_rnn, segmenter, hydra_policy)

    Returns:
        Canonical task ID (default: "drawer_open")
    """
    config = load_pipeline_config()
    return config.get("canonical_task_id", "drawer_open")


def get_phase1_config() -> Dict[str, Any]:
    """
    Get Phase I data generation configuration.

    Returns:
        Phase I configuration dictionary with settings for:
        - SIMA-2 stress generation
        - ROS→Stage2 pipeline
        - Isaac adapter
        - Manifest builder
    """
    config = load_pipeline_config()
    return config.get("phase1", _get_default_config()["phase1"])


def get_training_config(component: Optional[str] = None) -> Dict[str, Any]:
    """
    Get training configuration for a specific component or all components.

    Args:
        component: Optional component name (vision_backbone, spatial_rnn,
                   sima2_segmenter, hydra_policy). If None, returns all.

    Returns:
        Training configuration dictionary.
    """
    config = load_pipeline_config()
    training_config = config.get("training", _get_default_config()["training"])

    if component is None:
        return training_config

    return training_config.get(component, {})


def is_neural_mode_enabled(component: str) -> bool:
    """
    Check if neural mode is enabled for a specific training component.

    Args:
        component: Component name (vision_backbone, spatial_rnn,
                   sima2_segmenter, hydra_policy)

    Returns:
        True if neural mode is enabled (flag is True), False otherwise.
    """
    component_config = get_training_config(component)
    return component_config.get("use_neural", False)


def get_checkpoint_path(component: str) -> Path:
    """
    Get checkpoint path for a specific training component.

    Args:
        component: Component name (vision_backbone, spatial_rnn,
                   sima2_segmenter, hydra_policy)

    Returns:
        Path to checkpoint file.
    """
    component_config = get_training_config(component)
    default_path = f"checkpoints/{component}.pt"
    path_str = component_config.get("checkpoint_path", default_path)
    return ROOT / path_str


def get_determinism_config() -> Dict[str, Any]:
    """
    Get determinism configuration.

    Returns:
        Determinism settings (seed, CUDA enforcement, warnings).
    """
    config = load_pipeline_config()
    return config.get("determinism", _get_default_config()["determinism"])


def get_safety_config() -> Dict[str, Any]:
    """
    Get safety configuration for training.

    Returns:
        Safety settings (gradient clipping, NaN/Inf checks).
    """
    config = load_pipeline_config()
    return config.get("safety", _get_default_config()["safety"])
