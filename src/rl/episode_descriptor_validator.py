"""
Episode descriptor schema validation and normalization.

Keeps the Stage 1 → RL ingestion contract predictable without touching
training or reward logic.
"""
from typing import Any, Dict, List, Tuple
import copy


DEFAULT_OBJECTIVE_VECTOR = [1.0, 1.0, 1.0, 1.0, 0.0]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _normalize_objective_vector(vector: Any, target_len: int = 5) -> List[float]:
    """Ensure objective vector is a fixed-length float list."""
    if not isinstance(vector, list):
        vector_list: List[Any] = []
    else:
        vector_list = list(vector)

    normalized = [float(v) for v in vector_list[:target_len]]
    if len(normalized) < target_len:
        normalized.extend(DEFAULT_OBJECTIVE_VECTOR[len(normalized):target_len])
    return normalized


def normalize_episode_descriptor(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a normalized descriptor copy with deterministic defaults filled in.

    Only structural fields are touched; semantic values are preserved when
    present.
    """
    normalized = copy.deepcopy(descriptor)

    normalized["objective_vector"] = _normalize_objective_vector(
        descriptor.get("objective_vector")
    )

    env_name = descriptor.get("env_name") or descriptor.get("task_type") or "unknown_env"
    normalized["env_name"] = env_name
    normalized["task_type"] = descriptor.get("task_type") or env_name
    normalized["engine_type"] = descriptor.get("engine_type") or descriptor.get("backend") or "unknown_engine"

    tier = descriptor.get("tier", 1)
    try:
        tier_int = int(tier)
    except Exception:
        tier_int = 1
    normalized["tier"] = max(0, tier_int)

    trust_score = descriptor.get("trust_score", 0.5)
    try:
        trust_score_f = float(trust_score)
    except Exception:
        trust_score_f = 0.5
    normalized["trust_score"] = _clamp(trust_score_f, 0.0, 1.0)

    sampling_weight = descriptor.get("sampling_weight")
    if sampling_weight is None:
        sampling_weight = normalized["trust_score"] * (1.0 + 0.5 * normalized["tier"])
    try:
        sampling_weight_f = float(sampling_weight)
    except Exception:
        sampling_weight_f = 0.0
    normalized["sampling_weight"] = max(0.0, sampling_weight_f)

    return normalized


def validate_episode_descriptor(descriptor: Dict[str, Any]) -> List[str]:
    """
    Validate core fields for downstream RL ingestion.

    Returns:
        List of human-readable validation errors (empty if valid).
    """
    errors: List[str] = []

    obj_vec = descriptor.get("objective_vector")
    if not isinstance(obj_vec, list) or len(obj_vec) != 5:
        errors.append("objective_vector must be a length-5 list of floats")
    else:
        for i, v in enumerate(obj_vec):
            try:
                float(v)
            except Exception:
                errors.append(f"objective_vector[{i}] is not a float-compatible value")
                break

    for field in ("env_name", "engine_type", "task_type"):
        if not descriptor.get(field):
            errors.append(f"missing required field: {field}")

    tier = descriptor.get("tier")
    if tier is None or (isinstance(tier, (int, float)) and tier < 0):
        errors.append("tier must be a non-negative number")

    trust_score = descriptor.get("trust_score")
    if trust_score is None:
        errors.append("trust_score missing")
    else:
        try:
            ts = float(trust_score)
            if ts < 0.0 or ts > 1.0:
                errors.append("trust_score must be within [0, 1]")
        except Exception:
            errors.append("trust_score must be numeric")

    sampling_weight = descriptor.get("sampling_weight")
    if sampling_weight is None:
        errors.append("sampling_weight missing")
    else:
        try:
            sw = float(sampling_weight)
            if sw < 0.0:
                errors.append("sampling_weight must be non-negative")
        except Exception:
            errors.append("sampling_weight must be numeric")

    return errors


def normalize_and_validate(descriptor: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convenience wrapper to normalize then validate.

    Returns:
        (normalized_descriptor, validation_errors)
    """
    normalized = normalize_episode_descriptor(descriptor)
    errors = validate_episode_descriptor(normalized)
    return normalized, errors
