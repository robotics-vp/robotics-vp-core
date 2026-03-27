"""Runtime package loading for learned D4 knob calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.contracts.schemas import KnobPolicyV1, PlanPolicyConfigV1, RegimeFeaturesV1
from src.regal.knob_model import (
    HeuristicKnobProvider,
    KnobModel,
    MAX_CONSERVATIVE_MULTIPLIER,
    MAX_GAIN_MULTIPLIER,
    MAX_PATIENCE,
    MIN_CONSERVATIVE_MULTIPLIER,
    MIN_GAIN_MULTIPLIER,
    MIN_PATIENCE,
)
from src.regal.knob_model_training import (
    KNOB_FEATURE_NAMES,
    KnobCalibrationNet,
    TORCH_AVAILABLE,
    build_knob_feature_vector,
)

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover
    torch = None


def _mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def _decode_gain(value: float) -> float:
    return MIN_GAIN_MULTIPLIER + (float(value) * (MAX_GAIN_MULTIPLIER - MIN_GAIN_MULTIPLIER))


def _decode_conservative(value: float) -> float:
    return MIN_CONSERVATIVE_MULTIPLIER + (
        float(value) * (MAX_CONSERVATIVE_MULTIPLIER - MIN_CONSERVATIVE_MULTIPLIER)
    )


def _decode_patience(value: float) -> int:
    decoded = MIN_PATIENCE + (float(value) * float(MAX_PATIENCE - MIN_PATIENCE))
    return int(round(decoded))


@dataclass(frozen=True)
class KnobModelRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    model_config: Dict[str, Any]
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def load_knob_model_runtime_package(path: str | Path) -> KnobModelRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(payload.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = (package_path.parent / checkpoint_path).resolve()
    return KnobModelRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(checkpoint_path),
        model_config=_mapping(payload.get("model_config")),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        inference_contract=_mapping(payload.get("inference_contract")),
        promotion_stage=str(payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
        metadata=_mapping(payload.get("metadata")),
    )


class LoadedKnobModel(KnobModel):
    """Loaded learned knob helper with bounded heuristic blending."""

    def __init__(self, package: KnobModelRuntimePackage) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load a trained knob model package")
        self.package = package
        checkpoint_path = Path(package.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Knob model checkpoint not found: {checkpoint_path}")
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        input_dim = int(payload.get("input_dim", len(KNOB_FEATURE_NAMES)))
        hidden_dim = int(payload.get("hidden_dim", 32))
        self.feature_names = list(payload.get("feature_names") or KNOB_FEATURE_NAMES)
        self.model = KnobCalibrationNet(input_dim=input_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.model_sha = str(package.package_id)
        self.benchmark_gate = dict(package.benchmark_gate or {})
        self.execution_preconditions = dict(package.execution_preconditions or {})
        self.promotion_stage = str(package.promotion_stage or "shadow_candidate")
        self.inference_contract = dict(package.inference_contract or {})
        self._heuristic = HeuristicKnobProvider()

    def predict(
        self,
        features: RegimeFeaturesV1,
        base_config: PlanPolicyConfigV1,
    ) -> KnobPolicyV1:
        heuristic_policy = self._heuristic.predict(features, base_config)
        feature_vector = build_knob_feature_vector(features, base_config)
        tensor = torch.from_numpy(feature_vector).float().unsqueeze(0)
        with torch.no_grad():
            raw_outputs = self.model(tensor)
        predictions = torch.sigmoid(raw_outputs[0]).cpu().numpy()
        learned_candidate = {
            "gain_multiplier_override": _decode_gain(predictions[0]),
            "conservative_multiplier_override": _decode_conservative(predictions[1]),
            "patience_override": _decode_patience(predictions[2]),
        }
        benchmark_gate_ready = bool(self.benchmark_gate.get("ready", False))
        helper_weight = 0.55 if benchmark_gate_ready else 0.2
        prior_gain = (
            heuristic_policy.gain_multiplier_override
            if heuristic_policy.gain_multiplier_override is not None
            else base_config.gain_schedule.full_multiplier
        )
        prior_conservative = (
            heuristic_policy.conservative_multiplier_override
            if heuristic_policy.conservative_multiplier_override is not None
            else base_config.gain_schedule.conservative_multiplier
        )
        prior_patience = (
            heuristic_policy.patience_override
            if heuristic_policy.patience_override is not None
            else (base_config.gain_schedule.cooldown_steps or 3)
        )
        blended_gain = ((1.0 - helper_weight) * prior_gain) + (
            helper_weight * learned_candidate["gain_multiplier_override"]
        )
        blended_conservative = ((1.0 - helper_weight) * prior_conservative) + (
            helper_weight * learned_candidate["conservative_multiplier_override"]
        )
        blended_patience = int(
            round(
                ((1.0 - helper_weight) * prior_patience)
                + (helper_weight * learned_candidate["patience_override"])
            )
        )
        policy = KnobPolicyV1(
            policy_source="learned",
            model_sha=self.model_sha,
            regime_features_sha=features.sha256(),
            promotion_stage="promoted" if benchmark_gate_ready else self.promotion_stage,
            gain_multiplier_override=float(blended_gain),
            conservative_multiplier_override=float(blended_conservative),
            patience_override=blended_patience,
            threshold_overrides=heuristic_policy.threshold_overrides,
            task_family_biases=heuristic_policy.task_family_biases,
            trace={
                "helper_weight": helper_weight,
                "benchmark_gate_ready": benchmark_gate_ready,
                "heuristic_prior": {
                    "gain_multiplier_override": prior_gain,
                    "conservative_multiplier_override": prior_conservative,
                    "patience_override": prior_patience,
                    "threshold_overrides": heuristic_policy.threshold_overrides,
                    "task_family_biases": heuristic_policy.task_family_biases,
                },
                "learned_candidate": learned_candidate,
                "final_policy": {
                    "gain_multiplier_override": float(blended_gain),
                    "conservative_multiplier_override": float(blended_conservative),
                    "patience_override": blended_patience,
                },
            },
        )
        return self.apply_hard_constraints(policy)


def resolve_knob_model(
    *,
    use_learned: bool = False,
    model_path: Optional[str | Path] = None,
    required: bool = False,
) -> KnobModel:
    if not use_learned:
        return HeuristicKnobProvider()
    if model_path is None:
        if required:
            raise ValueError("learned knob model required but no package/checkpoint path was provided")
        return HeuristicKnobProvider()
    candidate = Path(model_path)
    package: Optional[KnobModelRuntimePackage] = None
    if candidate.suffix == ".json":
        if not candidate.exists():
            if required:
                raise FileNotFoundError(f"Knob model package not found: {candidate}")
            return HeuristicKnobProvider()
        package = load_knob_model_runtime_package(candidate)
    else:
        if not candidate.exists():
            if required:
                raise FileNotFoundError(f"Knob model checkpoint not found: {candidate}")
            return HeuristicKnobProvider()
        package = KnobModelRuntimePackage(
            package_id=candidate.stem,
            package_path=str(candidate.resolve()),
            checkpoint_path=str(candidate.resolve()),
            model_config={},
            benchmark_gate={},
            execution_preconditions={},
            inference_contract={},
            promotion_stage="shadow_candidate",
            metadata={},
        )
    model = LoadedKnobModel(package)
    if required and not bool(model.benchmark_gate.get("ready", False)):
        raise ValueError("learned knob model requires a benchmark-gated package")
    return model


__all__ = [
    "KnobModelRuntimePackage",
    "LoadedKnobModel",
    "load_knob_model_runtime_package",
    "resolve_knob_model",
]
