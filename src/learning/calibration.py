"""Deterministic calibration and drift helpers for shadow advisors and regal promotion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence


def _floats(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CalibrationSummary:
    """Compact, JSON-safe summary of learned advisor quality."""

    sample_count: int
    expected_calibration_error: float
    brier_score: float
    agreement_rate: float
    sign_consistency: float
    monotonicity_score: float
    drift_score: float
    confidence_mean: float
    target_mean: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "expected_calibration_error": float(self.expected_calibration_error),
            "brier_score": float(self.brier_score),
            "agreement_rate": float(self.agreement_rate),
            "sign_consistency": float(self.sign_consistency),
            "monotonicity_score": float(self.monotonicity_score),
            "drift_score": float(self.drift_score),
            "confidence_mean": float(self.confidence_mean),
            "target_mean": float(self.target_mean),
            "metadata": dict(self.metadata),
        }


def expected_calibration_error(confidences: Sequence[float], outcomes: Sequence[float], *, bins: int = 10) -> float:
    conf = [_clamp01(value) for value in confidences]
    truth = [_clamp01(value) for value in outcomes]
    if not conf or len(conf) != len(truth):
        return 1.0
    total = float(len(conf))
    error = 0.0
    for bin_idx in range(max(1, int(bins))):
        lo = bin_idx / float(max(1, bins))
        hi = (bin_idx + 1) / float(max(1, bins))
        indices = [index for index, value in enumerate(conf) if lo <= value <= hi if bin_idx == bins - 1 or value < hi]
        if not indices:
            continue
        mean_conf = sum(conf[index] for index in indices) / float(len(indices))
        mean_truth = sum(truth[index] for index in indices) / float(len(indices))
        error += (len(indices) / total) * abs(mean_conf - mean_truth)
    return float(error)


def brier_score(confidences: Sequence[float], outcomes: Sequence[float]) -> float:
    conf = [_clamp01(value) for value in confidences]
    truth = [_clamp01(value) for value in outcomes]
    if not conf or len(conf) != len(truth):
        return 1.0
    return float(sum((estimate - actual) ** 2 for estimate, actual in zip(conf, truth)) / len(conf))


def agreement_rate(predictions: Sequence[float], targets: Sequence[float], *, tolerance: float = 0.1) -> float:
    pred = _floats(predictions)
    tgt = _floats(targets)
    if not pred or len(pred) != len(tgt):
        return 0.0
    matches = sum(1 for estimate, actual in zip(pred, tgt) if abs(estimate - actual) <= float(tolerance))
    return float(matches / len(pred))


def sign_consistency(predictions: Sequence[float], targets: Sequence[float]) -> float:
    pred = _floats(predictions)
    tgt = _floats(targets)
    if not pred or len(pred) != len(tgt):
        return 0.0
    matches = 0
    for estimate, actual in zip(pred, tgt):
        if estimate == 0.0 and actual == 0.0:
            matches += 1
        elif estimate == 0.0 or actual == 0.0:
            continue
        elif (estimate > 0.0) == (actual > 0.0):
            matches += 1
    return float(matches / len(pred))


def monotonicity_score(inputs: Sequence[float], outputs: Sequence[float], *, increasing: bool = True) -> float:
    xs = _floats(inputs)
    ys = _floats(outputs)
    if len(xs) < 2 or len(xs) != len(ys):
        return 1.0
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    good = 0
    total = 0
    for (_, left), (_, right) in zip(pairs, pairs[1:]):
        total += 1
        if increasing and right >= left:
            good += 1
        if not increasing and right <= left:
            good += 1
    return float(good / max(1, total))


def distribution_shift_score(reference_vectors: Sequence[Sequence[float]], current_vectors: Sequence[Sequence[float]]) -> float:
    reference = [list(map(float, row)) for row in reference_vectors if row is not None]
    current = [list(map(float, row)) for row in current_vectors if row is not None]
    if not reference or not current:
        return 1.0
    dim = min(len(reference[0]), len(current[0]))
    if dim <= 0:
        return 1.0
    score = 0.0
    for index in range(dim):
        ref_values = [row[index] for row in reference if len(row) > index]
        cur_values = [row[index] for row in current if len(row) > index]
        if not ref_values or not cur_values:
            score += 1.0
            continue
        ref_mean = sum(ref_values) / float(len(ref_values))
        cur_mean = sum(cur_values) / float(len(cur_values))
        ref_scale = max(1e-6, sum(abs(value - ref_mean) for value in ref_values) / float(len(ref_values)))
        score += min(1.0, abs(cur_mean - ref_mean) / ref_scale)
    return float(score / dim)


def summarize_calibration(
    *,
    confidences: Sequence[float],
    outcomes: Sequence[float],
    predictions: Sequence[float],
    targets: Sequence[float],
    monotonic_inputs: Sequence[float],
    monotonic_outputs: Sequence[float],
    reference_vectors: Sequence[Sequence[float]],
    current_vectors: Sequence[Sequence[float]],
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationSummary:
    conf = [_clamp01(value) for value in confidences]
    truth = [_clamp01(value) for value in outcomes]
    target_values = _floats(targets)
    return CalibrationSummary(
        sample_count=min(len(conf), len(truth), len(predictions), len(target_values)),
        expected_calibration_error=expected_calibration_error(conf, truth),
        brier_score=brier_score(conf, truth),
        agreement_rate=agreement_rate(predictions, target_values),
        sign_consistency=sign_consistency(predictions, target_values),
        monotonicity_score=monotonicity_score(monotonic_inputs, monotonic_outputs),
        drift_score=distribution_shift_score(reference_vectors, current_vectors),
        confidence_mean=(sum(conf) / float(len(conf))) if conf else 0.0,
        target_mean=(sum(target_values) / float(len(target_values))) if target_values else 0.0,
        metadata=dict(metadata or {}),
    )
