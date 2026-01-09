"""Probe epiplexity harness for delta-epi-per-flop discrimination.

Deterministic, cheap, portable-first harness that computes
delta-epiplexity-per-compute with stability and transfer gates.

Stability gate includes:
- Sign consistency across seeds
- Delta std threshold
- Multi-variant and subsample support
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.contracts.schemas import ProbeConfigV1, ProbeEpiReportV1
from src.utils.config_digest import sha256_json


# =============================================================================
# Probe Models
# =============================================================================

class ProbeModel(ABC, nn.Module):
    """Abstract base for probe models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def flops_per_example(self) -> int:
        """FLOPs per forward pass per example."""
        pass

    @property
    @abstractmethod
    def variant_name(self) -> str:
        """Name of probe variant."""
        pass


class LinearProbe(ProbeModel):
    """Simple linear probe."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self._input_dim = input_dim
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def flops_per_example(self) -> int:
        return 2 * self._input_dim * self._output_dim

    @property
    def variant_name(self) -> str:
        return "linear"


class MLPProbe(ProbeModel):
    """2-layer MLP probe."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def flops_per_example(self) -> int:
        return (2 * self._input_dim * self._hidden_dim +
                self._hidden_dim +
                2 * self._hidden_dim * self._output_dim)

    @property
    def variant_name(self) -> str:
        return f"mlp_{self._hidden_dim}"


def create_probe(variant: str, input_dim: int, hidden_dim: int = 64) -> ProbeModel:
    """Factory to create probe model."""
    if variant == "linear":
        return LinearProbe(input_dim)
    elif variant == "mlp":
        return MLPProbe(input_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown probe variant: {variant}")


# =============================================================================
# Epiplexity Score Computation
# =============================================================================

def compute_epiplexity_score(
    model: ProbeModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Compute prequential/heldout loss as epiplexity proxy.

    Lower loss = higher predictability = lower epiplexity.
    We return negative loss so higher = better.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

    if n_samples == 0:
        return 0.0

    mean_loss = total_loss / n_samples
    return -mean_loss


# =============================================================================
# Probe Harness Configuration
# =============================================================================

@dataclass
class ProbeHarnessConfig:
    """Configuration for probe harness with tightened stability."""

    # Probe variants to run (stability across variants)
    probe_variants: List[str] = field(default_factory=lambda: ["linear"])
    probe_steps: int = 200
    batch_size: int = 32
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    input_dim: int = 128
    hidden_dim: int = 64
    learning_rate: float = 0.01

    # Stability thresholds (tightened)
    stability_sign_threshold: float = 0.7  # Min sign consistency
    stability_std_threshold: float = 0.5  # Max std relative to mean
    min_stability_runs: int = 3  # Minimum runs for stability

    # Raw delta floor (prevent vanishing changes)
    min_raw_delta: float = 0.01  # Require delta_epi >= ε

    # Transfer threshold
    transfer_threshold: float = 0.0

    # Subsampling for stability (fraction of data)
    subsample_fractions: List[float] = field(default_factory=lambda: [1.0])

    # Repr identifiers for provenance
    baseline_repr_id: str = "baseline"
    after_repr_id: str = "after"
    transform_id: Optional[str] = None

    def to_probe_config(self) -> ProbeConfigV1:
        return ProbeConfigV1(
            probe_variant=",".join(self.probe_variants),
            probe_steps=self.probe_steps,
            batch_size=self.batch_size,
            seeds=self.seeds,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim if "mlp" in self.probe_variants else None,
        )


# =============================================================================
# Extended Probe Report
# =============================================================================

@dataclass
class ExtendedStabilityStats:
    """Extended stability statistics."""

    sign_consistency: float
    delta_std: float
    delta_mean: float
    relative_std: float  # std / |mean| (if mean != 0)
    n_runs: int

    per_variant_deltas: Dict[str, List[float]] = field(default_factory=dict)
    per_subsample_deltas: Dict[str, List[float]] = field(default_factory=dict)

    # Combined pass: sign AND std AND n_runs
    stability_pass: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sign_consistency": self.sign_consistency,
            "delta_std": self.delta_std,
            "delta_mean": self.delta_mean,
            "relative_std": self.relative_std,
            "n_runs": self.n_runs,
            "per_variant_deltas": self.per_variant_deltas,
            "per_subsample_deltas": self.per_subsample_deltas,
            "stability_pass": self.stability_pass,
        }


# =============================================================================
# Probe Harness
# =============================================================================

class ProbeHarness:
    """Deterministic probe epiplexity harness with tightened stability.

    Computes delta-epiplexity-per-flop with:
    - Sign consistency across seeds
    - Delta std threshold
    - Multi-variant support
    - Subsample support
    - Raw delta floor
    """

    def __init__(self, config: Optional[ProbeHarnessConfig] = None):
        self.config = config or ProbeHarnessConfig()
        self.device = "cpu"

    def run(
        self,
        baseline_data: Tuple[np.ndarray, np.ndarray],
        after_data: Tuple[np.ndarray, np.ndarray],
        ood_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> ProbeEpiReportV1:
        """Run the probe harness with extended stability checks."""
        all_deltas: List[float] = []
        per_variant_deltas: Dict[str, List[float]] = {}
        per_subsample_deltas: Dict[str, List[float]] = {}
        per_seed_ood_delta: List[float] = []
        total_flops = 0

        # Run across variants x seeds x subsamples
        for variant in self.config.probe_variants:
            per_variant_deltas[variant] = []

            for subsample_frac in self.config.subsample_fractions:
                subsample_key = f"frac_{subsample_frac:.2f}"
                if subsample_key not in per_subsample_deltas:
                    per_subsample_deltas[subsample_key] = []

                # Subsample data
                baseline_sub = self._subsample(baseline_data, subsample_frac)
                after_sub = self._subsample(after_data, subsample_frac)

                for seed in self.config.seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    # Train + eval
                    baseline_score = self._train_and_eval(
                        baseline_sub, seed, variant
                    )
                    after_score = self._train_and_eval(
                        after_sub, seed, variant
                    )

                    delta = after_score - baseline_score
                    all_deltas.append(delta)
                    per_variant_deltas[variant].append(delta)
                    per_subsample_deltas[subsample_key].append(delta)

                    # Track FLOPs
                    probe = create_probe(variant, self.config.input_dim, self.config.hidden_dim)
                    total_flops += (
                        self.config.probe_steps *
                        self.config.batch_size *
                        probe.flops_per_example *
                        2  # baseline + after
                    )

                    # OOD evaluation
                    if ood_data is not None:
                        ood_baseline = self._train_and_eval(baseline_sub, seed, variant, ood_data)
                        ood_after = self._train_and_eval(after_sub, seed, variant, ood_data)
                        per_seed_ood_delta.append(ood_after - ood_baseline)

        # Compute aggregates
        mean_delta = float(np.mean(all_deltas))
        delta_std = float(np.std(all_deltas))

        # Sign consistency
        if mean_delta != 0:
            expected_sign = np.sign(mean_delta)
            same_sign = sum(1 for d in all_deltas if np.sign(d) == expected_sign)
            sign_consistency = same_sign / len(all_deltas)
        else:
            sign_consistency = 1.0

        # Relative std
        relative_std = delta_std / abs(mean_delta) if abs(mean_delta) > 1e-10 else float('inf')

        # Extended stability check
        stability_stats = ExtendedStabilityStats(
            sign_consistency=sign_consistency,
            delta_std=delta_std,
            delta_mean=mean_delta,
            relative_std=relative_std,
            n_runs=len(all_deltas),
            per_variant_deltas=per_variant_deltas,
            per_subsample_deltas=per_subsample_deltas,
        )

        # Stability pass: sign AND std AND n_runs AND raw delta floor
        stability_pass = (
            sign_consistency >= self.config.stability_sign_threshold and
            relative_std <= self.config.stability_std_threshold and
            len(all_deltas) >= self.config.min_stability_runs and
            abs(mean_delta) >= self.config.min_raw_delta
        )
        stability_stats.stability_pass = stability_pass

        # FLOPs normalization
        delta_epi_per_flop = mean_delta / total_flops if total_flops > 0 else 0.0

        # OOD / Transfer
        ood_delta = None
        transfer_pass = False
        if per_seed_ood_delta:
            ood_delta = float(np.mean(per_seed_ood_delta))
            transfer_pass = ood_delta > self.config.transfer_threshold

        # Build report
        report = ProbeEpiReportV1(
            report_id=str(uuid.uuid4())[:8],
            probe_config=self.config.to_probe_config(),
            baseline_score=float(np.mean([d for d in all_deltas])),  # Approx
            after_score=float(np.mean([d for d in all_deltas])) + mean_delta,
            delta=mean_delta,
            flops_estimate=float(total_flops),
            delta_epi_per_flop=delta_epi_per_flop,
            per_seed_deltas=all_deltas,
            sign_consistency=sign_consistency,
            stability_pass=stability_pass,
            ood_delta=ood_delta,
            transfer_pass=transfer_pass,
            num_samples_id=len(baseline_data[0]),
            num_samples_ood=len(ood_data[0]) if ood_data else 0,
        )
        report.compute_hashes()

        return report

    def _subsample(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data for stability check."""
        if fraction >= 1.0:
            return data
        X, y = data
        n = len(X)
        k = max(1, int(n * fraction))
        indices = np.random.choice(n, k, replace=False)
        return X[indices], y[indices]

    def _train_and_eval(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        seed: int,
        variant: str,
        eval_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> float:
        """Train probe and compute epiplexity score."""
        X, y = data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Create model
        probe = create_probe(
            variant,
            self.config.input_dim,
            self.config.hidden_dim,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(probe.parameters(), lr=self.config.learning_rate)

        # Train
        probe.train()
        steps = 0
        for epoch in range(100):
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = probe(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps >= self.config.probe_steps:
                    break
            if steps >= self.config.probe_steps:
                break

        # Eval
        if eval_data is not None:
            X_eval, y_eval = eval_data
            X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
            y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(-1)
            eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
            eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size)
        else:
            eval_loader = dataloader

        return compute_epiplexity_score(probe, eval_loader, criterion, self.device)


def write_probe_report(path: str, report: ProbeEpiReportV1) -> str:
    """Write probe report to JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, sort_keys=True)
    return report.report_sha


# =============================================================================
# Probe Harness Registry (like audit registry)
# =============================================================================

@dataclass(frozen=True)
class ProbeHarnessDefinition:
    """Immutable probe harness definition for registry."""

    harness_id: str
    probe_variants: tuple  # Frozen tuple
    probe_steps: int
    batch_size: int
    seeds: tuple
    input_dim: int
    hidden_dim: int
    stability_sign_threshold: float
    stability_std_threshold: float
    min_raw_delta: float
    transfer_threshold: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "probe_variants": list(self.probe_variants),
            "probe_steps": self.probe_steps,
            "batch_size": self.batch_size,
            "seeds": list(self.seeds),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "stability_sign_threshold": self.stability_sign_threshold,
            "stability_std_threshold": self.stability_std_threshold,
            "min_raw_delta": self.min_raw_delta,
            "transfer_threshold": self.transfer_threshold,
            "description": self.description,
        }

    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_config(self) -> ProbeHarnessConfig:
        return ProbeHarnessConfig(
            probe_variants=list(self.probe_variants),
            probe_steps=self.probe_steps,
            batch_size=self.batch_size,
            seeds=list(self.seeds),
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            stability_sign_threshold=self.stability_sign_threshold,
            stability_std_threshold=self.stability_std_threshold,
            min_raw_delta=self.min_raw_delta,
            transfer_threshold=self.transfer_threshold,
        )


# Default probe harness definitions
_SMOKE_PROBE_HARNESS = ProbeHarnessDefinition(
    harness_id="smoke_probe_v1",
    probe_variants=("linear",),
    probe_steps=100,
    batch_size=16,
    seeds=(42, 43, 44),
    input_dim=32,
    hidden_dim=32,
    stability_sign_threshold=0.7,
    stability_std_threshold=0.5,
    min_raw_delta=0.01,
    transfer_threshold=0.0,
    description="Minimal smoke test probe harness",
)

_STANDARD_PROBE_HARNESS = ProbeHarnessDefinition(
    harness_id="standard_probe_v1",
    probe_variants=("linear", "mlp"),
    probe_steps=200,
    batch_size=32,
    seeds=(42, 43, 44, 45, 46),
    input_dim=128,
    hidden_dim=64,
    stability_sign_threshold=0.8,
    stability_std_threshold=0.3,
    min_raw_delta=0.05,
    transfer_threshold=0.0,
    description="Standard probe harness with multi-variant stability",
)

# Registry
PROBE_HARNESS_REGISTRY: Dict[str, ProbeHarnessDefinition] = {
    "smoke_probe_v1": _SMOKE_PROBE_HARNESS,
    "standard_probe_v1": _STANDARD_PROBE_HARNESS,
}


def get_probe_harness(harness_id: str) -> ProbeHarnessDefinition:
    """Get probe harness definition by ID."""
    if harness_id not in PROBE_HARNESS_REGISTRY:
        raise KeyError(
            f"Unknown probe harness: {harness_id}. "
            f"Available: {list(PROBE_HARNESS_REGISTRY.keys())}"
        )
    return PROBE_HARNESS_REGISTRY[harness_id]


def get_probe_harness_sha(harness_id: str) -> str:
    """Get SHA of registered probe harness."""
    return get_probe_harness(harness_id).sha256()


__all__ = [
    "ProbeModel",
    "LinearProbe",
    "MLPProbe",
    "create_probe",
    "compute_epiplexity_score",
    "ProbeHarnessConfig",
    "ExtendedStabilityStats",
    "ProbeHarness",
    "write_probe_report",
    "ProbeHarnessDefinition",
    "PROBE_HARNESS_REGISTRY",
    "get_probe_harness",
    "get_probe_harness_sha",
]
