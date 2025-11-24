"""
Smoke test for Phase I datasets.

Validates:
- Datasets load without errors
- Deterministic ordering with fixed seed
- JSON-safe serialization
- Econ/condition invariants (econ slice + novelty/recovery fields present)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets import HydraPolicyDataset, Sima2SegmenterDataset, SpatialRNNDataset, VisionPhase1Dataset  # noqa: E402
from src.datasets.base import set_deterministic_seeds  # noqa: E402
from src.utils.json_safe import to_json_safe  # noqa: E402


def assert_deterministic(factory, seed: int, max_samples: int = 8) -> None:
    set_deterministic_seeds(seed)
    ds_a = factory(seed=seed, max_samples=max_samples)
    set_deterministic_seeds(seed)
    ds_b = factory(seed=seed, max_samples=max_samples)
    assert len(ds_a) == len(ds_b) > 0, "Datasets should have consistent length"
    first_a = json.dumps(to_json_safe(ds_a.samples[0]), sort_keys=True)
    first_b = json.dumps(to_json_safe(ds_b.samples[0]), sort_keys=True)
    assert first_a == first_b, "Ordering differs between runs with same seed"


def main() -> None:
    seed = 17
    set_deterministic_seeds(seed)

    datasets = {
        "vision": VisionPhase1Dataset(seed=seed, max_samples=8),
        "spatial_rnn": SpatialRNNDataset(seed=seed, max_samples=8),
        "sima2_segmenter": Sima2SegmenterDataset(seed=seed, max_samples=8),
        "hydra_policy": HydraPolicyDataset(seed=seed, max_samples=8),
    }

    for name, ds in datasets.items():
        assert len(ds) > 0, f"{name} dataset is empty"
        # JSON-safe
        json.dumps(to_json_safe(ds.samples[:2]), sort_keys=True)

    # Determinism checks
    assert_deterministic(VisionPhase1Dataset, seed)
    assert_deterministic(SpatialRNNDataset, seed)
    assert_deterministic(Sima2SegmenterDataset, seed)
    assert_deterministic(HydraPolicyDataset, seed)

    # Econ invariants: hydra condition features carry econ and novelty/recovery
    hydra_sample = datasets["hydra_policy"].samples[0]
    condition_features = hydra_sample.get("condition_features", {})
    econ_slice = condition_features.get("econ_slice") or {}
    assert "target_mpl" in econ_slice and "current_wage_parity" in econ_slice, "Econ slice missing required fields"
    assert condition_features.get("novelty_tier") is not None, "Novelty tier missing"
    assert condition_features.get("recovery_priority") is not None, "Recovery priority missing"

    print("Phase I dataset smoke test: PASSED")


if __name__ == "__main__":
    main()
