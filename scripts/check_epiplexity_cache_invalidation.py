"""Sanity checks for epiplexity cache key invalidation."""
from __future__ import annotations

import torch

from src.epiplexity.tracker import EpiplexityTracker, EpiplexityRunKey, ComputeBudget
from src.utils.determinism import maybe_enable_determinism_from_env


def _make_key(**overrides):
    base = dict(
        repr_id="raw",
        repr_version_hash="v1",
        tokenizer_version="v1",
        transform_chain_hash="v1",
        dataset_slice_id="slice",
        probe_model_id="probe",
        compute_budget_id="steps_5_bs_4",
        seed=0,
    )
    base.update(overrides)
    return EpiplexityRunKey(**base)


def main() -> None:
    seed = 0
    det_seed = maybe_enable_determinism_from_env(default_seed=seed)
    if det_seed is not None:
        seed = det_seed
    torch.manual_seed(seed)
    tokens = torch.randn(2, 4, 8)
    tracker = EpiplexityTracker(cache_dir="artifacts/epiplexity_cache_diag")
    budget = ComputeBudget(max_steps=5, batch_size=4)

    base_key = _make_key()
    base_hash = base_key.to_hash()
    print("base hash:", base_hash)

    variants = {
        "repr_id": _make_key(repr_id="raw2").to_hash(),
        "repr_version_hash": _make_key(repr_version_hash="v2").to_hash(),
        "tokenizer_version": _make_key(tokenizer_version="tok2").to_hash(),
        "transform_chain_hash": _make_key(transform_chain_hash="tx2").to_hash(),
        "probe_model_id": _make_key(probe_model_id="probe2").to_hash(),
        "compute_budget_id": _make_key(compute_budget_id="steps_10_bs_4").to_hash(),
        "seed": _make_key(seed=1).to_hash(),
    }

    for name, h in variants.items():
        print(f"{name} hash: {h} differs={h != base_hash}")
        assert h != base_hash, f"Hash did not change for {name}"

    result_a = tracker.evaluate_tokens(tokens, base_key, budget)
    result_b = tracker.evaluate_tokens(tokens, base_key, budget)

    assert result_a.S_T_proxy == result_b.S_T_proxy
    assert result_a.H_T_proxy == result_b.H_T_proxy
    assert result_a.epi_per_flop == result_b.epi_per_flop
    print("cache repeat result: OK")


if __name__ == "__main__":
    main()
