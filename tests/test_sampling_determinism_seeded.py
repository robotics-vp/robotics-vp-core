from src.rl.episode_sampling import DataPackRLSampler
from src.valuation.datapack_schema import DataPackMeta


def _dp(pack_id: str, tier: int, trust: float) -> DataPackMeta:
    dp = DataPackMeta(pack_id=pack_id)
    dp.attribution.tier = tier
    dp.attribution.trust_score = trust
    dp.semantic_tags = ["baseline"]
    return dp


def test_sampling_determinism_same_seed_same_order():
    datapacks = [
        _dp("p1", 0, 0.4),
        _dp("p2", 1, 0.6),
        _dp("p3", 2, 0.8),
        _dp("p4", 1, 0.5),
    ]
    sampler = DataPackRLSampler(datapacks=datapacks, default_strategy="balanced")
    batch_a = sampler.sample_batch(batch_size=3, seed=123, strategy="balanced")
    batch_b = sampler.sample_batch(batch_size=3, seed=123, strategy="balanced")

    ids_a = [x.get("pack_id") for x in batch_a]
    ids_b = [x.get("pack_id") for x in batch_b]
    assert ids_a == ids_b
