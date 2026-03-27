from src.rl.curriculum import DataPackCurriculum


class _StubSampler:
    def __init__(self) -> None:
        self.advisory = None

    def sample_batch(self, batch_size: int, seed: int, strategy: str):
        return [
            {
                "episode_id": f"{strategy}_{idx}",
                "sampling_metadata": {
                    "strategy": strategy,
                    "seed": seed,
                },
            }
            for idx in range(batch_size)
        ]


def test_curriculum_attaches_bounded_authority_receipt() -> None:
    curriculum = DataPackCurriculum(
        sampler=_StubSampler(),
        total_steps=100,
        config={"base_seed": 7},
    )

    batch = curriculum.sample_batch(step=10, batch_size=2)

    assert len(batch) == 2
    receipt = batch[0]["sampling_metadata"]["curriculum_receipt"]
    assert receipt["receipt_kind"] == "curriculum_dispatch_receipt_v1"
    assert receipt["authority_class"] == "bounded_authority"
    assert receipt["decision_scope"] == "training_distribution_curriculum"
    assert receipt["reward_math_mutation"] is False
    assert receipt["phase"] == "warmup"
