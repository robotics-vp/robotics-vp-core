from src.orchestrator.diffusion_requests import (
    build_diffusion_prompt_from_guidance,
    prompt_to_diffusion_stub_input,
)
from src.valuation.datapack_schema import DataPackMeta
from src.valuation.guidance_profile import GuidanceProfile


def test_diffusion_prompt_includes_constraint_set():
    dp = DataPackMeta(pack_id="dp1")
    dp.semantic_tags = ["fragile", "safety"]
    dp.episode_metrics = {
        "map_first_quality_score": 0.7,
        "semantic_fusion_confidence_mean": 0.8,
        "semantic_disagreement_vla_vs_map": 0.1,
    }

    guidance = GuidanceProfile(
        is_good=True,
        quality_label="high_value",
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="drawer_vase",
        customer_segment="balanced",
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        main_driver="safety_margin",
        delta_mpl=1.0,
        delta_error=-0.1,
        delta_energy_Wh=-0.1,
        delta_J=0.5,
        semantic_tags=["fragile"],
    )

    prompt = build_diffusion_prompt_from_guidance(dp, guidance)
    assert prompt.constraint_set_ref is not None
    assert "hard_bounds" in prompt.constraint_set_ref
    assert prompt.governed_hypotheses
    assert prompt.routing_context is not None
    assert prompt.routing_source == "guidance_contract"

    stub_input = prompt_to_diffusion_stub_input(prompt)
    assert "constraint_set" in stub_input
    assert stub_input["constraint_set"]
    assert stub_input["governed_hypotheses"]
    assert stub_input["routing_context"]["routing_source"] == "guidance_contract"
