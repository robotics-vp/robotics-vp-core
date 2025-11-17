from typing import Dict, Any

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.orchestrator.diffusion_requests import DiffusionPromptSpec


def build_episode_stub_from_diffusion_request(req: DiffusionPromptSpec) -> EpisodeInfoSummary:
    """
    Create an EpisodeInfoSummary stub from a diffusion request.
    Placeholder metrics (set to 0.0) to be filled when real rollouts arrive.
    """
    return EpisodeInfoSummary(
        termination_reason="unknown",
        mpl_episode=0.0,
        ep_episode=0.0,
        error_rate_episode=0.0,
        throughput_units_per_hour=0.0,
        energy_Wh=0.0,
        energy_Wh_per_unit=0.0,
        energy_Wh_per_hour=0.0,
        limb_energy_Wh={},
        skill_energy_Wh={},
        energy_per_limb={},
        energy_per_skill={},
        energy_per_joint={},
        energy_per_effector={},
        coordination_metrics={},
        profit=0.0,
        episode_id=req.request_id,
        media_refs={"sim_trace": ""},
        wage_parity=None,
    )


def attach_rollout_metrics_to_diffusion_stub(
    stub_summary: EpisodeInfoSummary,
    mpl: float,
    error_rate: float,
    energy_wh: float,
    **kwargs: Dict[str, Any],
) -> EpisodeInfoSummary:
    """
    Return a copy of stub_summary with rollout metrics filled in.
    """
    stub_summary.mpl_episode = mpl
    stub_summary.throughput_units_per_hour = mpl
    stub_summary.error_rate_episode = error_rate
    stub_summary.energy_Wh = energy_wh
    stub_summary.energy_Wh_per_unit = energy_wh
    stub_summary.energy_Wh_per_hour = energy_wh
    return stub_summary
