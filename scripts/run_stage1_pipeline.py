#!/usr/bin/env python3
"""
Stage 1 Pipeline: Real Video → Diffusion → VLA Distillation → DataPackMeta

Connects:
1. Video references (real demonstrations)
2. Diffusion stub (augmented clips based on semantic tags)
3. VLA controller (skill plan generation)
4. DataPackMeta creation (for downstream RL training)

No GPU, no actual generation - just structural correctness.
"""

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np

from src.diffusion.real_video_diffusion_stub import (
    VideoDiffusionStub,
    DiffusionProposal,
)
from src.vla.transformer_planner import VLATransformerPlanner, VLAInput, VLAPlan
from src.valuation.datapack_schema import (
    DataPackMeta,
    ConditionProfile,
    ObjectiveProfile,
    GuidanceProfile,
    AttributionProfile,
)
from src.orchestrator.diffusion_requests import (
    DiffusionPromptSpec,
    prompt_to_diffusion_stub_input,
)
from src.hrl.skills import SkillID


def simulate_real_video_reference() -> Dict[str, Any]:
    """
    Simulate a real video demonstration reference.

    In production, this would be actual video files from robot operation.
    """
    episode_id = f"real_demo_{int(time.time())}_{np.random.randint(1000)}"

    return {
        "episode_id": episode_id,
        "video_path": f"/data/demonstrations/{episode_id}.mp4",
        "depth_path": f"/data/demonstrations/{episode_id}_depth.npy",
        "timestamp": time.time(),
        "task_type": "drawer_vase",
        "demonstrator": "human_expert",
        "metadata": {
            "duration_s": np.random.uniform(10, 30),
            "success": np.random.random() > 0.2,
            "num_frames": np.random.randint(300, 900),
        },
    }


def extract_semantic_tags_from_video(video_ref: Dict[str, Any]) -> List[str]:
    """
    Extract semantic tags from video (stub).

    In production, this would use a vision model (DINO, CLIP, etc.)
    to extract semantic information from the video.
    """
    # Simulated tag extraction based on task type
    base_tags = ["robot_arm", "gripper", "workspace"]

    if "drawer" in video_ref.get("task_type", ""):
        base_tags.extend(["drawer", "handle", "grasp", "open"])

    if "vase" in video_ref.get("task_type", ""):
        base_tags.extend(["vase", "fragile", "avoid_collision", "safety"])

    # Random additional tags
    possible_tags = [
        "slow_motion", "high_precision", "energy_efficient", "high_speed",
        "careful", "error_recovery", "multiple_attempts",
    ]
    num_extra = np.random.randint(1, 4)
    base_tags.extend(np.random.choice(possible_tags, size=num_extra, replace=False).tolist())

    return base_tags


def generate_diffusion_proposals(
    video_ref: Dict[str, Any],
    semantic_tags: List[str],
    diffusion_stub: VideoDiffusionStub,
    objective_preset: str = "balanced",
    num_proposals: int = 3,
) -> List[DiffusionProposal]:
    """
    Generate augmented video clip proposals using diffusion stub.
    """
    proposals = diffusion_stub.propose_augmented_clips(
        episode_id=video_ref["episode_id"],
        media_refs=[video_ref["video_path"]],
        semantic_tags=semantic_tags,
        objective_preset=objective_preset,
        num_proposals=num_proposals,
    )
    return proposals


def extract_vla_plan_from_proposal(
    proposal: DiffusionProposal,
    vla_planner: VLATransformerPlanner,
) -> VLAPlan:
    """
    Generate VLA skill plan based on diffusion proposal.

    The VLA planner takes the augmentation type and semantic tags
    to generate an appropriate skill sequence.
    """
    # Build instruction from proposal
    instruction = f"{proposal.augmentation_type} with "
    instruction += ", ".join(proposal.semantic_tags[:5])

    vla_input = VLAInput(
        instruction=instruction,
        # In production, would include actual visual features
    )

    plan = vla_planner.plan(vla_input)
    return plan


def create_datapack_from_pipeline(
    video_ref: Dict[str, Any],
    semantic_tags: List[str],
    diffusion_proposal: DiffusionProposal,
    vla_plan: VLAPlan,
    objective_preset: str = "balanced",
) -> DataPackMeta:
    """
    Create DataPackMeta from Stage 1 pipeline outputs.

    This datapack can be used for downstream RL training.
    """
    # Determine objective vector from preset
    if objective_preset == "throughput":
        objective_vector = [2.0, 1.0, 0.5, 1.0, 0.0]
    elif objective_preset == "safety":
        objective_vector = [1.0, 1.0, 0.5, 3.0, 0.0]
    elif objective_preset == "energy_saver":
        objective_vector = [1.0, 1.0, 2.0, 1.0, 0.0]
    else:  # balanced
        objective_vector = [1.0, 1.0, 1.0, 1.0, 0.0]

    # Create profiles using actual schema
    condition_profile = ConditionProfile(
        task_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",  # Would be real robot backend
        world_id="stage1_world",
        vase_offset=(0.0, 0.0, 0.0),
        drawer_friction=0.3,
        lighting_profile="normal",
        occlusion_level=0.0,
        econ_preset="drawer_vase",
        price_per_unit=5.0,
        vase_break_cost=50.0,
        energy_price_kWh=diffusion_proposal.econ_context.get("energy_price_kWh", 0.12),
        objective_vector=objective_vector[:3],  # ConditionProfile uses 3-dim vector
        tags={
            "fragile": "fragile" in semantic_tags,
            "safety_critical": "safety" in semantic_tags,
            "source": "stage1_diffusion",
        },
    )

    objective_profile = ObjectiveProfile(
        env_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",
        task_type=video_ref.get("task_type", "unknown"),
        customer_segment=diffusion_proposal.econ_context.get("customer_segment", "balanced"),
        market_region="US",
        objective_vector=objective_vector,
        wage_human=diffusion_proposal.econ_context.get("wage_human", 18.0),
        energy_price_kWh=diffusion_proposal.econ_context.get("energy_price_kWh", 0.12),
    )

    guidance_profile = GuidanceProfile(
        env_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",
        semantic_tags=semantic_tags,
        focus_areas=[diffusion_proposal.augmentation_type],
        priority="high" if "safety" in semantic_tags else "medium",
        sampling_weight=diffusion_proposal.confidence,
    )

    # Attribution based on diffusion novelty and VLA confidence
    # Get mean confidence, handling potentially empty lists
    vla_conf = vla_plan.confidence
    if isinstance(vla_conf, (list, np.ndarray)) and len(vla_conf) > 0:
        mean_conf = float(np.mean(vla_conf))
    else:
        mean_conf = 0.5  # Default

    attribution_profile = AttributionProfile(
        env_name=video_ref.get("task_type", "drawer_vase"),
        engine_type="pybullet",
        delta_mpl=diffusion_proposal.estimated_novelty * 5.0,  # Novelty correlates with learning
        delta_error=-0.01 if "safety" in semantic_tags else 0.0,
        delta_J=diffusion_proposal.estimated_novelty * 2.0,
        trust_score=mean_conf,
        w_econ=0.8,
        tier=2 if diffusion_proposal.estimated_novelty > 0.6 else 1,
    )

    datapack = DataPackMeta(
        pack_id=f"stage1_{video_ref['episode_id']}_{int(time.time())}",
        task_name=video_ref.get("task_type", "drawer_vase"),
        env_type="pybullet",
        schema_version="2.0-stage1",
        condition=condition_profile,
        objective_profile=objective_profile,
        guidance=guidance_profile,
        attribution=attribution_profile,
        source_type="stage1_diffusion_vla",
        tags=semantic_tags + [f"vla_skill_{s}" for s in vla_plan.skill_sequence[:3]],
    )

    return datapack


def run_stage1_pipeline(
    num_videos: int = 5,
    proposals_per_video: int = 3,
    objective_preset: str = "balanced",
    output_dir: str = "results/stage1_pipeline",
) -> Dict[str, Any]:
    """
    Run full Stage 1 pipeline.

    Returns:
        results: Dict with pipeline outputs and statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    diffusion_stub = VideoDiffusionStub()
    vla_planner = VLATransformerPlanner()

    all_datapacks = []
    all_proposals = []
    all_plans = []
    pipeline_log = []

    print(f"Running Stage 1 pipeline with {num_videos} videos...")

    for i in range(num_videos):
        print(f"\n--- Video {i+1}/{num_videos} ---")

        # Step 1: Simulate real video reference
        video_ref = simulate_real_video_reference()
        print(f"  Video: {video_ref['episode_id']}")

        # Step 2: Extract semantic tags
        semantic_tags = extract_semantic_tags_from_video(video_ref)
        print(f"  Tags: {semantic_tags[:5]}...")

        # Step 3: Generate diffusion proposals
        proposals = generate_diffusion_proposals(
            video_ref, semantic_tags, diffusion_stub, objective_preset, proposals_per_video
        )
        print(f"  Generated {len(proposals)} diffusion proposals")

        # Step 4: For each proposal, generate VLA plan and create datapack
        for j, proposal in enumerate(proposals):
            print(f"    Proposal {j+1}: {proposal.augmentation_type}")

            # Generate VLA plan
            vla_plan = extract_vla_plan_from_proposal(proposal, vla_planner)
            print(f"      VLA skills: {vla_plan.skill_sequence[:3]}...")

            # Create datapack
            datapack = create_datapack_from_pipeline(
                video_ref, semantic_tags, proposal, vla_plan, objective_preset
            )
            print(f"      DataPack: {datapack.pack_id}")
            print(f"      Tier: {datapack.attribution.tier}, Trust: {datapack.attribution.trust_score:.3f}")

            all_datapacks.append(datapack)
            all_proposals.append(proposal)
            all_plans.append(vla_plan)

            # Log
            pipeline_log.append({
                "video_id": video_ref["episode_id"],
                "proposal_id": proposal.proposal_id,
                "augmentation_type": proposal.augmentation_type,
                "semantic_tags": semantic_tags,
                "vla_skills": vla_plan.skill_sequence[:5],
                "vla_confidence": float(np.mean(vla_plan.confidence)),
                "datapack_id": datapack.pack_id,
                "tier": datapack.attribution.tier,
                "trust_score": datapack.attribution.trust_score,
                "estimated_novelty": proposal.estimated_novelty,
            })

    # Save outputs
    # 1. Datapacks
    datapacks_path = os.path.join(output_dir, "datapacks.json")
    datapacks_data = [
        {
            "pack_id": dp.pack_id,
            "task_name": dp.task_name,
            "env_type": dp.env_type,
            "schema_version": dp.schema_version,
            "source_type": dp.source_type,
            "tags": dp.tags,
            "condition": asdict(dp.condition),
            "objective_profile": asdict(dp.objective_profile),
            "guidance": asdict(dp.guidance),
            "attribution": asdict(dp.attribution),
        }
        for dp in all_datapacks
    ]
    with open(datapacks_path, "w") as f:
        json.dump(datapacks_data, f, indent=2)
    print(f"\nSaved {len(all_datapacks)} datapacks to {datapacks_path}")

    # 2. Pipeline log
    log_path = os.path.join(output_dir, "pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(pipeline_log, f, indent=2)
    print(f"Saved pipeline log to {log_path}")

    # Compute statistics
    tier_counts = {1: 0, 2: 0}
    avg_trust = 0.0
    avg_novelty = 0.0
    augmentation_types = {}

    for entry in pipeline_log:
        tier_counts[entry["tier"]] += 1
        avg_trust += entry["trust_score"]
        avg_novelty += entry["estimated_novelty"]
        aug_type = entry["augmentation_type"]
        augmentation_types[aug_type] = augmentation_types.get(aug_type, 0) + 1

    avg_trust /= len(pipeline_log)
    avg_novelty /= len(pipeline_log)

    stats = {
        "total_videos": num_videos,
        "total_proposals": len(all_proposals),
        "total_datapacks": len(all_datapacks),
        "tier_distribution": tier_counts,
        "avg_trust_score": avg_trust,
        "avg_novelty": avg_novelty,
        "augmentation_type_distribution": augmentation_types,
        "objective_preset": objective_preset,
    }

    stats_path = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Pipeline: Video → Diffusion → VLA → DataPack")
    parser.add_argument("--num-videos", type=int, default=5, help="Number of video references")
    parser.add_argument("--proposals-per-video", type=int, default=3, help="Diffusion proposals per video")
    parser.add_argument("--objective-preset", type=str, default="balanced",
                        choices=["balanced", "throughput", "safety", "energy_saver"])
    parser.add_argument("--output-dir", type=str, default="results/stage1_pipeline")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage 1 Pipeline: Real Video → Diffusion → VLA → DataPackMeta")
    print("=" * 70)
    print(f"Videos: {args.num_videos}")
    print(f"Proposals per video: {args.proposals_per_video}")
    print(f"Objective preset: {args.objective_preset}")
    print("=" * 70)

    stats = run_stage1_pipeline(
        num_videos=args.num_videos,
        proposals_per_video=args.proposals_per_video,
        objective_preset=args.objective_preset,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("Stage 1 Pipeline Summary")
    print("=" * 70)
    print(f"Total videos processed: {stats['total_videos']}")
    print(f"Total diffusion proposals: {stats['total_proposals']}")
    print(f"Total datapacks created: {stats['total_datapacks']}")
    print(f"Tier distribution: {stats['tier_distribution']}")
    print(f"Average trust score: {stats['avg_trust_score']:.3f}")
    print(f"Average novelty: {stats['avg_novelty']:.3f}")
    print(f"Augmentation types: {stats['augmentation_type_distribution']}")
    print("\nStage 1 pipeline complete! Datapacks ready for downstream RL training.")


if __name__ == "__main__":
    main()
