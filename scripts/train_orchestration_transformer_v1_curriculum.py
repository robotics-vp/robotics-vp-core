#!/usr/bin/env python3
"""
Train Orchestration Transformer with V1 Curriculum.

Enhanced supervised learning with:
- Multi-head auxiliary supervision (profile, preset, urgency, Pareto)
- Curriculum mixing: heuristic labels + Pareto-optimal choices + semantic consistency
- Semantic consistency evaluation

This builds the "brain" that learns from econ/datapack/semantic feedback loops.
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from src.orchestrator.orchestration_transformer import (
    OrchestrationTransformer,
    TOOL_NAMES,
    _encode_ctx,
)
from src.orchestrator.training_dataset import (
    build_training_dataset,
    build_mixed_training_dataset,
    split_dataset_by_source,
    dataset_to_tensors,
    save_dataset,
    OrchestrationSample,
    EconSemanticDecisionSummary,
    generate_synthetic_context,
    classify_pareto_frontier,
    derive_urgency_level,
    derive_chosen_profile_from_signals,
)
from src.orchestrator.toolspecs import ToolCall


# =============================================================================
# Auxiliary Label Mappings
# =============================================================================

PROFILE_TO_IDX = {"BASE": 0, "BOOST": 1, "SAVER": 2, "SAFE": 3}
IDX_TO_PROFILE = {v: k for k, v in PROFILE_TO_IDX.items()}

PRESET_TO_IDX = {"balanced": 0, "throughput": 1, "energy_saver": 2, "safety": 3}
IDX_TO_PRESET = {v: k for k, v in PRESET_TO_IDX.items()}

URGENCY_TO_IDX = {"none": 0, "moderate": 1, "high": 2, "critical": 3}
IDX_TO_URGENCY = {v: k for k, v in URGENCY_TO_IDX.items()}

PARETO_TO_IDX = {"balanced": 0, "energy_tight": 1, "mpl_tight": 2, "safety_focused": 3}
IDX_TO_PARETO = {v: k for k, v in PARETO_TO_IDX.items()}


# =============================================================================
# Enhanced Model with Auxiliary Heads
# =============================================================================

class OrchestrationTransformerWithAuxHeads(nn.Module):
    """
    Orchestration transformer with auxiliary prediction heads.

    Predicts:
    - Primary: tool sequence
    - Auxiliary: profile, preset, urgency, Pareto classification
    """

    def __init__(self, vocab_size=128, hidden=96, ctx_dim=36):
        super().__init__()
        self.hidden = hidden
        self.ctx_dim = ctx_dim

        # Base transformer
        self.instr_embed = nn.Embedding(vocab_size, hidden)
        self.ctx_proj = nn.Linear(ctx_dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Primary head: tool prediction
        self.tool_head = nn.Linear(hidden, len(TOOL_NAMES))
        self.arg_head = nn.Linear(hidden, 32)  # Argument vector

        # Auxiliary heads
        self.profile_head = nn.Linear(hidden, len(PROFILE_TO_IDX))
        self.preset_head = nn.Linear(hidden, len(PRESET_TO_IDX))
        self.urgency_head = nn.Linear(hidden, len(URGENCY_TO_IDX))
        self.pareto_head = nn.Linear(hidden, len(PARETO_TO_IDX))

        # Consistency score regression head
        self.consistency_head = nn.Linear(hidden, 1)

    def forward(self, instr_tokens, ctx_features):
        """
        Forward pass with all heads.

        Args:
            instr_tokens: (B, seq_len) instruction token IDs
            ctx_features: (B, ctx_dim) context features

        Returns:
            tool_logits: (B, num_tools)
            arg_vec: (B, 32)
            aux_outputs: dict of auxiliary predictions
        """
        # Embed instruction
        instr_emb = self.instr_embed(instr_tokens)  # (B, seq_len, hidden)

        # Embed context and add as first token
        ctx_emb = self.ctx_proj(ctx_features).unsqueeze(1)  # (B, 1, hidden)
        x = torch.cat([ctx_emb, instr_emb], dim=1)  # (B, seq_len+1, hidden)

        # Transformer
        x = self.transformer(x)

        # Use CLS token (first position = context) for predictions
        cls_repr = x[:, 0, :]  # (B, hidden)

        # Primary predictions
        tool_logits = self.tool_head(cls_repr)
        arg_vec = self.arg_head(cls_repr)

        # Auxiliary predictions
        aux_outputs = {
            "profile_logits": self.profile_head(cls_repr),
            "preset_logits": self.preset_head(cls_repr),
            "urgency_logits": self.urgency_head(cls_repr),
            "pareto_logits": self.pareto_head(cls_repr),
            "consistency_score": torch.sigmoid(self.consistency_head(cls_repr)).squeeze(-1),
        }

        return tool_logits, arg_vec, aux_outputs


# =============================================================================
# Curriculum Dataset Generation
# =============================================================================

def generate_pareto_optimal_sample(seed: int = None) -> OrchestrationSample:
    """
    Generate a sample where the tool choice is Pareto-optimal.

    This teaches the transformer to make economically rational choices
    that balance multiple objectives on the Pareto frontier.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate context with clear Pareto trade-offs
    ctx = generate_synthetic_context(seed)

    # Simulate econ/datapack signals
    econ_signals = {
        "mpl_urgency": np.random.uniform(0.0, 1.0),
        "error_urgency": np.random.uniform(0.0, 1.0),
        "energy_urgency": np.random.uniform(0.0, 1.0),
        "wage_parity": ctx.mean_trust * ctx.mean_w_econ,
    }

    datapack_signals = {
        "coverage_score": np.random.uniform(0.3, 1.0),
        "tier2_fraction": np.random.uniform(0.0, 0.3),
        "tier1_fraction": np.random.uniform(0.2, 0.5),
    }

    # Derive Pareto-optimal choices
    pareto_class = classify_pareto_frontier(econ_signals, datapack_signals)
    urgency = derive_urgency_level(econ_signals)
    chosen_profile = derive_chosen_profile_from_signals(econ_signals, datapack_signals)

    # Determine objective preset based on Pareto position
    if pareto_class == "safety_focused":
        objective_preset = "safety"
    elif pareto_class == "energy_tight":
        objective_preset = "energy_saver"
    elif pareto_class == "mpl_tight":
        objective_preset = "throughput"
    else:
        objective_preset = "balanced"

    # Create tool sequence that implements Pareto-optimal choice
    profile_weights = {p: 0.0 for p in ["BASE", "BOOST", "SAVER", "SAFE"]}
    profile_weights[chosen_profile] = 1.0

    target_sequence = [
        ToolCall(name="SET_OBJECTIVE_PRESET", args={"preset": objective_preset}),
        ToolCall(name="SET_ENERGY_PROFILE", args={"profile_mix": profile_weights}),
        ToolCall(name="SET_BACKEND", args={"backend": ctx.engine_type}),
    ]

    rationales = [
        f"Pareto-optimal preset for {pareto_class} configuration",
        f"Pareto-optimal profile {chosen_profile} (urgency={urgency})",
        f"Backend consistent with env",
    ]

    # Create econ/semantic summary
    econ_summary = EconSemanticDecisionSummary(
        chosen_profile=chosen_profile,
        objective_preset=objective_preset,
        pareto_classification=pareto_class,
        urgency_level=urgency,
        recommended_focus=f"{pareto_class}_optimization",
        semantic_priority_fraction=econ_signals.get("error_urgency", 0.0),
        data_coverage_score=datapack_signals.get("coverage_score", 0.5),
        wage_parity=econ_signals.get("wage_parity", 1.0),
    )

    return OrchestrationSample(
        context=ctx,
        context_features=_encode_ctx(ctx),
        target_tool_sequence=target_sequence,
        heuristic_rationale=rationales,
        metadata={"pareto_class": pareto_class, "urgency": urgency},
        econ_semantic_summary=econ_summary,
        source_type="pareto_optimal",
    )


def generate_semantic_consistency_sample(seed: int = None) -> OrchestrationSample:
    """
    Generate a sample that prioritizes semantic consistency.

    This teaches the transformer to maintain semantic coherence across
    ontology, task graph, and economic objectives.
    """
    if seed is not None:
        np.random.seed(seed)

    ctx = generate_synthetic_context(seed)

    # Simulate semantic metrics
    semantic_metrics = {
        "task_cluster_purity": np.random.uniform(0.6, 1.0),
        "concept_drift_score": np.random.uniform(0.0, 0.3),
        "label_conflict_rate": np.random.uniform(0.0, 0.2),
        "high_priority_task_fraction": np.random.uniform(0.0, 0.5),
        "critical_priority_task_fraction": np.random.uniform(0.0, 0.2),
        "safety_tag_fraction": np.random.uniform(0.0, 0.4),
        "energy_tag_fraction": np.random.uniform(0.0, 0.3),
        "consistency_score": np.random.uniform(0.7, 1.0),
    }

    # Determine profile based on semantic tags
    if semantic_metrics["safety_tag_fraction"] > 0.3:
        chosen_profile = "SAFE"
        objective_preset = "safety"
    elif semantic_metrics["energy_tag_fraction"] > 0.25:
        chosen_profile = "SAVER"
        objective_preset = "energy_saver"
    elif semantic_metrics["high_priority_task_fraction"] > 0.4:
        chosen_profile = "BOOST"
        objective_preset = "throughput"
    else:
        chosen_profile = "BASE"
        objective_preset = "balanced"

    # Consistency-preserving data mix
    if semantic_metrics["label_conflict_rate"] > 0.1:
        data_mix = {"real": 0.4, "synthetic": 0.5, "hybrid": 0.1}
        mix_rationale = "High label conflict -> increase synthetic for consistency"
    elif semantic_metrics["concept_drift_score"] > 0.2:
        data_mix = {"real": 0.3, "synthetic": 0.6, "hybrid": 0.1}
        mix_rationale = "High drift -> favor synthetic to stabilize"
    else:
        data_mix = {"real": 0.7, "synthetic": 0.2, "hybrid": 0.1}
        mix_rationale = "Good consistency -> favor real data"

    profile_weights = {p: 0.0 for p in ["BASE", "BOOST", "SAVER", "SAFE"]}
    profile_weights[chosen_profile] = 1.0

    target_sequence = [
        ToolCall(name="SET_OBJECTIVE_PRESET", args={"preset": objective_preset}),
        ToolCall(name="SET_ENERGY_PROFILE", args={"profile_mix": profile_weights}),
        ToolCall(name="SET_DATA_MIX", args={"data_mix": data_mix}),
    ]

    rationales = [
        f"Semantic consistency: {objective_preset} aligns with tag distribution",
        f"Profile {chosen_profile} matches semantic priorities",
        mix_rationale,
    ]

    # Derive urgency from semantic metrics
    if semantic_metrics["critical_priority_task_fraction"] > 0.15:
        urgency = "critical"
    elif semantic_metrics["high_priority_task_fraction"] > 0.3:
        urgency = "high"
    elif semantic_metrics["label_conflict_rate"] > 0.1:
        urgency = "moderate"
    else:
        urgency = "none"

    econ_summary = EconSemanticDecisionSummary(
        chosen_profile=chosen_profile,
        objective_preset=objective_preset,
        pareto_classification="balanced",  # Semantic doesn't focus on Pareto
        urgency_level=urgency,
        recommended_focus="semantic_consistency",
        semantic_priority_fraction=semantic_metrics["high_priority_task_fraction"],
        data_coverage_score=semantic_metrics["task_cluster_purity"],
        wage_parity=1.0,  # Not the focus here
    )

    return OrchestrationSample(
        context=ctx,
        context_features=_encode_ctx(ctx),
        target_tool_sequence=target_sequence,
        heuristic_rationale=rationales,
        metadata={"semantic_metrics": semantic_metrics},
        econ_semantic_summary=econ_summary,
        source_type="semantic_consistency",
    )


def build_curriculum_dataset(
    num_heuristic: int = 300,
    num_pareto: int = 400,
    num_semantic: int = 300,
) -> tuple:
    """
    Build training dataset with curriculum mixing.

    Returns:
        samples: List of OrchestrationSample
        stats: Dataset statistics
    """
    samples = []

    # Stage 1: Heuristic samples (simple rules)
    print(f"Generating {num_heuristic} heuristic samples...")
    heuristic_samples = build_training_dataset(num_samples=num_heuristic)
    samples.extend(heuristic_samples)

    # Stage 2: Pareto-optimal samples (economically rational)
    print(f"Generating {num_pareto} Pareto-optimal samples...")
    for i in range(num_pareto):
        samples.append(generate_pareto_optimal_sample(seed=i + 10000))

    # Stage 3: Semantic consistency samples
    print(f"Generating {num_semantic} semantic consistency samples...")
    for i in range(num_semantic):
        samples.append(generate_semantic_consistency_sample(seed=i + 20000))

    # Compute statistics
    stats = {
        "total_samples": len(samples),
        "heuristic_count": num_heuristic,
        "pareto_count": num_pareto,
        "semantic_count": num_semantic,
        "profile_distribution": defaultdict(int),
        "preset_distribution": defaultdict(int),
        "urgency_distribution": defaultdict(int),
        "pareto_distribution": defaultdict(int),
        "source_distribution": defaultdict(int),
    }

    for sample in samples:
        stats["source_distribution"][sample.source_type] += 1
        if sample.econ_semantic_summary:
            stats["profile_distribution"][sample.econ_semantic_summary.chosen_profile] += 1
            stats["preset_distribution"][sample.econ_semantic_summary.objective_preset] += 1
            stats["urgency_distribution"][sample.econ_semantic_summary.urgency_level] += 1
            stats["pareto_distribution"][sample.econ_semantic_summary.pareto_classification] += 1

    # Convert defaultdicts to regular dicts for JSON
    stats["profile_distribution"] = dict(stats["profile_distribution"])
    stats["preset_distribution"] = dict(stats["preset_distribution"])
    stats["urgency_distribution"] = dict(stats["urgency_distribution"])
    stats["pareto_distribution"] = dict(stats["pareto_distribution"])
    stats["source_distribution"] = dict(stats["source_distribution"])

    return samples, stats


def extract_auxiliary_labels(samples):
    """
    Extract auxiliary supervision labels from samples.

    Returns:
        profile_labels: (N,) profile indices
        preset_labels: (N,) preset indices
        urgency_labels: (N,) urgency indices
        pareto_labels: (N,) Pareto classification indices
        consistency_scores: (N,) consistency scores (regression)
    """
    N = len(samples)

    profile_labels = np.zeros(N, dtype=np.int64)
    preset_labels = np.zeros(N, dtype=np.int64)
    urgency_labels = np.zeros(N, dtype=np.int64)
    pareto_labels = np.zeros(N, dtype=np.int64)
    consistency_scores = np.ones(N, dtype=np.float32)  # Default high consistency

    for i, sample in enumerate(samples):
        if sample.econ_semantic_summary:
            summary = sample.econ_semantic_summary
            profile_labels[i] = PROFILE_TO_IDX.get(summary.chosen_profile, 0)
            preset_labels[i] = PRESET_TO_IDX.get(summary.objective_preset, 0)
            urgency_labels[i] = URGENCY_TO_IDX.get(summary.urgency_level, 0)
            pareto_labels[i] = PARETO_TO_IDX.get(summary.pareto_classification, 0)
            consistency_scores[i] = summary.data_coverage_score  # Use as proxy
        else:
            # Heuristic samples: derive from tool sequence
            for tc in sample.target_tool_sequence:
                if tc.name == "SET_ENERGY_PROFILE":
                    # Extract profile from args
                    profile_mix = tc.args.get("profile_mix", {})
                    for p, w in profile_mix.items():
                        if w > 0.5:
                            profile_labels[i] = PROFILE_TO_IDX.get(p, 0)
                            break
                elif tc.name == "SET_OBJECTIVE_PRESET":
                    preset = tc.args.get("preset", "balanced")
                    preset_labels[i] = PRESET_TO_IDX.get(preset, 0)

    return profile_labels, preset_labels, urgency_labels, pareto_labels, consistency_scores


# =============================================================================
# Training Loop with Multi-Head Loss
# =============================================================================

def train_epoch_multihead(
    model: OrchestrationTransformerWithAuxHeads,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    tool_criterion: nn.Module,
    aux_criterion: nn.Module,
    vocab_size: int = 128,
    aux_weight: float = 0.3,
) -> dict:
    """Train for one epoch with multi-head losses."""
    model.train()

    metrics = {
        "total_loss": 0.0,
        "tool_loss": 0.0,
        "profile_loss": 0.0,
        "preset_loss": 0.0,
        "urgency_loss": 0.0,
        "pareto_loss": 0.0,
        "tool_acc": 0.0,
        "profile_acc": 0.0,
        "preset_acc": 0.0,
        "urgency_acc": 0.0,
        "pareto_acc": 0.0,
    }

    total_samples = 0

    for batch in dataloader:
        (batch_ctx, batch_tools, batch_profile, batch_preset,
         batch_urgency, batch_pareto, batch_consistency) = batch

        batch_size = batch_ctx.shape[0]
        instr_tokens = torch.randint(1, vocab_size, (batch_size, 8))

        optimizer.zero_grad()

        # Forward
        tool_logits, arg_vec, aux_out = model(instr_tokens, batch_ctx)

        # Primary loss: first tool
        tool_loss = tool_criterion(tool_logits, batch_tools[:, 0])

        # Auxiliary losses
        profile_loss = aux_criterion(aux_out["profile_logits"], batch_profile)
        preset_loss = aux_criterion(aux_out["preset_logits"], batch_preset)
        urgency_loss = aux_criterion(aux_out["urgency_logits"], batch_urgency)
        pareto_loss = aux_criterion(aux_out["pareto_logits"], batch_pareto)

        # Consistency regression loss (MSE)
        consistency_loss = nn.functional.mse_loss(
            aux_out["consistency_score"], batch_consistency
        )

        # Combined loss
        aux_total = (profile_loss + preset_loss + urgency_loss + pareto_loss) / 4
        total_loss = tool_loss + aux_weight * aux_total + 0.1 * consistency_loss

        # Backward
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        metrics["total_loss"] += total_loss.item()
        metrics["tool_loss"] += tool_loss.item()
        metrics["profile_loss"] += profile_loss.item()
        metrics["preset_loss"] += preset_loss.item()
        metrics["urgency_loss"] += urgency_loss.item()
        metrics["pareto_loss"] += pareto_loss.item()

        # Accuracies
        with torch.no_grad():
            metrics["tool_acc"] += (tool_logits.argmax(-1) == batch_tools[:, 0]).sum().item()
            metrics["profile_acc"] += (aux_out["profile_logits"].argmax(-1) == batch_profile).sum().item()
            metrics["preset_acc"] += (aux_out["preset_logits"].argmax(-1) == batch_preset).sum().item()
            metrics["urgency_acc"] += (aux_out["urgency_logits"].argmax(-1) == batch_urgency).sum().item()
            metrics["pareto_acc"] += (aux_out["pareto_logits"].argmax(-1) == batch_pareto).sum().item()

        total_samples += batch_size

    # Average
    num_batches = len(dataloader)
    for key in ["total_loss", "tool_loss", "profile_loss", "preset_loss", "urgency_loss", "pareto_loss"]:
        metrics[key] /= num_batches
    for key in ["tool_acc", "profile_acc", "preset_acc", "urgency_acc", "pareto_acc"]:
        metrics[key] /= total_samples

    return metrics


def evaluate_multihead(
    model: OrchestrationTransformerWithAuxHeads,
    dataloader: DataLoader,
    tool_criterion: nn.Module,
    aux_criterion: nn.Module,
    vocab_size: int = 128,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()

    metrics = {
        "total_loss": 0.0,
        "tool_acc": 0.0,
        "profile_acc": 0.0,
        "preset_acc": 0.0,
        "urgency_acc": 0.0,
        "pareto_acc": 0.0,
    }

    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            (batch_ctx, batch_tools, batch_profile, batch_preset,
             batch_urgency, batch_pareto, batch_consistency) = batch

            batch_size = batch_ctx.shape[0]
            instr_tokens = torch.randint(1, vocab_size, (batch_size, 8))

            tool_logits, arg_vec, aux_out = model(instr_tokens, batch_ctx)

            # Losses
            tool_loss = tool_criterion(tool_logits, batch_tools[:, 0])
            profile_loss = aux_criterion(aux_out["profile_logits"], batch_profile)
            preset_loss = aux_criterion(aux_out["preset_logits"], batch_preset)
            aux_total = (profile_loss + preset_loss) / 2
            total_loss = tool_loss + 0.3 * aux_total

            metrics["total_loss"] += total_loss.item()

            # Accuracies
            metrics["tool_acc"] += (tool_logits.argmax(-1) == batch_tools[:, 0]).sum().item()
            metrics["profile_acc"] += (aux_out["profile_logits"].argmax(-1) == batch_profile).sum().item()
            metrics["preset_acc"] += (aux_out["preset_logits"].argmax(-1) == batch_preset).sum().item()
            metrics["urgency_acc"] += (aux_out["urgency_logits"].argmax(-1) == batch_urgency).sum().item()
            metrics["pareto_acc"] += (aux_out["pareto_logits"].argmax(-1) == batch_pareto).sum().item()

            total_samples += batch_size

    metrics["total_loss"] /= len(dataloader)
    for key in ["tool_acc", "profile_acc", "preset_acc", "urgency_acc", "pareto_acc"]:
        metrics[key] /= total_samples

    return metrics


def evaluate_semantic_consistency(
    model: OrchestrationTransformerWithAuxHeads,
    samples,
    vocab_size: int = 128,
) -> dict:
    """
    Evaluate semantic consistency of model predictions.

    Checks if:
    - High urgency -> SAFE profile
    - High safety tags -> safety preset
    - Energy tight Pareto -> SAVER profile
    """
    model.eval()

    consistency_checks = {
        "high_urgency_safe_profile": {"correct": 0, "total": 0},
        "safety_focused_pareto_safe_profile": {"correct": 0, "total": 0},
        "energy_tight_pareto_saver_profile": {"correct": 0, "total": 0},
    }

    with torch.no_grad():
        for sample in samples:
            if not sample.econ_semantic_summary:
                continue

            summary = sample.econ_semantic_summary
            ctx_tensor = torch.from_numpy(sample.context_features).float().unsqueeze(0)
            instr_tokens = torch.randint(1, vocab_size, (1, 8))

            _, _, aux_out = model(instr_tokens, ctx_tensor)
            pred_profile = IDX_TO_PROFILE[aux_out["profile_logits"].argmax(-1).item()]

            # Check 1: High urgency should predict SAFE
            if summary.urgency_level in ["high", "critical"]:
                consistency_checks["high_urgency_safe_profile"]["total"] += 1
                if pred_profile == "SAFE":
                    consistency_checks["high_urgency_safe_profile"]["correct"] += 1

            # Check 2: Safety-focused Pareto should predict SAFE
            if summary.pareto_classification == "safety_focused":
                consistency_checks["safety_focused_pareto_safe_profile"]["total"] += 1
                if pred_profile == "SAFE":
                    consistency_checks["safety_focused_pareto_safe_profile"]["correct"] += 1

            # Check 3: Energy-tight Pareto should predict SAVER
            if summary.pareto_classification == "energy_tight":
                consistency_checks["energy_tight_pareto_saver_profile"]["total"] += 1
                if pred_profile == "SAVER":
                    consistency_checks["energy_tight_pareto_saver_profile"]["correct"] += 1

    # Compute rates
    results = {}
    for check_name, data in consistency_checks.items():
        if data["total"] > 0:
            results[check_name] = {
                "rate": data["correct"] / data["total"],
                "correct": data["correct"],
                "total": data["total"],
            }
        else:
            results[check_name] = {"rate": 0.0, "correct": 0, "total": 0}

    return results


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Orchestration Transformer with V1 Curriculum"
    )
    parser.add_argument("--num-heuristic", type=int, default=300)
    parser.add_argument("--num-pareto", type=int, default=400)
    parser.add_argument("--num-semantic", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.3, help="Weight for auxiliary losses")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--ctx-dim", type=int, default=36)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="checkpoints/orchestrator_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("Orchestration Transformer V1 Curriculum Training")
    print("=" * 70)
    print(f"Curriculum: {args.num_heuristic} heuristic + {args.num_pareto} Pareto + {args.num_semantic} semantic")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Auxiliary loss weight: {args.aux_weight}")
    print("=" * 70)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate curriculum dataset
    print("\nBuilding curriculum dataset...")
    samples, dataset_stats = build_curriculum_dataset(
        num_heuristic=args.num_heuristic,
        num_pareto=args.num_pareto,
        num_semantic=args.num_semantic,
    )

    print(f"\nDataset statistics:")
    print(f"  Total: {dataset_stats['total_samples']}")
    print(f"  Source: {dataset_stats['source_distribution']}")
    print(f"  Profiles: {dataset_stats['profile_distribution']}")
    print(f"  Presets: {dataset_stats['preset_distribution']}")
    print(f"  Urgency: {dataset_stats['urgency_distribution']}")
    print(f"  Pareto: {dataset_stats['pareto_distribution']}")

    # Save dataset stats
    with open(save_dir / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    # Convert to tensors
    X, Y, tool_names = dataset_to_tensors(samples)
    profile_labels, preset_labels, urgency_labels, pareto_labels, consistency_scores = extract_auxiliary_labels(samples)

    # Adjust context dim
    actual_ctx_dim = X.shape[1]
    if actual_ctx_dim != args.ctx_dim:
        print(f"Adjusting ctx_dim from {args.ctx_dim} to {actual_ctx_dim}")
        args.ctx_dim = actual_ctx_dim

    # Train/val split
    num_val = int(len(samples) * args.val_split)
    num_train = len(samples) - num_val

    # Shuffle indices
    indices = np.random.permutation(len(samples))
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    # Create tensors
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).long()
    profile_tensor = torch.from_numpy(profile_labels).long()
    preset_tensor = torch.from_numpy(preset_labels).long()
    urgency_tensor = torch.from_numpy(urgency_labels).long()
    pareto_tensor = torch.from_numpy(pareto_labels).long()
    consistency_tensor = torch.from_numpy(consistency_scores).float()

    # Create datasets
    train_dataset = TensorDataset(
        X_tensor[train_idx], Y_tensor[train_idx],
        profile_tensor[train_idx], preset_tensor[train_idx],
        urgency_tensor[train_idx], pareto_tensor[train_idx],
        consistency_tensor[train_idx],
    )
    val_dataset = TensorDataset(
        X_tensor[val_idx], Y_tensor[val_idx],
        profile_tensor[val_idx], preset_tensor[val_idx],
        urgency_tensor[val_idx], pareto_tensor[val_idx],
        consistency_tensor[val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nTrain: {num_train}, Val: {num_val}")

    # Create model
    model = OrchestrationTransformerWithAuxHeads(
        vocab_size=args.vocab_size,
        hidden=args.hidden,
        ctx_dim=args.ctx_dim,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    tool_criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\nTraining...")
    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch_multihead(
            model, train_loader, optimizer, tool_criterion, aux_criterion,
            args.vocab_size, args.aux_weight
        )
        val_metrics = evaluate_multihead(
            model, val_loader, tool_criterion, aux_criterion, args.vocab_size
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Save best model
        composite_acc = (val_metrics["tool_acc"] + val_metrics["profile_acc"] + val_metrics["preset_acc"]) / 3
        if composite_acc > best_val_acc:
            best_val_acc = composite_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Tool Acc: {train_metrics['tool_acc']:.3f} | Val: {val_metrics['tool_acc']:.3f}")
            print(f"  Profile Acc: {train_metrics['profile_acc']:.3f} | Val: {val_metrics['profile_acc']:.3f}")
            print(f"  Preset Acc: {train_metrics['preset_acc']:.3f} | Val: {val_metrics['preset_acc']:.3f}")
            print(f"  Urgency Acc: {train_metrics['urgency_acc']:.3f} | Val: {val_metrics['urgency_acc']:.3f}")
            print(f"  Pareto Acc: {train_metrics['pareto_acc']:.3f} | Val: {val_metrics['pareto_acc']:.3f}")

    # Save final model
    torch.save(model.state_dict(), save_dir / "final_model.pt")

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save dataset
    save_dataset(samples, str(save_dir / "training_dataset.json"))

    # Evaluate semantic consistency
    print("\n" + "=" * 70)
    print("Semantic Consistency Evaluation")
    print("=" * 70)

    model.load_state_dict(torch.load(save_dir / "best_model.pt"))
    consistency_results = evaluate_semantic_consistency(model, samples, args.vocab_size)

    for check, result in consistency_results.items():
        print(f"{check}:")
        print(f"  Rate: {result['rate']:.3f} ({result['correct']}/{result['total']})")

    with open(save_dir / "semantic_consistency.json", "w") as f:
        json.dump(consistency_results, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best composite val accuracy: {best_val_acc:.3f}")
    print(f"Final tool accuracy: {history[-1]['val']['tool_acc']:.3f}")
    print(f"Final profile accuracy: {history[-1]['val']['profile_acc']:.3f}")
    print(f"Final preset accuracy: {history[-1]['val']['preset_acc']:.3f}")
    print(f"\nCheckpoint: {save_dir / 'best_model.pt'}")
    print(f"Dataset: {save_dir / 'training_dataset.json'}")
    print(f"History: {save_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
