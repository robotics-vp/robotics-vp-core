#!/usr/bin/env python3
"""
Evaluate Orchestration Transformer V1 Checkpoint.

Loads trained model and evaluates:
- Tool prediction accuracy by source type
- Profile/preset/urgency/Pareto prediction accuracy
- Semantic consistency (does SAFE get predicted when urgency is high?)
- Pareto alignment (does profile match Pareto frontier position?)

Integrates with existing analysis tools for comprehensive evaluation.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from src.orchestrator.orchestration_transformer import TOOL_NAMES

# Import from v1 curriculum script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_orchestration_transformer_v1_curriculum import (
    OrchestrationTransformerWithAuxHeads,
    PROFILE_TO_IDX, IDX_TO_PROFILE,
    PRESET_TO_IDX, IDX_TO_PRESET,
    URGENCY_TO_IDX, IDX_TO_URGENCY,
    PARETO_TO_IDX, IDX_TO_PARETO,
    extract_auxiliary_labels,
)

from src.orchestrator.training_dataset import (
    load_dataset,
    dataset_to_tensors,
    OrchestrationSample,
    EconSemanticDecisionSummary,
)
from src.orchestrator.context import OrchestratorContext
from src.orchestrator.toolspecs import ToolCall


def load_samples_from_json(path: str):
    """Load OrchestrationSample objects from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        # Reconstruct context
        ctx_dict = item["context"]
        ctx = OrchestratorContext(**ctx_dict)

        # Reconstruct tool sequence
        tools = [
            ToolCall(name=tc["name"], args=tc["args"])
            for tc in item["target_tool_sequence"]
        ]

        # Reconstruct econ/semantic summary if present
        econ_summary = None
        if "econ_semantic_summary" in item and item["econ_semantic_summary"]:
            econ_summary = EconSemanticDecisionSummary(**item["econ_semantic_summary"])

        sample = OrchestrationSample(
            context=ctx,
            context_features=np.array(item["context_features"], dtype=np.float32),
            target_tool_sequence=tools,
            heuristic_rationale=item["heuristic_rationale"],
            metadata=item.get("metadata", {}),
            econ_semantic_summary=econ_summary,
            source_type=item.get("source_type", "heuristic"),
        )
        samples.append(sample)

    return samples


def evaluate_by_source_type(
    model: OrchestrationTransformerWithAuxHeads,
    samples,
    vocab_size: int = 128,
) -> dict:
    """Evaluate accuracy by source type (heuristic, pareto_optimal, semantic_consistency)."""
    model.eval()

    results_by_source = defaultdict(lambda: {
        "tool_correct": 0,
        "profile_correct": 0,
        "preset_correct": 0,
        "urgency_correct": 0,
        "pareto_correct": 0,
        "total": 0,
    })

    # Extract ground truth labels
    X, Y, _ = dataset_to_tensors(samples)
    profile_labels, preset_labels, urgency_labels, pareto_labels, _ = extract_auxiliary_labels(samples)

    with torch.no_grad():
        for i, sample in enumerate(samples):
            ctx_tensor = torch.from_numpy(sample.context_features).float().unsqueeze(0)
            instr_tokens = torch.randint(1, vocab_size, (1, 8))

            tool_logits, _, aux_out = model(instr_tokens, ctx_tensor)

            source = sample.source_type

            # Tool prediction
            pred_tool = tool_logits.argmax(-1).item()
            if pred_tool == Y[i, 0]:
                results_by_source[source]["tool_correct"] += 1

            # Auxiliary predictions
            pred_profile = aux_out["profile_logits"].argmax(-1).item()
            pred_preset = aux_out["preset_logits"].argmax(-1).item()
            pred_urgency = aux_out["urgency_logits"].argmax(-1).item()
            pred_pareto = aux_out["pareto_logits"].argmax(-1).item()

            if pred_profile == profile_labels[i]:
                results_by_source[source]["profile_correct"] += 1
            if pred_preset == preset_labels[i]:
                results_by_source[source]["preset_correct"] += 1
            if pred_urgency == urgency_labels[i]:
                results_by_source[source]["urgency_correct"] += 1
            if pred_pareto == pareto_labels[i]:
                results_by_source[source]["pareto_correct"] += 1

            results_by_source[source]["total"] += 1

    # Compute percentages
    final_results = {}
    for source, data in results_by_source.items():
        total = data["total"]
        final_results[source] = {
            "total": total,
            "tool_acc": data["tool_correct"] / total,
            "profile_acc": data["profile_correct"] / total,
            "preset_acc": data["preset_correct"] / total,
            "urgency_acc": data["urgency_correct"] / total,
            "pareto_acc": data["pareto_correct"] / total,
        }

    return final_results


def evaluate_pareto_alignment(
    model: OrchestrationTransformerWithAuxHeads,
    samples,
    vocab_size: int = 128,
) -> dict:
    """
    Evaluate if model's profile predictions align with Pareto frontier.

    Expected alignments:
    - safety_focused -> SAFE
    - energy_tight -> SAVER
    - mpl_tight -> BOOST
    - balanced -> BASE or context-dependent
    """
    model.eval()

    alignments = {
        "safety_focused": {"expected": "SAFE", "correct": 0, "total": 0},
        "energy_tight": {"expected": "SAVER", "correct": 0, "total": 0},
        "mpl_tight": {"expected": "BOOST", "correct": 0, "total": 0},
        "balanced": {"expected": "BASE", "correct": 0, "total": 0},
    }

    with torch.no_grad():
        for sample in samples:
            if not sample.econ_semantic_summary:
                continue

            pareto_class = sample.econ_semantic_summary.pareto_classification
            if pareto_class not in alignments:
                continue

            ctx_tensor = torch.from_numpy(sample.context_features).float().unsqueeze(0)
            instr_tokens = torch.randint(1, vocab_size, (1, 8))

            _, _, aux_out = model(instr_tokens, ctx_tensor)
            pred_profile = IDX_TO_PROFILE[aux_out["profile_logits"].argmax(-1).item()]

            alignments[pareto_class]["total"] += 1
            if pred_profile == alignments[pareto_class]["expected"]:
                alignments[pareto_class]["correct"] += 1

    results = {}
    for pareto_class, data in alignments.items():
        if data["total"] > 0:
            results[pareto_class] = {
                "expected_profile": data["expected"],
                "alignment_rate": data["correct"] / data["total"],
                "correct": data["correct"],
                "total": data["total"],
            }

    return results


def evaluate_urgency_response(
    model: OrchestrationTransformerWithAuxHeads,
    samples,
    vocab_size: int = 128,
) -> dict:
    """
    Evaluate if model responds appropriately to urgency levels.

    Expected:
    - critical/high urgency -> SAFE profile
    - none/moderate urgency -> more flexibility
    """
    model.eval()

    results = defaultdict(lambda: {
        "safe_count": 0, "saver_count": 0, "boost_count": 0, "base_count": 0, "total": 0
    })

    with torch.no_grad():
        for sample in samples:
            if not sample.econ_semantic_summary:
                continue

            urgency = sample.econ_semantic_summary.urgency_level
            ctx_tensor = torch.from_numpy(sample.context_features).float().unsqueeze(0)
            instr_tokens = torch.randint(1, vocab_size, (1, 8))

            _, _, aux_out = model(instr_tokens, ctx_tensor)
            pred_profile = IDX_TO_PROFILE[aux_out["profile_logits"].argmax(-1).item()]

            results[urgency]["total"] += 1
            if pred_profile == "SAFE":
                results[urgency]["safe_count"] += 1
            elif pred_profile == "SAVER":
                results[urgency]["saver_count"] += 1
            elif pred_profile == "BOOST":
                results[urgency]["boost_count"] += 1
            else:
                results[urgency]["base_count"] += 1

    final_results = {}
    for urgency, data in results.items():
        total = data["total"]
        if total > 0:
            final_results[urgency] = {
                "total": total,
                "safe_rate": data["safe_count"] / total,
                "saver_rate": data["saver_count"] / total,
                "boost_rate": data["boost_count"] / total,
                "base_rate": data["base_count"] / total,
            }

    return final_results


def generate_evaluation_report(
    source_results: dict,
    pareto_results: dict,
    urgency_results: dict,
    output_path: str,
) -> None:
    """Generate comprehensive evaluation report."""
    report = {
        "summary": {
            "overall_tool_acc": 0.0,
            "overall_profile_acc": 0.0,
            "overall_preset_acc": 0.0,
            "overall_urgency_acc": 0.0,
            "overall_pareto_acc": 0.0,
        },
        "by_source": source_results,
        "pareto_alignment": pareto_results,
        "urgency_response": urgency_results,
    }

    # Compute overall metrics
    total_samples = sum(d["total"] for d in source_results.values())
    if total_samples > 0:
        for metric in ["tool_acc", "profile_acc", "preset_acc", "urgency_acc", "pareto_acc"]:
            weighted_sum = sum(
                d["total"] * d[metric] for d in source_results.values()
            )
            report["summary"][f"overall_{metric}"] = weighted_sum / total_samples

    # Compute Pareto alignment score
    pareto_total = sum(d["total"] for d in pareto_results.values())
    pareto_correct = sum(d["correct"] for d in pareto_results.values())
    report["summary"]["pareto_alignment_score"] = pareto_correct / pareto_total if pareto_total > 0 else 0.0

    # Compute urgency-appropriate response rate (critical/high -> SAFE)
    high_urgency_safe = 0
    high_urgency_total = 0
    for urg in ["critical", "high"]:
        if urg in urgency_results:
            high_urgency_safe += urgency_results[urg]["safe_rate"] * urgency_results[urg]["total"]
            high_urgency_total += urgency_results[urg]["total"]
    report["summary"]["high_urgency_safe_rate"] = high_urgency_safe / high_urgency_total if high_urgency_total > 0 else 0.0

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved evaluation report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Orchestration Transformer V1")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset JSON")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--ctx-dim", type=int, default=36)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()

    print("=" * 70)
    print("Orchestration Transformer V1 Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    samples = load_samples_from_json(args.dataset)
    print(f"Loaded {len(samples)} samples")

    # Get actual context dimension
    actual_ctx_dim = samples[0].context_features.shape[0]
    if actual_ctx_dim != args.ctx_dim:
        print(f"Adjusting ctx_dim from {args.ctx_dim} to {actual_ctx_dim}")
        args.ctx_dim = actual_ctx_dim

    # Load model
    print("\nLoading model...")
    model = OrchestrationTransformerWithAuxHeads(
        vocab_size=args.vocab_size,
        hidden=args.hidden,
        ctx_dim=args.ctx_dim,
    )
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate by source type
    print("\n--- Evaluation by Source Type ---")
    source_results = evaluate_by_source_type(model, samples, args.vocab_size)
    for source, metrics in source_results.items():
        print(f"\n{source} ({metrics['total']} samples):")
        print(f"  Tool Acc: {metrics['tool_acc']:.3f}")
        print(f"  Profile Acc: {metrics['profile_acc']:.3f}")
        print(f"  Preset Acc: {metrics['preset_acc']:.3f}")
        print(f"  Urgency Acc: {metrics['urgency_acc']:.3f}")
        print(f"  Pareto Acc: {metrics['pareto_acc']:.3f}")

    # Evaluate Pareto alignment
    print("\n--- Pareto Frontier Alignment ---")
    pareto_results = evaluate_pareto_alignment(model, samples, args.vocab_size)
    for pareto_class, data in pareto_results.items():
        print(f"{pareto_class} -> expected {data['expected_profile']}:")
        print(f"  Alignment rate: {data['alignment_rate']:.3f} ({data['correct']}/{data['total']})")

    # Evaluate urgency response
    print("\n--- Urgency Response Distribution ---")
    urgency_results = evaluate_urgency_response(model, samples, args.vocab_size)
    for urgency in ["none", "moderate", "high", "critical"]:
        if urgency in urgency_results:
            data = urgency_results[urgency]
            print(f"{urgency} ({data['total']} samples):")
            print(f"  SAFE: {data['safe_rate']:.2%}, SAVER: {data['saver_rate']:.2%}, "
                  f"BOOST: {data['boost_rate']:.2%}, BASE: {data['base_rate']:.2%}")

    # Generate report
    output_dir = args.output_dir or str(Path(args.checkpoint).parent)
    report_path = Path(output_dir) / "evaluation_report.json"
    generate_evaluation_report(source_results, pareto_results, urgency_results, str(report_path))

    # Summary
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    # Compute overall scores
    total_samples = sum(d["total"] for d in source_results.values())
    overall_tool_acc = sum(d["total"] * d["tool_acc"] for d in source_results.values()) / total_samples
    overall_profile_acc = sum(d["total"] * d["profile_acc"] for d in source_results.values()) / total_samples

    pareto_alignment = sum(d["correct"] for d in pareto_results.values()) / sum(d["total"] for d in pareto_results.values())

    high_urgency_safe = 0
    high_urgency_total = 0
    for urg in ["critical", "high"]:
        if urg in urgency_results:
            high_urgency_safe += urgency_results[urg]["safe_rate"] * urgency_results[urg]["total"]
            high_urgency_total += urgency_results[urg]["total"]
    high_urgency_safe_rate = high_urgency_safe / high_urgency_total if high_urgency_total > 0 else 0.0

    print(f"Overall Tool Accuracy: {overall_tool_acc:.3f}")
    print(f"Overall Profile Accuracy: {overall_profile_acc:.3f}")
    print(f"Pareto Alignment Score: {pareto_alignment:.3f}")
    print(f"High Urgency -> SAFE Rate: {high_urgency_safe_rate:.3f}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
