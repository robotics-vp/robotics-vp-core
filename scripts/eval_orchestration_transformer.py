#!/usr/bin/env python3
"""
Evaluate Orchestration Transformer vs Heuristic Teacher.

Samples contexts, compares model suggestions vs heuristic,
and prints diffs to see where the model generalizes differently.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.orchestrator.orchestration_transformer import (
    OrchestrationTransformer,
    TOOL_NAMES,
    decode_tool,
)
from src.orchestrator.training_dataset import (
    generate_synthetic_context,
    generate_heuristic_tool_sequence,
    context_to_sample,
)


def load_model(checkpoint_path: str, hidden: int = 96, ctx_dim: int = 36, vocab_size: int = 128):
    """Load trained model from checkpoint."""
    model = OrchestrationTransformer(vocab_size=vocab_size, hidden=hidden, ctx_dim=ctx_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict_tool_sequence(model: OrchestrationTransformer, ctx_features: np.ndarray, vocab_size: int = 128):
    """
    Get model's tool prediction for given context.

    For simplicity, we only predict the first tool.
    In full implementation, we'd predict entire sequence.
    """
    ctx_tensor = torch.from_numpy(ctx_features).float().unsqueeze(0)

    # Pad/trim context if needed
    expected_dim = model.ctx_proj.in_features
    if ctx_tensor.shape[1] < expected_dim:
        pad = torch.zeros(1, expected_dim - ctx_tensor.shape[1])
        ctx_tensor = torch.cat([ctx_tensor, pad], dim=1)
    elif ctx_tensor.shape[1] > expected_dim:
        ctx_tensor = ctx_tensor[:, :expected_dim]

    # Dummy instruction tokens
    instr_tokens = torch.randint(1, vocab_size, (1, 8))

    with torch.no_grad():
        tool_logits, arg_vec = model(instr_tokens, ctx_tensor)

    predicted_tool = decode_tool(tool_logits[0])
    probs = torch.softmax(tool_logits[0], dim=-1).numpy()

    return {
        "predicted_tool": predicted_tool,
        "probabilities": {TOOL_NAMES[i]: float(probs[i]) for i in range(len(TOOL_NAMES))},
        "arg_vec": arg_vec[0].numpy().tolist(),
    }


def compare_model_vs_heuristic(model, ctx, vocab_size: int = 128):
    """
    Compare model prediction to heuristic teacher for a single context.

    Returns:
        dict with comparison results
    """
    sample = context_to_sample(ctx)

    # Heuristic prediction
    heuristic_decisions = generate_heuristic_tool_sequence(ctx)
    heuristic_first_tool = heuristic_decisions[0].tool if heuristic_decisions else "NONE"
    heuristic_rationale = heuristic_decisions[0].rationale if heuristic_decisions else "N/A"

    # Model prediction
    model_pred = predict_tool_sequence(model, sample.context_features, vocab_size)
    model_first_tool = model_pred["predicted_tool"]

    # Check agreement
    agree = model_first_tool == heuristic_first_tool

    return {
        "context": {
            "customer_segment": ctx.customer_segment,
            "energy_price_kWh": ctx.energy_price_kWh,
            "objective_vector": ctx.objective_vector,
            "mean_trust": ctx.mean_trust,
            "mean_delta_mpl": ctx.mean_delta_mpl,
            "mean_delta_error": ctx.mean_delta_error,
        },
        "heuristic": {
            "tool": heuristic_first_tool,
            "rationale": heuristic_rationale,
        },
        "model": {
            "tool": model_first_tool,
            "probabilities": model_pred["probabilities"],
        },
        "agree": agree,
    }


def print_comparison(comparison: dict, index: int):
    """Pretty print a single comparison."""
    ctx = comparison["context"]
    heur = comparison["heuristic"]
    model = comparison["model"]
    agree = comparison["agree"]

    print(f"\n{'='*60}")
    print(f"Sample {index+1}")
    print(f"{'='*60}")

    print(f"Context:")
    print(f"  Customer: {ctx['customer_segment']}")
    print(f"  Energy Price: ${ctx['energy_price_kWh']:.3f}/kWh")
    print(f"  Objective: {[round(w, 2) for w in ctx['objective_vector']]}")
    print(f"  Trust: {ctx['mean_trust']:.3f}")
    print(f"  ΔMPL: {ctx['mean_delta_mpl']:.2f}, ΔError: {ctx['mean_delta_error']:.3f}")

    print(f"\nHeuristic Teacher:")
    print(f"  Tool: {heur['tool']}")
    print(f"  Rationale: {heur['rationale']}")

    print(f"\nModel Prediction:")
    print(f"  Tool: {model['tool']}")
    print(f"  Top Probabilities:")
    sorted_probs = sorted(model["probabilities"].items(), key=lambda x: -x[1])
    for tool, prob in sorted_probs[:3]:
        marker = "*" if tool == model["tool"] else " "
        print(f"    {marker} {tool}: {prob:.3f}")

    if agree:
        print(f"\n  ✓ AGREE")
    else:
        print(f"\n  ✗ DISAGREE")
        print(f"    Model chose {model['tool']} instead of {heur['tool']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Orchestration Transformer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/orchestrator/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--hidden", type=int, default=96, help="Model hidden dimension")
    parser.add_argument("--ctx-dim", type=int, default=36, help="Context feature dimension")
    parser.add_argument("--vocab-size", type=int, default=128, help="Vocabulary size")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for synthetic contexts")
    parser.add_argument("--save-results", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("Orchestration Transformer Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\nERROR: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first:")
        print("  python -m scripts.train_orchestration_transformer")
        return

    # Load model
    print("\nLoading model...")
    try:
        model = load_model(args.checkpoint, args.hidden, args.ctx_dim, args.vocab_size)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying to infer context dimension from checkpoint...")
        # Try to load and get actual dimension
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "ctx_proj.weight" in checkpoint:
            actual_ctx_dim = checkpoint["ctx_proj.weight"].shape[1]
            print(f"Detected ctx_dim={actual_ctx_dim}")
            args.ctx_dim = actual_ctx_dim
            model = load_model(args.checkpoint, args.hidden, args.ctx_dim, args.vocab_size)
        else:
            raise e

    # Generate synthetic contexts
    print(f"\nGenerating {args.num_samples} synthetic contexts...")
    np.random.seed(args.seed)

    comparisons = []
    agreements = 0

    for i in range(args.num_samples):
        ctx = generate_synthetic_context(seed=args.seed + i)
        comparison = compare_model_vs_heuristic(model, ctx, args.vocab_size)
        comparisons.append(comparison)
        if comparison["agree"]:
            agreements += 1

        print_comparison(comparison, i)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    agreement_rate = agreements / args.num_samples * 100
    print(f"Agreement rate: {agreements}/{args.num_samples} ({agreement_rate:.1f}%)")

    # Analyze disagreements
    disagreements = [c for c in comparisons if not c["agree"]]
    if disagreements:
        print(f"\nDisagreement patterns:")
        tool_disagree = {}
        for d in disagreements:
            key = f"{d['heuristic']['tool']} -> {d['model']['tool']}"
            tool_disagree[key] = tool_disagree.get(key, 0) + 1

        for pattern, count in sorted(tool_disagree.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")

    # Save results if requested
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "num_samples": args.num_samples,
                    "agreement_rate": agreement_rate,
                    "comparisons": comparisons,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.save_results}")

    # Interpretability notes
    print("\n" + "=" * 60)
    print("INTERPRETABILITY NOTES")
    print("=" * 60)
    print("Where model generalizes differently from heuristic:")
    print("")

    if agreement_rate > 90:
        print("  High agreement (>90%): Model closely imitates heuristic")
        print("  This suggests supervised learning is working, but model")
        print("  hasn't learned to generalize beyond the teacher.")
    elif agreement_rate > 70:
        print("  Moderate agreement (70-90%): Model captures main patterns")
        print("  Some generalization is happening - check disagreement patterns")
        print("  to see if model is making sensible alternative choices.")
    else:
        print("  Low agreement (<70%): Model diverging from heuristic")
        print("  Either training needs more data/epochs, or model is")
        print("  finding different (possibly valid) strategies.")

    print("\nNext steps:")
    print("  1. Examine disagreement patterns for sensibility")
    print("  2. Add more heuristic rules for edge cases")
    print("  3. Increase training data diversity")
    print("  4. Eventually: replace heuristic with RL-based learning")


if __name__ == "__main__":
    main()
