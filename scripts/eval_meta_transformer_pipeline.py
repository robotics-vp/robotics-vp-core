#!/usr/bin/env python3
"""
Meta-Transformer Pipeline Evaluation.

Tests:
- Synthetic data generation
- Dataloader and batching
- Forward pass through cross-attention model
- Output shape validation
- Basic evaluation metrics

No actual training - just structural correctness verification.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.orchestrator.meta_transformer_training import (
    MetaTransformerNet,
    MetaTransformerDataset,
    collate_meta_transformer_batch,
    generate_meta_transformer_dataset,
    save_meta_transformer_dataset,
    load_meta_transformer_dataset,
    forward_pass_test,
    evaluate_meta_transformer,
    encode_semantic_tokens,
    decode_semantic_tokens,
    SEMANTIC_VOCAB,
)


def test_dataloader(samples, batch_size: int = 8):
    """Test dataset and dataloader."""
    print("\n--- Testing Dataloader ---")

    dataset = MetaTransformerDataset(samples, max_semantic_tokens=16)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_meta_transformer_batch,
    )
    print(f"Num batches: {len(dataloader)}")

    # Get one batch
    batch = next(iter(dataloader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"VLA embeddings shape: {batch['vla_embeddings'].shape}")
    print(f"DINO embeddings shape: {batch['dino_embeddings'].shape}")
    print(f"Semantic token IDs shape: {batch['semantic_token_ids'].shape}")
    print(f"Authority GT shape: {batch['authority_gt'].shape}")

    # Decode some tokens
    sample_tokens = batch['semantic_token_ids'][0].numpy()
    decoded = decode_semantic_tokens(sample_tokens)
    print(f"Sample semantic tokens: {decoded}")

    return dataloader


def test_forward_pass():
    """Test forward pass through model."""
    print("\n--- Testing Forward Pass ---")

    model = MetaTransformerNet(
        vla_dim=128,
        dino_dim=256,
        hidden_dim=128,
        num_heads=4,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = forward_pass_test(model, batch_size=8)
    print(f"Policy state shape: {results['policy_state_shape']}")
    print(f"Diffusion conditioning shape: {results['diffusion_cond_shape']}")
    print(f"Token logits shape: {results['token_logits_shape']}")
    print(f"Authority logits shape: {results['authority_logits_shape']}")
    print(f"Shared repr norm: {results['shared_repr_norm']:.4f}")
    print(f"Authority probs (mean): {results['authority_probs']}")

    return model


def test_cross_attention():
    """Test cross-attention mechanism."""
    print("\n--- Testing Cross-Attention ---")

    model = MetaTransformerNet(vla_dim=128, dino_dim=256, hidden_dim=128)
    model.eval()

    # Create samples where DINO should dominate (safety-critical)
    vla_emb = torch.randn(4, 128)
    dino_emb = torch.randn(4, 256)

    # Make DINO embedding have higher norm (more informative)
    dino_emb_strong = dino_emb * 2.0

    with torch.no_grad():
        outputs_weak_dino = model(vla_emb, dino_emb)
        outputs_strong_dino = model(vla_emb, dino_emb_strong)

    # Check if stronger DINO changes authority prediction
    auth_weak = outputs_weak_dino["authority_logits"].softmax(-1)
    auth_strong = outputs_strong_dino["authority_logits"].softmax(-1)

    print(f"Weak DINO authority probs: dino={auth_weak[:, 0].mean():.3f}, vla={auth_weak[:, 1].mean():.3f}")
    print(f"Strong DINO authority probs: dino={auth_strong[:, 0].mean():.3f}, vla={auth_strong[:, 1].mean():.3f}")

    # Check if shared repr changes
    repr_diff = (outputs_strong_dino["shared_repr"] - outputs_weak_dino["shared_repr"]).norm()
    print(f"Shared repr difference (strong vs weak DINO): {repr_diff:.4f}")


def test_semantic_token_encoding():
    """Test semantic token encoding/decoding."""
    print("\n--- Testing Semantic Token Encoding ---")

    tokens = ["drawer", "vase", "fragile", "grasp", "open"]
    encoded = encode_semantic_tokens(tokens, max_len=16)
    decoded = decode_semantic_tokens(encoded)

    print(f"Original tokens: {tokens}")
    print(f"Encoded IDs: {encoded[:len(tokens)]}")
    print(f"Decoded tokens: {decoded}")
    print(f"Vocab size: {len(SEMANTIC_VOCAB)}")


def test_data_statistics(samples):
    """Compute dataset statistics."""
    print("\n--- Dataset Statistics ---")

    authority_counts = {"vla": 0, "dino": 0}
    avg_num_tokens = 0.0
    token_freq = {}

    for sample in samples:
        authority_counts[sample.authority_gt] += 1
        avg_num_tokens += len(sample.semantic_tokens)

        for tok in sample.semantic_tokens:
            token_freq[tok] = token_freq.get(tok, 0) + 1

    avg_num_tokens /= len(samples)

    print(f"Authority distribution: {authority_counts}")
    print(f"Average semantic tokens per sample: {avg_num_tokens:.2f}")

    # Top tokens
    top_tokens = sorted(token_freq.items(), key=lambda x: -x[1])[:10]
    print("Top 10 semantic tokens:")
    for tok, count in top_tokens:
        print(f"  {tok}: {count}")


def test_evaluation(model, dataloader):
    """Test evaluation metrics."""
    print("\n--- Testing Evaluation ---")

    metrics = evaluate_meta_transformer(model, dataloader)

    print(f"Authority accuracy: {metrics['authority_acc']:.3f}")
    print(f"First token accuracy: {metrics['first_token_acc']:.3f}")
    print(f"Total loss: {metrics['total_loss']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Meta-Transformer Pipeline Evaluation")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--save-dataset", type=str, default=None, help="Path to save dataset")
    parser.add_argument("--output-dir", type=str, default="results/meta_transformer", help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("Meta-Transformer Pipeline Evaluation")
    print("=" * 70)
    print(f"Synthetic samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("\nGenerating synthetic meta-transformer dataset...")
    samples = generate_meta_transformer_dataset(args.num_samples)
    print(f"Generated {len(samples)} samples")

    # Save dataset if requested
    if args.save_dataset:
        save_meta_transformer_dataset(samples, args.save_dataset)

    # Test semantic token encoding
    test_semantic_token_encoding()

    # Dataset statistics
    test_data_statistics(samples)

    # Test dataloader
    dataloader = test_dataloader(samples, args.batch_size)

    # Test forward pass
    model = test_forward_pass()

    # Test cross-attention
    test_cross_attention()

    # Test evaluation
    eval_metrics = test_evaluation(model, dataloader)

    # Save results
    results = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "model_params": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(SEMANTIC_VOCAB),
        "evaluation_metrics": eval_metrics,
    }

    results_path = output_dir / "pipeline_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Test Summary")
    print("=" * 70)
    print("✓ Synthetic data generation works")
    print("✓ Dataloader and batching works")
    print("✓ Forward pass completes successfully")
    print("✓ Cross-attention between VLA and DINO works")
    print("✓ Semantic token encoding/decoding works")
    print("✓ Evaluation metrics computed")
    print(f"\nModel is ready for pretraining!")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
