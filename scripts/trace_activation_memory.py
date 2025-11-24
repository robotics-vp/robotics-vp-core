#!/usr/bin/env python3
"""
Trace activation memory footprint for Stage 6 models.
Runs dummy forward/backward passes and reports peak memory usage.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.gpu_env import get_gpu_memory_info, clear_gpu_cache
from src.vision.regnet_backbone import RegNetBackbone
from src.vision.spatial_rnn import SpatialRNN
from src.ontology.sima2_segmenter import SIMA2Segmenter
from src.rl.hydra_heads import HydraPolicy

def trace_model(name, model_fn, input_fn, device="cuda"):
    print(f"\n--- Tracing {name} ---")
    clear_gpu_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_mem = torch.cuda.memory_allocated()
    
    # Create model
    model = model_fn().to(device)
    model.train() # Ensure gradients are tracked
    
    model_mem = torch.cuda.memory_allocated()
    print(f"Model Weights: {(model_mem - start_mem) / 1024**2:.2f} MB")
    
    # Create input
    inputs = input_fn(device)
    
    # Forward
    outputs = model(inputs)
    
    fwd_mem = torch.cuda.memory_allocated()
    print(f"Forward Pass (Activations): {(fwd_mem - model_mem) / 1024**2:.2f} MB")
    
    # Backward
    if isinstance(outputs, dict):
        loss = sum([v.mean() for v in outputs.values() if isinstance(v, torch.Tensor)])
    elif isinstance(outputs, tuple):
        loss = sum([v.mean() for v in outputs if isinstance(v, torch.Tensor)])
    else:
        loss = outputs.mean()
        
    loss.backward()
    
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"Peak Memory: {peak_mem / 1024**2:.2f} MB")
    
    del model, inputs, outputs, loss
    clear_gpu_cache()
    
    return {
        "model_weights_mb": (model_mem - start_mem) / 1024**2,
        "activations_mb": (fwd_mem - model_mem) / 1024**2,
        "peak_mb": peak_mem / 1024**2
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping trace.")
        return

    device = "cuda"
    results = {}

    # 1. Vision Backbone
    def vision_input(dev):
        return torch.randn(4, 3, 224, 224).to(dev) # Batch 4
        
    results["vision"] = trace_model(
        "Vision Backbone (RegNet)",
        lambda: RegNetBackbone(use_checkpointing=True),
        vision_input,
        device
    )

    # 2. Spatial RNN
    def spatial_input(dev):
        # List of feature pyramids
        seq_len = 8
        feats = []
        for _ in range(seq_len):
            feats.append({
                "P3": np.random.randn(256).astype(np.float32),
                "P4": np.random.randn(256).astype(np.float32),
                "P5": np.random.randn(256).astype(np.float32)
            })
        return feats # SpatialRNN handles numpy inputs and converts to tensor internally
        
    # Note: SpatialRNN forward expects list of dicts (numpy), so we don't move input to device here
    # But trace_model expects input_fn to return something passed to model.
    # SpatialRNN.forward takes (sequence, initial_state).
    # We need to wrap it.
    
    class SpatialWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = SpatialRNN(hidden_dim=64, feature_dim=256, use_checkpointing=True)
        def forward(self, x):
            return self.rnn(x)
            
    results["spatial_rnn"] = trace_model(
        "Spatial RNN",
        lambda: SpatialWrapper(),
        lambda dev: spatial_input(dev), # dev ignored
        device
    )

    # Save results
    with open("results/activation_trace.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nTrace complete. Saved to results/activation_trace.json")

if __name__ == "__main__":
    main()
