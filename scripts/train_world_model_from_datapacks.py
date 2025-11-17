#!/usr/bin/env python3
"""
Train World Model from DataPacks.

Uses DataPackRepo as the data source with trust-weighted sampling.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.phase_b_integration import PhaseBDataPackIntegration
from src.valuation.episode_features import make_full_datapack_features


class DataPackWorldModelDataset(Dataset):
    """PyTorch dataset from DataPacks."""

    def __init__(self, datapacks, use_trust_weighting=True):
        """
        Args:
            datapacks: List of DataPackMeta objects
            use_trust_weighting: Weight samples by trust score
        """
        self.datapacks = datapacks
        self.use_trust_weighting = use_trust_weighting

        # Extract features and targets
        self.features = []
        self.targets = []
        self.weights = []

        for dp in datapacks:
            # Feature: condition + current metrics
            feat = make_full_datapack_features(dp)
            self.features.append(feat)

            # Target: next step metrics (delta_J, delta_mpl, delta_error, delta_ep)
            target = np.array([
                dp.attribution.delta_J,
                dp.attribution.delta_mpl,
                dp.attribution.delta_error,
                dp.attribution.delta_ep,
            ], dtype=np.float32)
            self.targets.append(target)

            # Weight: trust * w_econ
            if use_trust_weighting:
                weight = dp.attribution.trust_score * dp.attribution.w_econ
            else:
                weight = 1.0
            self.weights.append(weight)

        self.features = np.array(self.features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.weights = np.array(self.weights, dtype=np.float32)

        # Normalize weights
        if self.weights.sum() > 0:
            self.weights = self.weights / self.weights.sum() * len(self.weights)

    def __len__(self):
        return len(self.datapacks)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'weights': torch.FloatTensor([self.weights[idx]]),
        }


class SimpleWorldModel(nn.Module):
    """Simple MLP world model for datapack prediction."""

    def __init__(self, input_dim, output_dim=4, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.predictor = nn.Linear(hidden_dim, output_dim)

        # Uncertainty head
        self.uncertainty_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) features

        Returns:
            predictions: (batch, output_dim) predicted deltas
            uncertainties: (batch, output_dim) prediction uncertainties
        """
        features = self.encoder(x)
        predictions = self.predictor(features)
        uncertainties = torch.exp(self.uncertainty_head(features))  # Positive variance
        return predictions, uncertainties

    def predict_with_uncertainty(self, x):
        """Get predictions with uncertainty estimates."""
        with torch.no_grad():
            pred, unc = self.forward(x)
        return pred.cpu().numpy(), unc.cpu().numpy()


def weighted_mse_loss(predictions, targets, weights, uncertainties=None):
    """
    Weighted MSE loss with optional uncertainty.

    Args:
        predictions: (batch, output_dim) predictions
        targets: (batch, output_dim) targets
        weights: (batch, 1) sample weights
        uncertainties: (batch, output_dim) prediction variances (optional)

    Returns:
        loss: Scalar loss value
    """
    if uncertainties is not None:
        # Negative log likelihood with learned variance
        # loss = 0.5 * log(var) + 0.5 * (pred - target)^2 / var
        nll = 0.5 * torch.log(uncertainties) + 0.5 * (predictions - targets) ** 2 / uncertainties
        weighted_nll = nll * weights
        return weighted_nll.mean()
    else:
        # Simple weighted MSE
        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights
        return weighted_mse.mean()


def train_world_model_from_datapacks(
    repo: DataPackRepo,
    task_name: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    min_trust: float = 0.5,
    source_type: str = None,  # "real", "synthetic", or None for all
    use_trust_weighting: bool = True,
    output_dir: str = "results/wm_datapack_training",
    run_id: str = None
):
    """
    Train world model using DataPacks.

    Args:
        repo: DataPackRepo instance
        task_name: Task identifier
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        min_trust: Minimum trust threshold
        source_type: Filter by source type
        use_trust_weighting: Use trust scores for weighting
        output_dir: Output directory
        run_id: Training run ID

    Returns:
        Training results dict
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot train world model.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    if run_id is None:
        run_id = f"wm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 70)
    print("WORLD MODEL TRAINING FROM DATAPACKS")
    print("=" * 70)

    # Query datapacks
    print(f"\n1. Loading datapacks from repository...")
    datapacks = repo.query(
        task_name=task_name,
        min_trust=min_trust,
        source_type=source_type,
        limit=10000,
        sort_by="trust_score",
        sort_descending=True
    )

    if len(datapacks) < 10:
        print(f"Not enough datapacks ({len(datapacks)}). Need at least 10.")
        return {}

    print(f"   Found {len(datapacks)} datapacks")
    print(f"   Min trust: {min_trust}")
    print(f"   Source type: {source_type or 'all'}")

    # Mark as used for world model training
    print(f"\n2. Marking datapacks for training run: {run_id}")
    for dp in datapacks:
        if run_id not in dp.attribution.used_in_training_runs:
            dp.attribution.used_in_training_runs.append(run_id)
        dp.attribution.wm_role = "wm_train"

    # Create dataset
    print(f"\n3. Creating PyTorch dataset...")
    dataset = DataPackWorldModelDataset(datapacks, use_trust_weighting=use_trust_weighting)
    print(f"   Feature dim: {dataset.features.shape[1]}")
    print(f"   Target dim: {dataset.targets.shape[1]}")

    # Train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_indices = np.random.permutation(len(dataset))[:n_train]
    val_indices = np.random.permutation(len(dataset))[n_train:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Train samples: {n_train}")
    print(f"   Val samples: {n_val}")

    # Create model
    print(f"\n4. Creating world model...")
    input_dim = dataset.features.shape[1]
    output_dim = dataset.targets.shape[1]
    model = SimpleWorldModel(input_dim, output_dim)
    print(f"   Input dim: {input_dim}")
    print(f"   Output dim: {output_dim}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    print(f"\n5. Training for {epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features = batch['features']
            targets = batch['targets']
            weights = batch['weights']

            optimizer.zero_grad()

            predictions, uncertainties = model(features)
            loss = weighted_mse_loss(predictions, targets, weights, uncertainties)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        epoch_train_loss /= max(n_batches, 1)
        train_losses.append(epoch_train_loss)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                targets = batch['targets']
                weights = batch['weights']

                predictions, uncertainties = model(features)
                loss = weighted_mse_loss(predictions, targets, weights, uncertainties)

                epoch_val_loss += loss.item()
                n_val_batches += 1

        epoch_val_loss /= max(n_val_batches, 1)
        val_losses.append(epoch_val_loss)

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_id}_best.pt"))

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:4d}: train_loss={epoch_train_loss:.6f}, val_loss={epoch_val_loss:.6f}")

    print(f"\n6. Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f}")

    # Evaluate on test set
    print(f"\n7. Evaluating model...")
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch['features']
            targets = batch['targets']

            predictions, uncertainties = model(features)

            all_predictions.append(predictions.numpy())
            all_targets.append(targets.numpy())
            all_uncertainties.append(uncertainties.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)

    # Compute metrics
    mse_per_dim = np.mean((all_predictions - all_targets) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(all_predictions - all_targets), axis=0)

    target_names = ['delta_J', 'delta_mpl', 'delta_error', 'delta_ep']
    print(f"\n   Per-target metrics:")
    for i, name in enumerate(target_names):
        print(f"     {name}: MSE={mse_per_dim[i]:.6f}, MAE={mae_per_dim[i]:.6f}")

    # Save training history
    history = {
        'run_id': run_id,
        'task_name': task_name,
        'n_datapacks': len(datapacks),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'min_trust': min_trust,
        'source_type': source_type,
        'use_trust_weighting': use_trust_weighting,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_mse': mse_per_dim.tolist(),
        'final_mae': mae_per_dim.tolist(),
        'target_names': target_names,
    }

    history_path = os.path.join(output_dir, f"{run_id}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n8. Saved training history to {history_path}")

    # Generate synthetic datapacks
    print(f"\n9. Generating synthetic datapacks...")
    n_synthetic = min(100, len(datapacks) // 10)
    synthetic_packs = generate_synthetic_datapacks(
        model, datapacks[:n_synthetic], run_id
    )
    print(f"   Generated {len(synthetic_packs)} synthetic datapacks")

    # Store synthetic datapacks
    if synthetic_packs:
        repo.append_batch(synthetic_packs)
        print(f"   Appended to repository")

    return {
        'run_id': run_id,
        'n_datapacks': len(datapacks),
        'best_val_loss': best_val_loss,
        'final_mse': mse_per_dim.tolist(),
        'final_mae': mae_per_dim.tolist(),
        'history_path': history_path,
        'n_synthetic': len(synthetic_packs),
    }


def generate_synthetic_datapacks(model, source_datapacks, run_id, horizon=10):
    """
    Generate synthetic datapacks using world model.

    Args:
        model: Trained world model
        source_datapacks: Source datapacks to branch from
        run_id: Training run ID
        horizon: Prediction horizon

    Returns:
        List of synthetic DataPackMeta objects
    """
    from src.valuation.datapack_schema import (
        DataPackMeta,
        ConditionProfile,
        AttributionProfile,
        create_positive_datapack,
        create_negative_datapack,
    )

    synthetic = []

    for source_dp in source_datapacks:
        # Get source features
        source_features = make_full_datapack_features(source_dp)
        source_features_t = torch.FloatTensor(source_features).unsqueeze(0)

        # Predict outcomes
        with torch.no_grad():
            predictions, uncertainties = model(source_features_t)

        pred_np = predictions.squeeze().numpy()
        unc_np = uncertainties.squeeze().numpy()

        # Create synthetic datapack
        # Perturb based on predictions + noise
        new_delta_j = float(pred_np[0] + np.random.randn() * np.sqrt(unc_np[0]) * 0.1)
        new_delta_mpl = float(pred_np[1] + np.random.randn() * np.sqrt(unc_np[1]) * 0.1)
        new_delta_error = float(pred_np[2] + np.random.randn() * np.sqrt(unc_np[2]) * 0.1)
        new_delta_ep = float(pred_np[3] + np.random.randn() * np.sqrt(unc_np[3]) * 0.1)

        # Create new attribution
        new_attribution = AttributionProfile(
            delta_mpl=new_delta_mpl,
            delta_error=new_delta_error,
            delta_ep=new_delta_ep,
            delta_J=new_delta_j,
            trust_score=source_dp.attribution.trust_score * 0.9,  # Slightly lower trust
            w_econ=source_dp.attribution.w_econ,
            lambda_budget=source_dp.attribution.lambda_budget,
            source_type="synthetic",
            wm_model_id=run_id,
            wm_horizon_used=horizon,
            wm_branch_depth=1,
            wm_trust_over_horizon=[source_dp.attribution.trust_score * (0.9 ** i) for i in range(horizon)],
            wm_role="wm_synth_target",
        )

        # Create datapack based on delta_j
        if new_delta_j >= 0:
            synth_dp = create_positive_datapack(
                task_name=source_dp.task_name,
                condition=source_dp.condition,
                attribution=new_attribution,
                skill_trace=source_dp.skill_trace,
                semantic_tags=source_dp.semantic_tags + ["synthetic"],
                sima_annotation=source_dp.sima_annotation,
                episode_id=f"synth_{source_dp.episode_id}"
            )
        else:
            synth_dp = create_negative_datapack(
                task_name=source_dp.task_name,
                condition=source_dp.condition,
                attribution=new_attribution,
                skill_trace=source_dp.skill_trace,
                counterfactual_plan=source_dp.counterfactual_plan or {
                    'skills': [0, 1, 2, 3, 4, 5],
                    'source': 'world_model'
                },
                counterfactual_source="world_model",
                semantic_tags=source_dp.semantic_tags + ["synthetic"],
                sima_annotation=source_dp.sima_annotation,
                episode_id=f"synth_{source_dp.episode_id}"
            )

        synthetic.append(synth_dp)

    return synthetic


def main():
    parser = argparse.ArgumentParser(description='Train world model from DataPacks')
    parser.add_argument('--data-dir', type=str, default='data/datapacks',
                        help='DataPack repository directory')
    parser.add_argument('--task', type=str, default='drawer_vase',
                        help='Task name')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--min-trust', type=float, default=0.5,
                        help='Minimum trust threshold')
    parser.add_argument('--source-type', type=str, default=None,
                        choices=['real', 'synthetic', 'hybrid', None],
                        help='Filter by source type')
    parser.add_argument('--no-trust-weighting', action='store_true',
                        help='Disable trust-based sample weighting')
    parser.add_argument('--output-dir', type=str, default='results/wm_datapack_training',
                        help='Output directory')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Training run ID')
    parser.add_argument('--generate-synthetic', type=int, default=0,
                        help='Generate synthetic datapacks for testing first')

    args = parser.parse_args()

    # Load or create repository
    repo = DataPackRepo(base_dir=args.data_dir)

    # Generate synthetic data if requested
    if args.generate_synthetic > 0:
        print(f"Generating {args.generate_synthetic} synthetic datapacks...")
        from scripts.build_datapacks_from_episodes import generate_synthetic_episodes, build_datapacks_from_episodes
        from src.config.econ_params import EconParams

        episodes = generate_synthetic_episodes(args.generate_synthetic)
        econ = EconParams(
            price_per_unit=5.0,
            mpl_human=20.0,
            wage_human=18.0,
            energy_price_kWh=0.12,
            energy_Wh_per_attempt=5.0,
            max_error_rate_sla=0.10,
            damage_cost_per_error=50.0
        )
        datapacks = build_datapacks_from_episodes(episodes, econ)
        repo.append_batch(datapacks)
        print(f"Added {len(datapacks)} datapacks to repository")

    # Train world model
    results = train_world_model_from_datapacks(
        repo=repo,
        task_name=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_trust=args.min_trust,
        source_type=args.source_type,
        use_trust_weighting=not args.no_trust_weighting,
        output_dir=args.output_dir,
        run_id=args.run_id
    )

    if results:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Run ID: {results['run_id']}")
        print(f"Datapacks used: {results['n_datapacks']}")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Synthetic datapacks generated: {results['n_synthetic']}")
        print(f"History saved to: {results['history_path']}")


if __name__ == "__main__":
    main()
