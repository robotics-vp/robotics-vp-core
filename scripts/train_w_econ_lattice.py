#!/usr/bin/env python3
"""
Train Deep Lattice Economic Weighting Network

Distills a heuristic teacher weight function into a smooth, stable
deep lattice model with monotonicity constraints.

The teacher uses simple weighted combination:
    w_teacher = w_mpl * norm(ΔMPL) + w_error * norm(-Δerror) + w_ep * norm(ΔEP) + w_novelty * bandpass(novelty)

The student (lattice) learns this with proper monotonic structure.

Usage:
    python scripts/train_w_econ_lattice.py
    python scripts/train_w_econ_lattice.py --epochs 500 --n-samples 10000
"""
# NOTE: Experimental configuration;
# actual synthetic weighting is DL-driven (trust × w_econ).
# TODO: migrate to full PolicyProfile after demo.

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
from src.valuation.w_econ_lattice import WEconLattice, compute_heuristic_teacher_weight
from src.config.internal_profile import get_internal_experiment_profile


def generate_synthetic_training_data(n_samples=10000, n_bricks=5):
    """
    Generate synthetic training data with realistic distributions.

    Returns:
        data: dict of numpy arrays
    """
    # ΔMPL: mostly small positive, sometimes negative, occasionally large
    delta_mpl = np.random.exponential(0.3, n_samples) - 0.2
    delta_mpl = np.clip(delta_mpl, -0.5, 2.0)

    # Δerror: mostly small (near zero), sometimes positive (worse), sometimes negative (better)
    delta_error = np.random.normal(0, 0.15, n_samples)
    delta_error = np.clip(delta_error, -0.5, 0.5)

    # ΔEP: similar to ΔMPL but different distribution
    delta_ep = np.random.exponential(0.25, n_samples) - 0.15
    delta_ep = np.clip(delta_ep, -0.5, 2.0)

    # Novelty: uniform [0, 1] but biased towards middle
    novelty = np.random.beta(2, 2, n_samples)

    # Brick ID: categorical
    brick_id = np.random.randint(0, n_bricks, n_samples)

    # Compute teacher weights
    teacher_weights = np.array([
        compute_heuristic_teacher_weight(
            delta_mpl[i], delta_error[i], delta_ep[i], novelty[i], brick_id[i]
        )
        for i in range(n_samples)
    ])

    return {
        'delta_mpl': delta_mpl.astype(np.float32),
        'delta_error': delta_error.astype(np.float32),
        'delta_ep': delta_ep.astype(np.float32),
        'novelty': novelty.astype(np.float32),
        'brick_id': brick_id.astype(np.int64),
        'teacher_weight': teacher_weights.astype(np.float32),
    }


def load_real_episode_data(data_path, brick_manifest_path):
    """
    Load real episode metrics from physics rollouts.

    Returns:
        data: dict of numpy arrays with real ΔMPL, Δerror, ΔEP, novelty, brick_id
    """
    import json

    # Load physics rollouts
    if not os.path.exists(data_path):
        return None

    npz_data = np.load(data_path, allow_pickle=True)
    n_episodes = int(npz_data['n_episodes'])

    # Load brick manifest
    brick_manifest = None
    if os.path.exists(brick_manifest_path):
        with open(brick_manifest_path, 'r') as f:
            brick_manifest = json.load(f)

    # Build episode → brick_id mapping
    ep_to_brick = {}
    if brick_manifest:
        for brick in brick_manifest:
            brick_id_num = int(brick['brick_id'].split('_')[1])
            for ep_id in brick.get('episode_ids', []):
                ep_to_brick[ep_id] = brick_id_num

    # Extract episode-level metrics
    episode_metrics = []
    for ep in range(n_episodes):
        mpl = float(npz_data.get(f'ep_{ep}_metric_mpl', 0.0))
        error_rate = float(npz_data.get(f'ep_{ep}_metric_error_rate', 0.5))

        episode_metrics.append({
            'episode_id': ep,
            'mpl': mpl,
            'error_rate': error_rate,
            'brick_id': ep_to_brick.get(ep, 0),
        })

    if len(episode_metrics) == 0:
        return None

    # Compute global baselines
    global_mpl = np.mean([m['mpl'] for m in episode_metrics])
    global_error = np.mean([m['error_rate'] for m in episode_metrics])

    # Compute per-brick baselines (if enough data)
    brick_baselines = {}
    if brick_manifest:
        for brick in brick_manifest:
            brick_id_num = int(brick['brick_id'].split('_')[1])
            brick_baselines[brick_id_num] = {
                'mean_mpl': brick['impact_profile']['cluster_stats']['mean_mpl'],
                'mean_error': brick['impact_profile']['cluster_stats']['mean_error_rate'],
            }

    # Compute deltas relative to brick or global baseline
    delta_mpls = []
    delta_errors = []
    delta_eps = []
    novelties = []
    brick_ids = []

    for m in episode_metrics:
        brick_id = m['brick_id']

        # Use brick baseline if available, else global
        if brick_id in brick_baselines:
            baseline_mpl = brick_baselines[brick_id]['mean_mpl']
            baseline_error = brick_baselines[brick_id]['mean_error']
        else:
            baseline_mpl = global_mpl
            baseline_error = global_error

        # ΔMPL = episode MPL - baseline MPL
        delta_mpl = m['mpl'] - baseline_mpl

        # Δerror = episode error - baseline error (positive = worse)
        delta_error = m['error_rate'] - baseline_error

        # ΔEP: energy productivity
        # For now, approximate as MPL-weighted (no direct energy data)
        # ΔEP ~ ΔMPL scaled by some factor (assuming constant energy per unit)
        delta_ep = delta_mpl * 0.8  # Proxy: EP correlates with MPL

        # Novelty: how different is this episode from brick mean?
        # Use distance in MPL/error space normalized
        mpl_dev = abs(m['mpl'] - baseline_mpl) / (global_mpl + 1e-6)
        error_dev = abs(m['error_rate'] - baseline_error) / (global_error + 1e-6)
        novelty = min(1.0, np.sqrt(mpl_dev**2 + error_dev**2))

        delta_mpls.append(delta_mpl)
        delta_errors.append(delta_error)
        delta_eps.append(delta_ep)
        novelties.append(novelty)
        brick_ids.append(brick_id)

    # Normalize deltas to fit calibrator ranges
    # ΔMPL: calibrator expects [-0.5, 2.0], scale actual range to fit
    delta_mpls = np.array(delta_mpls)
    delta_errors = np.array(delta_errors)
    delta_eps = np.array(delta_eps)

    # Scale to calibrator ranges while preserving relative ordering
    # ΔMPL: map actual range to [-0.5, 2.0]
    mpl_min, mpl_max = delta_mpls.min(), delta_mpls.max()
    if mpl_max - mpl_min > 0:
        delta_mpls = -0.5 + (delta_mpls - mpl_min) / (mpl_max - mpl_min) * 2.5
    else:
        delta_mpls = np.zeros_like(delta_mpls)

    # Δerror: map to [-0.5, 0.5]
    err_min, err_max = delta_errors.min(), delta_errors.max()
    if err_max - err_min > 0:
        delta_errors = -0.5 + (delta_errors - err_min) / (err_max - err_min) * 1.0
    else:
        delta_errors = np.zeros_like(delta_errors)

    # ΔEP: map to [-0.5, 2.0]
    ep_min, ep_max = delta_eps.min(), delta_eps.max()
    if ep_max - ep_min > 0:
        delta_eps = -0.5 + (delta_eps - ep_min) / (ep_max - ep_min) * 2.5
    else:
        delta_eps = np.zeros_like(delta_eps)

    # Convert to arrays
    novelties = np.array(novelties)
    brick_ids = np.array(brick_ids)

    # Compute teacher weights using normalized values
    teacher_weights = np.array([
        compute_heuristic_teacher_weight(
            delta_mpls[i], delta_errors[i], delta_eps[i], novelties[i], brick_ids[i]
        )
        for i in range(len(episode_metrics))
    ])

    return {
        'delta_mpl': delta_mpls.astype(np.float32),
        'delta_error': delta_errors.astype(np.float32),
        'delta_ep': delta_eps.astype(np.float32),
        'novelty': novelties.astype(np.float32),
        'brick_id': brick_ids.astype(np.int64),
        'teacher_weight': teacher_weights.astype(np.float32),
    }


def train_lattice(model, train_data, val_data, n_epochs=200, batch_size=256, lr=1e-3, device='cpu'):
    """
    Train the lattice model to match teacher weights.

    Args:
        model: WEconLattice instance
        train_data: dict of training arrays
        val_data: dict of validation arrays
        n_epochs: number of epochs
        batch_size: batch size
        lr: learning rate
        device: cpu or cuda

    Returns:
        history: dict of training metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_train = len(train_data['delta_mpl'])
    n_val = len(val_data['delta_mpl'])

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
    }

    print(f"\nTraining lattice on {n_train} samples, validating on {n_val} samples")

    for epoch in range(n_epochs):
        # Shuffle training data
        indices = np.random.permutation(n_train)

        epoch_loss = 0.0
        n_batches = 0

        # Training
        model.train()
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < 2:
                continue

            # Get batch
            delta_mpl = torch.FloatTensor(train_data['delta_mpl'][batch_idx]).to(device)
            delta_error = torch.FloatTensor(train_data['delta_error'][batch_idx]).to(device)
            delta_ep = torch.FloatTensor(train_data['delta_ep'][batch_idx]).to(device)
            novelty = torch.FloatTensor(train_data['novelty'][batch_idx]).to(device)
            brick_id = torch.LongTensor(train_data['brick_id'][batch_idx]).to(device)
            target = torch.FloatTensor(train_data['teacher_weight'][batch_idx]).to(device)

            # Forward
            pred = model(delta_mpl, delta_error, delta_ep, novelty, brick_id)
            loss = criterion(pred, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_delta_mpl = torch.FloatTensor(val_data['delta_mpl']).to(device)
            val_delta_error = torch.FloatTensor(val_data['delta_error']).to(device)
            val_delta_ep = torch.FloatTensor(val_data['delta_ep']).to(device)
            val_novelty = torch.FloatTensor(val_data['novelty']).to(device)
            val_brick_id = torch.LongTensor(val_data['brick_id']).to(device)
            val_target = torch.FloatTensor(val_data['teacher_weight']).to(device)

            val_pred = model(val_delta_mpl, val_delta_error, val_delta_ep, val_novelty, val_brick_id)
            val_loss = criterion(val_pred, val_target).item()
            val_mae = torch.abs(val_pred - val_target).mean().item()

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_MAE={val_mae:.4f}")

    return history


def evaluate_monotonicity(model, device='cpu'):
    """
    Test that the model respects monotonicity constraints.

    Returns:
        results: dict of monotonicity test results
    """
    results = {}

    # Test ΔMPL monotonicity: increasing ΔMPL should increase w_econ
    delta_mpl_test = torch.linspace(-0.5, 2.0, 100).to(device)
    fixed_error = torch.zeros(100).to(device)
    fixed_ep = torch.zeros(100).to(device)
    fixed_novelty = torch.full((100,), 0.5).to(device)
    fixed_brick = torch.zeros(100, dtype=torch.long).to(device)

    with torch.no_grad():
        w_mpl = model(delta_mpl_test, fixed_error, fixed_ep, fixed_novelty, fixed_brick)

    # Check if monotonically increasing
    diffs_mpl = (w_mpl[1:] - w_mpl[:-1]).cpu().numpy()
    mpl_monotone = (diffs_mpl >= -0.01).all()  # Allow small numerical errors
    results['delta_mpl_monotone'] = bool(mpl_monotone)
    results['delta_mpl_violations'] = int((diffs_mpl < -0.01).sum())

    # Test -Δerror monotonicity
    fixed_mpl = torch.zeros(100).to(device)
    delta_error_test = torch.linspace(-0.5, 0.5, 100).to(device)  # Note: -error should increase w

    with torch.no_grad():
        w_err = model(fixed_mpl, delta_error_test, fixed_ep, fixed_novelty, fixed_brick)

    # As delta_error increases (worse), w should decrease → as -delta_error increases, w increases
    # So we expect w to decrease as delta_error goes from -0.5 to 0.5
    diffs_err = (w_err[1:] - w_err[:-1]).cpu().numpy()
    err_monotone = (diffs_err <= 0.01).all()  # Should be monotone decreasing
    results['neg_delta_error_monotone'] = bool(err_monotone)
    results['neg_delta_error_violations'] = int((diffs_err > 0.01).sum())

    # Test ΔEP monotonicity
    delta_ep_test = torch.linspace(-0.5, 2.0, 100).to(device)

    with torch.no_grad():
        w_ep = model(fixed_mpl, fixed_error, delta_ep_test, fixed_novelty, fixed_brick)

    diffs_ep = (w_ep[1:] - w_ep[:-1]).cpu().numpy()
    ep_monotone = (diffs_ep >= -0.01).all()
    results['delta_ep_monotone'] = bool(ep_monotone)
    results['delta_ep_violations'] = int((diffs_ep < -0.01).sum())

    return results


def main():
    # Load experiment profile for defaults
    profile = get_internal_experiment_profile("default")

    parser = argparse.ArgumentParser(description='Train w_econ lattice network')
    parser.add_argument('--n-samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--n-bricks', type=int, default=profile['lattice_n_bricks'], help='Number of bricks')
    parser.add_argument('--epochs', type=int, default=profile['lattice_epochs'])
    parser.add_argument('--batch-size', type=int, default=profile['ab_test_batch_size'])
    parser.add_argument('--lr', type=float, default=profile['ab_test_lr'])
    parser.add_argument('--n-keypoints', type=int, default=profile['lattice_n_keypoints'],
                        help='Keypoints per calibrator')
    parser.add_argument('--hidden-dim', type=int, default=profile['lattice_hidden_dim'],
                        help='Lattice MLP hidden dim')
    parser.add_argument('--output', type=str, default=profile['w_econ_lattice_path'])
    parser.add_argument('--use-real-data', action='store_true', default=True,
                        help='Use real episode metrics (default: True)')
    parser.add_argument('--data-path', type=str, default=profile['real_data_path'],
                        help='Path to physics rollouts data')
    parser.add_argument('--brick-manifest', type=str, default=profile['brick_manifest_path'],
                        help='Path to brick manifest')
    parser.add_argument('--min-real-samples', type=int, default=20,
                        help='Minimum real samples before augmenting with synthetic')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("TRAINING DEEP LATTICE ECONOMIC WEIGHTING NETWORK")
    print("="*70)
    print("Teacher: heuristic weighted combination")
    print("Student: deep lattice with monotonic calibrators")
    print()

    # Try loading real episode data first
    real_data = None
    n_real = 0
    data_source = "synthetic"

    if args.use_real_data:
        print(f"Attempting to load real episode metrics from {args.data_path}...")
        real_data = load_real_episode_data(args.data_path, args.brick_manifest)

        if real_data is not None:
            n_real = len(real_data['delta_mpl'])
            print(f"  Loaded {n_real} real episode samples")

            # Show real data statistics
            print(f"\nReal episode data statistics:")
            print(f"  ΔMPL: mean={real_data['delta_mpl'].mean():.4f}, "
                  f"std={real_data['delta_mpl'].std():.4f}, "
                  f"range=[{real_data['delta_mpl'].min():.4f}, {real_data['delta_mpl'].max():.4f}]")
            print(f"  Δerror: mean={real_data['delta_error'].mean():.4f}, "
                  f"std={real_data['delta_error'].std():.4f}, "
                  f"range=[{real_data['delta_error'].min():.4f}, {real_data['delta_error'].max():.4f}]")
            print(f"  ΔEP: mean={real_data['delta_ep'].mean():.4f}, "
                  f"std={real_data['delta_ep'].std():.4f}")
            print(f"  Novelty: mean={real_data['novelty'].mean():.4f}, "
                  f"std={real_data['novelty'].std():.4f}")
            print(f"  Brick IDs: unique={np.unique(real_data['brick_id']).tolist()}")

            data_source = "real"
        else:
            print(f"  Could not load real data from {args.data_path}")

    # Determine number of bricks from data
    if real_data is not None:
        n_bricks = max(args.n_bricks, int(real_data['brick_id'].max()) + 1)
    else:
        n_bricks = args.n_bricks

    # Augment with synthetic data if needed
    if n_real < args.min_real_samples:
        n_synthetic = args.n_samples - n_real
        print(f"\nGenerating {n_synthetic} synthetic training samples to augment...")
        synthetic_data = generate_synthetic_training_data(n_synthetic, n_bricks)

        if real_data is not None:
            # Combine real + synthetic
            all_data = {
                k: np.concatenate([real_data[k], synthetic_data[k]], axis=0)
                for k in real_data.keys()
            }
            data_source = f"real({n_real})+synthetic({n_synthetic})"
        else:
            all_data = synthetic_data
            data_source = "synthetic"
    else:
        # Use only real data
        all_data = real_data
        data_source = f"real({n_real})"

    total_samples = len(all_data['delta_mpl'])
    print(f"\nData source: {data_source}")
    print(f"Total samples: {total_samples}")

    # Split train/val (80/20)
    n_train = int(0.8 * total_samples)
    train_data = {k: v[:n_train] for k, v in all_data.items()}
    val_data = {k: v[n_train:] for k, v in all_data.items()}

    print(f"Training: {n_train}, Validation: {total_samples - n_train}")

    # Teacher weight statistics
    print(f"\nTeacher weight statistics:")
    print(f"  Mean: {all_data['teacher_weight'].mean():.4f}")
    print(f"  Std:  {all_data['teacher_weight'].std():.4f}")
    print(f"  Min:  {all_data['teacher_weight'].min():.4f}")
    print(f"  Max:  {all_data['teacher_weight'].max():.4f}")

    # Create model
    print(f"\nCreating lattice model...")
    model = WEconLattice(
        n_bricks=n_bricks,
        brick_emb_dim=8,
        n_keypoints=args.n_keypoints,
        hidden_dim=args.hidden_dim,
        objective_dim=4,
    ).to(device)

    print(f"  N keypoints: {args.n_keypoints}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  N bricks: {n_bricks}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = train_lattice(
        model, train_data, val_data,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    # Evaluate monotonicity
    print("\nChecking monotonicity constraints...")
    mono_results = evaluate_monotonicity(model, device)
    print(f"  ΔMPL monotone: {mono_results['delta_mpl_monotone']} "
          f"(violations: {mono_results['delta_mpl_violations']})")
    print(f"  -Δerror monotone: {mono_results['neg_delta_error_monotone']} "
          f"(violations: {mono_results['neg_delta_error_violations']})")
    print(f"  ΔEP monotone: {mono_results['delta_ep_monotone']} "
          f"(violations: {mono_results['delta_ep_violations']})")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_delta_mpl = torch.FloatTensor(val_data['delta_mpl']).to(device)
        val_delta_error = torch.FloatTensor(val_data['delta_error']).to(device)
        val_delta_ep = torch.FloatTensor(val_data['delta_ep']).to(device)
        val_novelty = torch.FloatTensor(val_data['novelty']).to(device)
        val_brick_id = torch.LongTensor(val_data['brick_id']).to(device)
        val_target = torch.FloatTensor(val_data['teacher_weight']).to(device)

        val_pred = model(val_delta_mpl, val_delta_error, val_delta_ep, val_novelty, val_brick_id)

    # Metrics
    mse = ((val_pred - val_target) ** 2).mean().item()
    mae = torch.abs(val_pred - val_target).mean().item()
    corr = np.corrcoef(val_pred.cpu().numpy(), val_target.cpu().numpy())[0, 1]

    print(f"\nFinal evaluation:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Correlation: {corr:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_bricks': n_bricks,
        'brick_emb_dim': 8,
        'n_keypoints': args.n_keypoints,
        'hidden_dim': args.hidden_dim,
        'objective_dim': 4,
        'final_mse': mse,
        'final_mae': mae,
        'correlation': corr,
        'monotonicity': mono_results,
        'history': history,
        'config': vars(args),
        'data_source': data_source,
        'n_real_samples': n_real,
        'total_samples': total_samples,
    }, args.output)
    print(f"\nSaved model to {args.output}")

    # Save report
    report = {
        'final_mse': mse,
        'final_mae': mae,
        'correlation': corr,
        'monotonicity': mono_results,
        'teacher_weight_mean': float(all_data['teacher_weight'].mean()),
        'teacher_weight_std': float(all_data['teacher_weight'].std()),
        'data_source': data_source,
        'n_real_samples': n_real,
        'total_samples': total_samples,
        'config': vars(args),
    }

    os.makedirs('results', exist_ok=True)
    with open('results/w_econ_lattice_training.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to results/w_econ_lattice_training.json")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"Deep lattice economic weighting network trained successfully")
    print(f"Ready for integration into offline RL with trust + econ weights")


if __name__ == '__main__':
    main()
