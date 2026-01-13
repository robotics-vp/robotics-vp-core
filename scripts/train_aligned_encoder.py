#!/usr/bin/env python3
"""
Train Aligned Visual Encoder via Distillation (Phase A.5)

Trains student encoder to align with pretrained teacher representations.
Creates canonical visual backbone (z_V) for physics environment.

Usage:
    python scripts/train_aligned_encoder.py --config configs/dishwashing_physics_aligned.yaml
    python scripts/train_aligned_encoder.py --config configs/dishwashing_physics_aligned.yaml --epochs 100 --episodes 200
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Regality wrapper
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from src.training.wrap_training_entrypoint import regal_training


class VideoFrameDataset(Dataset):
    """
    Dataset of video observations collected from physics environment.

    Each item is a (T, C, H, W) video tensor from one episode.
    """
    def __init__(self, videos):
        """
        Args:
            videos: List of numpy arrays, each shape (T, C, H, W)
        """
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        return torch.FloatTensor(video)


def collect_video_dataset(config_path, n_episodes=100):
    """
    Collect video observations from physics environment.

    Args:
        config_path: Path to config file
        n_episodes: Number of episodes to collect

    Returns:
        List of video arrays, each shape (T, C, H, W)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env_config = config['env']
    env_type = env_config.get('type', 'dishwashing')

    if env_type == 'dishwashing_physics':
        from src.envs.physics.dishwashing_physics_env import DishwashingPhysicsEnv
        env = DishwashingPhysicsEnv(
            frames=env_config.get('frames', 4),
            image_size=tuple(env_config.get('image_size', [64, 64])),
            max_steps=env_config.get('max_steps', 60),
            headless=env_config.get('headless', True),
            randomize_dishes=env_config.get('randomize_dishes', True),
            camera_jitter=env_config.get('camera_jitter', 0.02),
            lighting_variation=env_config.get('lighting_variation', 0.1),
            slip_probability=env_config.get('slip_probability', 0.015),
            gripper_failure_rate=env_config.get('gripper_failure_rate', 0.008),
            max_speed_multiplier=env_config.get('max_speed_multiplier', 2.0),
            max_acceleration=env_config.get('max_acceleration', 1.0)
        )
    else:
        raise ValueError(f"Unknown env type: {env_type}")

    print(f"Collecting {n_episodes} episodes from {env_type} environment...")

    videos = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False

        # Collect all observations in episode
        episode_obs = [obs]
        while not done:
            action = np.random.rand(2)  # [speed, care]
            obs, info, done = env.step(action)
            episode_obs.append(obs)

        # Use final observation as representative frame
        # Shape: (T, C, H, W)
        videos.append(obs)

        if (ep + 1) % 20 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes")

    print(f"✅ Collected {len(videos)} video samples")
    return videos, config


def build_teacher_encoder(config):
    """Build frozen teacher encoder from config."""
    from src.encoders.teacher_adapter import TeacherAdapter

    encoder_config = config['encoder']
    latent_dim = encoder_config['latent_dim']
    teacher_config = encoder_config.get('teacher', {})

    teacher = TeacherAdapter(
        teacher_type=teacher_config.get('teacher_type', 'random'),
        latent_dim=latent_dim,
        freeze=True,
        input_channels=teacher_config.get('input_channels', 3),
    )

    print(f"[Teacher] Built {teacher_config.get('teacher_type', 'random')} encoder (frozen)")
    return teacher


def build_student_encoder(config):
    """Build trainable student encoder from config."""
    from src.encoders.student_video_encoder import AlignedVideoEncoder

    encoder_config = config['encoder']
    latent_dim = encoder_config['latent_dim']
    aligned_config = encoder_config.get('aligned', {})

    student = AlignedVideoEncoder(
        latent_dim=latent_dim,
        arch=aligned_config.get('arch', 'simple2dcnn'),
        input_channels=aligned_config.get('input_channels', 3),
        projection_dim=aligned_config.get('projection_dim', None),
        alignment_type=aligned_config.get('alignment_type', 'mse'),
        temperature=aligned_config.get('temperature', 0.1),
    )

    print(f"[Student] Built AlignedVideoEncoder (trainable)")
    print(f"  Architecture: {aligned_config.get('arch', 'simple2dcnn')}")
    print(f"  Alignment: {aligned_config.get('alignment_type', 'mse')}")
    print(f"  Parameters: {sum(p.numel() for p in student.parameters()):,}")

    return student


def temporal_consistency_loss(z_current, z_previous, margin=0.1):
    """
    Temporal consistency loss: nearby frames should have similar embeddings.

    Args:
        z_current: Current frame embeddings (B, D)
        z_previous: Previous frame embeddings (B, D)
        margin: Minimum similarity threshold

    Returns:
        loss: Temporal consistency loss
    """
    # L2 distance should be small
    dist = (z_current - z_previous).pow(2).sum(dim=-1).sqrt()
    # Hinge loss: penalize if distance > margin
    loss = torch.relu(dist - margin).mean()
    return loss


def train_aligned_encoder(
    config_path,
    n_episodes=100,
    n_epochs=50,
    batch_size=16,
    lr=1e-4,
    alpha_alignment=1.0,
    alpha_temporal=0.1,
    save_dir='checkpoints',
    device=None
):
    """
    Train aligned video encoder via distillation.

    Args:
        config_path: Path to config YAML
        n_episodes: Number of episodes for dataset
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        alpha_alignment: Weight for alignment loss
        alpha_temporal: Weight for temporal consistency loss
        save_dir: Directory to save checkpoint
        device: Torch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Collect dataset
    videos, config = collect_video_dataset(config_path, n_episodes)
    dataset = VideoFrameDataset(videos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build models
    teacher = build_teacher_encoder(config).to(device)
    student = build_student_encoder(config).to(device)

    # Freeze teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Alpha alignment: {alpha_alignment}")
    print(f"  Alpha temporal: {alpha_temporal}")
    print()

    train_log = []

    for epoch in range(n_epochs):
        student.train()
        epoch_losses = []
        epoch_align = []
        epoch_temporal = []
        epoch_cos_sim = []

        for batch_idx, videos in enumerate(dataloader):
            videos = videos.to(device)  # (B, T, C, H, W)
            B = videos.shape[0]

            # Forward pass
            z_student, loss_align, metrics = student.distillation_step(
                videos, teacher, alpha=alpha_alignment
            )

            # Temporal consistency (optional)
            # Shift videos by 1 frame and compute consistency
            if alpha_temporal > 0:
                # Create shifted version (simulate temporal neighbors)
                # Add small noise to current embeddings as "previous" frame proxy
                z_noisy = z_student + 0.01 * torch.randn_like(z_student)
                loss_temporal = temporal_consistency_loss(z_student, z_noisy)
            else:
                loss_temporal = torch.tensor(0.0, device=device)

            # Total loss
            total_loss = loss_align + alpha_temporal * loss_temporal

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track metrics
            epoch_losses.append(total_loss.item())
            epoch_align.append(metrics['align_loss'])
            epoch_temporal.append(loss_temporal.item())
            epoch_cos_sim.append(metrics['cos_sim'])

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_align = np.mean(epoch_align)
        avg_temporal = np.mean(epoch_temporal)
        avg_cos_sim = np.mean(epoch_cos_sim)

        train_log.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'align_loss': avg_align,
            'temporal_loss': avg_temporal,
            'cos_sim': avg_cos_sim,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Align: {avg_align:.4f} | "
                  f"Temp: {avg_temporal:.4f} | "
                  f"CosSim: {avg_cos_sim:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(save_dir, 'student_video_aligned.pt')
    checkpoint = {
        'student_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_log': train_log,
        'n_episodes': n_episodes,
        'n_epochs': n_epochs,
        'final_cos_sim': avg_cos_sim,
    }
    torch.save(checkpoint, checkpoint_path)

    print(f"\n✅ Saved checkpoint to {checkpoint_path}")
    print(f"   Final cosine similarity: {avg_cos_sim:.4f}")

    # Save training log
    log_path = os.path.join(save_dir, 'aligned_encoder_train.csv')
    import csv
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'align_loss', 'temporal_loss', 'cos_sim'])
        writer.writeheader()
        writer.writerows(train_log)
    print(f"   Training log saved to {log_path}")

    return student, checkpoint_path


@regal_training(env_type="workcell")
def main(runner=None):
    """Main entrypoint with regality wrapper."""
    if runner:
        runner.start_training()
    
    parser = argparse.ArgumentParser(description='Train aligned visual encoder')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dishwashing_physics_aligned.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--alpha-alignment',
        type=float,
        default=1.0,
        help='Weight for alignment loss'
    )
    parser.add_argument(
        '--alpha-temporal',
        type=float,
        default=0.1,
        help='Weight for temporal consistency loss'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoint'
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase A.5: Training Aligned Visual Encoder")
    print("="*60)

    student, checkpoint_path = train_aligned_encoder(
        config_path=args.config,
        n_episodes=args.episodes,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_alignment=args.alpha_alignment,
        alpha_temporal=args.alpha_temporal,
        save_dir=args.save_dir,
    )

    if runner:
        runner.update_step(args.epochs * 100)  # Approximate

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Validate aligned encoder:")
    print(f"   python scripts/validate_dmpl_novelty.py \\")
    print(f"     --config {args.config} \\")
    print(f"     --checkpoint {checkpoint_path} \\")
    print(f"     --episodes 50")
    print()
    print("2. Use in SAC training:")
    print(f"   python train_sac_v2.py {args.config}")
    print("="*60)


if __name__ == '__main__':
    main()
