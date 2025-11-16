"""
VLA Transformer Training Infrastructure.

Trains the VLA planner on:
- Scripted baseline trajectories
- HRL trajectories
- SIMA-style demonstration trajectories
"""

import os
import json
import numpy as np
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .transformer_planner import VLATransformerPlanner, VLAInput, VLAPlan
from src.hrl.skills import SkillID


class VLADataset(Dataset if TORCH_AVAILABLE else object):
    """
    Dataset for VLA training.

    Each sample contains:
    - Instruction text
    - Optional visual/state features
    - Ground truth skill sequence
    - Ground truth skill parameters
    """

    def __init__(self, trajectories, tokenizer, max_seq_len=32):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # Tokenize instruction
        token_ids = self.tokenizer.encode(traj['instruction'], self.max_seq_len)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Ground truth skill sequence
        gt_skills = torch.tensor(traj['skill_sequence'], dtype=torch.long)

        # Ground truth parameters (if available)
        if 'skill_params' in traj:
            gt_params = torch.tensor(traj['skill_params'], dtype=torch.float32)
        else:
            gt_params = torch.zeros(len(traj['skill_sequence']), 5)

        # Optional features
        sample = {
            'token_ids': token_ids,
            'gt_skills': gt_skills,
            'gt_params': gt_params,
        }

        if 'z_v' in traj:
            sample['z_v'] = torch.tensor(traj['z_v'], dtype=torch.float32)

        if 'state' in traj:
            sample['state'] = torch.tensor(traj['state'], dtype=torch.float32)

        if 'risk_map' in traj:
            sample['risk_map'] = torch.tensor(traj['risk_map'], dtype=torch.float32)

        if 'affordance_map' in traj:
            sample['affordance_map'] = torch.tensor(traj['affordance_map'], dtype=torch.float32)

        return sample


def collate_vla_batch(batch):
    """
    Custom collate function for variable-length skill sequences.

    Args:
        batch: List of samples from VLADataset

    Returns:
        collated: dict with batched tensors
    """
    # Stack fixed-size tensors
    token_ids = torch.stack([s['token_ids'] for s in batch])

    # Pad skill sequences to max length in batch
    max_skill_len = max(len(s['gt_skills']) for s in batch)

    gt_skills = torch.zeros(len(batch), max_skill_len, dtype=torch.long)
    gt_params = torch.zeros(len(batch), max_skill_len, 5)

    for i, s in enumerate(batch):
        skill_len = len(s['gt_skills'])
        gt_skills[i, :skill_len] = s['gt_skills']
        gt_params[i, :skill_len] = s['gt_params']

    collated = {
        'token_ids': token_ids,
        'gt_skills': gt_skills,
        'gt_params': gt_params,
    }

    # Optional features (if all samples have them)
    if 'z_v' in batch[0]:
        collated['z_v'] = torch.stack([s['z_v'] for s in batch])

    if 'state' in batch[0]:
        collated['state'] = torch.stack([s['state'] for s in batch])

    if 'risk_map' in batch[0]:
        collated['risk_map'] = torch.stack([s['risk_map'] for s in batch])

    if 'affordance_map' in batch[0]:
        collated['affordance_map'] = torch.stack([s['affordance_map'] for s in batch])

    return collated


class VLATrainer:
    """
    Trainer for VLA Transformer.
    """

    def __init__(
        self,
        model,
        lr=1e-4,
        weight_decay=1e-5,
        device='cpu'
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VLATrainer")

        self.model = model.to(device)
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.skill_accuracies = []

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with VLA samples

        Returns:
            metrics: dict with epoch metrics
        """
        self.model.train()
        epoch_loss = 0
        epoch_skill_acc = 0
        n_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Move to device
            token_ids = batch['token_ids'].to(self.device)
            gt_skills = batch['gt_skills'].to(self.device)
            gt_params = batch['gt_params'].to(self.device)

            # Optional features
            z_v = batch.get('z_v', None)
            if z_v is not None:
                z_v = z_v.to(self.device)

            state = batch.get('state', None)
            if state is not None:
                state = state.to(self.device)

            risk_map = batch.get('risk_map', None)
            if risk_map is not None:
                risk_map = risk_map.to(self.device)

            affordance_map = batch.get('affordance_map', None)
            if affordance_map is not None:
                affordance_map = affordance_map.to(self.device)

            # Forward pass
            loss, metrics = self.model.compute_loss(
                token_ids, gt_skills, gt_params,
                z_v, state, risk_map, affordance_map
            )

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += metrics['total_loss']

            # Compute skill accuracy
            with torch.no_grad():
                skill_logits, _, _, _ = self.model.forward(
                    token_ids, z_v, state, risk_map, affordance_map
                )
                pred_skills = skill_logits.argmax(dim=-1)
                # Only compare non-EOS positions
                plan_len = gt_skills.shape[1]
                acc = (pred_skills[:, :plan_len] == gt_skills).float().mean()
                epoch_skill_acc += acc.item()

            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_skill_acc / n_batches

        self.train_losses.append(avg_loss)
        self.skill_accuracies.append(avg_acc)

        return {
            'loss': avg_loss,
            'skill_accuracy': avg_acc,
        }

    def evaluate(self, dataloader):
        """
        Evaluate model on validation set.

        Args:
            dataloader: Validation DataLoader

        Returns:
            metrics: dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                token_ids = batch['token_ids'].to(self.device)
                gt_skills = batch['gt_skills'].to(self.device)
                gt_params = batch['gt_params'].to(self.device)

                z_v = batch.get('z_v', None)
                if z_v is not None:
                    z_v = z_v.to(self.device)

                state = batch.get('state', None)
                if state is not None:
                    state = state.to(self.device)

                risk_map = batch.get('risk_map', None)
                if risk_map is not None:
                    risk_map = risk_map.to(self.device)

                affordance_map = batch.get('affordance_map', None)
                if affordance_map is not None:
                    affordance_map = affordance_map.to(self.device)

                loss, _ = self.model.compute_loss(
                    token_ids, gt_skills, gt_params,
                    z_v, state, risk_map, affordance_map
                )

                total_loss += loss.item()

                # Accuracy
                skill_logits, _, _, _ = self.model.forward(
                    token_ids, z_v, state, risk_map, affordance_map
                )
                pred_skills = skill_logits.argmax(dim=-1)
                plan_len = gt_skills.shape[1]
                acc = (pred_skills[:, :plan_len] == gt_skills).float().mean()
                total_acc += acc.item()

                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        self.val_losses.append(avg_loss)
        self.scheduler.step(avg_loss)

        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_acc,
        }

    def train(
        self,
        train_loader,
        val_loader=None,
        n_epochs=100,
        log_interval=10,
        save_path=None
    ):
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader
            n_epochs: Number of epochs
            log_interval: Log every N epochs
            save_path: Optional path to save best model

        Returns:
            history: dict with training history
        """
        print("Training VLA Transformer...")

        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(train_loader)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    if save_path:
                        self.model.save(save_path)
                        print(f"  Saved best model (val_loss={best_val_loss:.4f})")

            if (epoch + 1) % log_interval == 0:
                msg = f"Epoch {epoch + 1}/{n_epochs}: "
                msg += f"train_loss={train_metrics['loss']:.4f}, "
                msg += f"skill_acc={train_metrics['skill_accuracy']:.2%}"

                if val_loader is not None:
                    msg += f", val_loss={val_metrics['val_loss']:.4f}, "
                    msg += f"val_acc={val_metrics['val_accuracy']:.2%}"

                print(msg)

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'skill_accuracies': self.skill_accuracies,
        }

        return history


def generate_synthetic_vla_data(n_samples=1000):
    """
    Generate synthetic VLA training data.

    Creates instruction-plan pairs with variations.

    Args:
        n_samples: Number of samples to generate

    Returns:
        trajectories: List of training samples
    """
    # Instruction templates
    templates = [
        "open the drawer without hitting the vase",
        "carefully open the top drawer",
        "open drawer while avoiding the fragile vase",
        "grasp the handle and pull the drawer open safely",
        "open the drawer and maintain clearance from vase",
        "quickly open the drawer but be careful with the vase",
        "slowly and safely open the top drawer",
        "pull open the drawer without breaking anything",
    ]

    # Modifiers
    speed_modifiers = ['', 'quickly ', 'slowly ', 'carefully ', 'safely ']
    clearance_modifiers = ['', 'maintain clearance ', 'avoid collision ', 'be careful ']

    trajectories = []

    for i in range(n_samples):
        # Random instruction
        base_instruction = np.random.choice(templates)

        # Add random modifiers
        if np.random.random() < 0.3:
            base_instruction = np.random.choice(speed_modifiers) + base_instruction

        # Standard skill sequence (with small variations)
        skill_sequence = [
            SkillID.LOCATE_DRAWER,
            SkillID.LOCATE_VASE,
            SkillID.PLAN_SAFE_APPROACH,
            SkillID.GRASP_HANDLE,
            SkillID.OPEN_WITH_CLEARANCE,
            SkillID.RETRACT_SAFE,
        ]

        # Random variations
        if np.random.random() < 0.2:
            # Skip locate vase (assume known)
            skill_sequence = [
                SkillID.LOCATE_DRAWER,
                SkillID.PLAN_SAFE_APPROACH,
                SkillID.GRASP_HANDLE,
                SkillID.OPEN_WITH_CLEARANCE,
                SkillID.RETRACT_SAFE,
            ]

        if np.random.random() < 0.1:
            # Skip retract
            skill_sequence = skill_sequence[:-1]

        # Skill parameters (with noise)
        skill_params = []
        for sid in skill_sequence:
            params = np.array([
                0.15 + np.random.normal(0, 0.02),  # clearance
                0.6 + np.random.normal(0, 0.1),    # speed
                0.5 + np.random.normal(0, 0.05),   # grasp force
                0.5 + np.random.normal(0, 0.05),   # retract speed
                1.0  # normalized timeout
            ], dtype=np.float32)
            params = np.clip(params, 0, 1)
            skill_params.append(params)

        traj = {
            'instruction': base_instruction,
            'skill_sequence': skill_sequence,
            'skill_params': skill_params,
        }

        # Optional: add synthetic features
        if np.random.random() < 0.5:
            traj['z_v'] = np.random.randn(128).astype(np.float32) * 0.1
            traj['state'] = np.random.randn(13).astype(np.float32) * 0.1

        trajectories.append(traj)

    return trajectories


def save_vla_dataset(trajectories, path):
    """Save VLA training dataset."""
    # Convert numpy arrays to lists for JSON
    serializable = []
    for traj in trajectories:
        item = {
            'instruction': traj['instruction'],
            'skill_sequence': traj['skill_sequence'],
            'skill_params': [p.tolist() for p in traj['skill_params']],
        }
        if 'z_v' in traj:
            item['z_v'] = traj['z_v'].tolist()
        if 'state' in traj:
            item['state'] = traj['state'].tolist()
        serializable.append(item)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(serializable, f)


def load_vla_dataset(path):
    """Load VLA training dataset."""
    with open(path, 'r') as f:
        data = json.load(f)

    trajectories = []
    for item in data:
        traj = {
            'instruction': item['instruction'],
            'skill_sequence': item['skill_sequence'],
            'skill_params': [np.array(p, dtype=np.float32) for p in item['skill_params']],
        }
        if 'z_v' in item:
            traj['z_v'] = np.array(item['z_v'], dtype=np.float32)
        if 'state' in item:
            traj['state'] = np.array(item['state'], dtype=np.float32)
        trajectories.append(traj)

    return trajectories
