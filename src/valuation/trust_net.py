"""
Trust Network for Real vs Synthetic Episode Classification

Learns to distinguish real z_V episodes from synthetic ones, providing a trust score
that can be used to weight synthetic data during offline RL training.

The key insight: synthetic latents with high p(real) are more likely to be useful
for policy training, while low p(real) indicates OOD garbage.
"""

import torch
import torch.nn as nn
import numpy as np


class TrustNet(nn.Module):
    """
    MLP classifier: episode features -> p(real).

    Input: Per-episode features extracted from z_V sequence
    Output: p_real in [0, 1] indicating trust/realness
    """

    def __init__(self, input_dim=6, hidden_dim=64):
        """
        Args:
            input_dim: Number of episode-level features (mean, std, etc.)
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Episode features [batch, input_dim]

        Returns:
            p_real: Trust score [batch, 1] in [0, 1]
        """
        return self.net(x)


class TemporalTrustNet(nn.Module):
    """
    GRU-based trust classifier that looks at z_V sequence directly.

    For when you want to capture temporal patterns, not just summary stats.
    """

    def __init__(self, latent_dim=128, hidden_dim=64, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_sequence):
        """
        Args:
            z_sequence: [batch, seq_len, latent_dim]

        Returns:
            p_real: [batch, 1]
        """
        _, h_n = self.gru(z_sequence)  # h_n: [num_layers, batch, hidden]
        final_hidden = h_n[-1]  # [batch, hidden]
        return self.classifier(final_hidden)


def extract_episode_features(z_sequence):
    """
    Extract simple statistical features from a z_V episode.

    Args:
        z_sequence: [T+1, latent_dim] numpy array

    Returns:
        features: [n_features] numpy array
    """
    features = [
        z_sequence.mean(),           # Global mean
        z_sequence.std(),            # Global std
        z_sequence.min(),            # Min value
        z_sequence.max(),            # Max value
        z_sequence.mean(axis=0).std(),  # Variance across dimensions
        np.abs(np.diff(z_sequence, axis=0)).mean(),  # Temporal smoothness
    ]
    return np.array(features, dtype=np.float32)


def build_trust_dataset(real_path, synthetic_path):
    """
    Build training dataset for trust_net.

    Args:
        real_path: Path to real z_V rollouts npz
        synthetic_path: Path to synthetic z_V rollouts npz

    Returns:
        X: [n_samples, n_features] features
        y: [n_samples] labels (1=real, 0=synthetic)
    """
    features = []
    labels = []

    # Load real data
    real_data = np.load(real_path, allow_pickle=True)
    n_real = int(real_data['n_episodes'])

    for ep in range(n_real):
        z_seq = real_data[f'ep_{ep}_z_sequence']
        feat = extract_episode_features(z_seq)
        features.append(feat)
        labels.append(1.0)  # Real

    # Load synthetic data
    syn_data = np.load(synthetic_path, allow_pickle=True)
    n_syn = int(syn_data['n_episodes'])

    for ep in range(n_syn):
        z_seq = syn_data[f'ep_{ep}_z_sequence']
        feat = extract_episode_features(z_seq)
        features.append(feat)
        labels.append(0.0)  # Synthetic

    X = np.stack(features, axis=0)
    y = np.array(labels)

    return X, y


def train_trust_net(
    real_path,
    synthetic_path,
    n_epochs=100,
    batch_size=32,
    lr=1e-3,
    device='cpu',
):
    """
    Train trust_net to distinguish real from synthetic episodes.

    Returns:
        model: Trained TrustNet
        metrics: Training metrics
    """
    print("=" * 60)
    print("TRUST_NET TRAINING")
    print("=" * 60)

    # Build dataset
    print("\nBuilding dataset...")
    X, y = build_trust_dataset(real_path, synthetic_path)

    n_real = int(y.sum())
    n_syn = len(y) - n_real
    print(f"  Real episodes: {n_real}")
    print(f"  Synthetic episodes: {n_syn}")
    print(f"  Feature dim: {X.shape[1]}")

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std

    print(f"\nFeature statistics:")
    feature_names = ['mean', 'std', 'min', 'max', 'dim_var', 'smoothness']
    for i, name in enumerate(feature_names):
        print(f"  {name}: mean={X[:, i].mean():.6f}, std={X[:, i].std():.6f}")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_norm).to(device)
    y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)

    # Split train/val
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    print(f"\nTrain/val split: {len(train_idx)}/{len(val_idx)}")

    # Create model
    model = TrustNet(input_dim=X.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()

        # Shuffle training data
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()

            train_pred = model(X_train)
            train_acc = ((train_pred > 0.5).float() == y_train).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={epoch_loss/n_batches:.4f}, "
                  f"train_acc={train_acc:.3f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.3f}")

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        all_pred = model(X_tensor).cpu().numpy().squeeze()

    real_trust = all_pred[y == 1].mean()
    syn_trust = all_pred[y == 0].mean()

    print(f"\nFinal Results:")
    print(f"  Best validation accuracy: {best_val_acc:.3f}")
    print(f"  Mean trust score (real episodes): {real_trust:.4f}")
    print(f"  Mean trust score (synthetic episodes): {syn_trust:.4f}")
    print(f"  Trust gap: {real_trust - syn_trust:.4f}")

    # ROC-AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, all_pred)
        print(f"  ROC-AUC: {auc:.4f}")
    except ImportError:
        auc = None
        print("  (sklearn not available for ROC-AUC)")

    metrics = {
        'best_val_acc': best_val_acc,
        'real_trust_mean': real_trust,
        'syn_trust_mean': syn_trust,
        'trust_gap': real_trust - syn_trust,
        'roc_auc': auc,
        'X_mean': X_mean,
        'X_std': X_std,
    }

    return model, metrics


def score_episodes_with_trust(
    model,
    npz_path,
    X_mean,
    X_std,
    device='cpu',
):
    """
    Score all episodes in a dataset with trust values.

    Args:
        model: Trained TrustNet
        npz_path: Path to z_V rollouts
        X_mean, X_std: Normalization stats from training

    Returns:
        trust_scores: [n_episodes] array of p_real values
    """
    data = np.load(npz_path, allow_pickle=True)
    n_episodes = int(data['n_episodes'])

    trust_scores = []

    model.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            z_seq = data[f'ep_{ep}_z_sequence']
            feat = extract_episode_features(z_seq)
            feat_norm = (feat - X_mean) / X_std

            feat_tensor = torch.FloatTensor(feat_norm).unsqueeze(0).to(device)
            trust = model(feat_tensor).item()
            trust_scores.append(trust)

    return np.array(trust_scores)
