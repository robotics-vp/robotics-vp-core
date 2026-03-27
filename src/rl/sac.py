"""
Soft Actor-Critic (SAC) with novelty-weighted sampling.

Features:
- Twin critics with target networks
- Tanh-squashed Gaussian actor
- Automatic entropy tuning
- Replay buffer with novelty-based prioritization
- Integrated encoder training
"""
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

Transition = tuple[Any, Any, Any, Any, Any, float]


class Actor(nn.Module):
    """
    Gaussian policy with tanh squashing: π_θ(a|z).

    Output: mean, logstd → sample → tanh → [0,1]²
    """

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.logstd_head = nn.Linear(hidden_dim, action_dim)

        # Initialize small weights for stable early training
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.logstd_head.weight, gain=0.01)

    def forward(self, latent):
        """
        Args:
            latent: [B, latent_dim]

        Returns:
            mean: [B, action_dim]
            logstd: [B, action_dim]
        """
        features = self.net(latent)
        mean = self.mean_head(features)
        logstd = self.logstd_head(features)

        # Clip logstd for numerical stability
        logstd = torch.clamp(logstd, -20, 2)

        return mean, logstd

    def sample(self, latent, deterministic=False, return_log_prob=True):
        """
        Sample action with reparameterization trick.

        Returns:
            action: [B, action_dim] in [0, 1]
            logprob: [B, 1] log probability (None if deterministic or return_log_prob=False)
        """
        mean, logstd = self.forward(latent)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action_raw = mean
            logprob = None
        else:
            # Reparameterization trick
            eps = dist.rsample()
            action_raw = eps

            if return_log_prob:
                logprob = dist.log_prob(eps) - torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-6)
                logprob = logprob.sum(dim=-1, keepdim=True)
            else:
                logprob = None

        # Tanh squashing
        action = torch.tanh(action_raw)

        # Scale to [0, 1] from [-1, 1]
        action = (action + 1) / 2

        return action, logprob


class Critic(nn.Module):
    """
    Q-function: Q_ϕ(z, a).

    Twin critics for double Q-learning (reduces overestimation).
    """

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent, action):
        """
        Args:
            latent: [B, latent_dim]
            action: [B, action_dim]

        Returns:
            q1, q2: [B, 1] Q-values from both critics
        """
        x = torch.cat([latent, action], dim=1)
        return self.q1(x), self.q2(x)


class NoveltyReplayBuffer:
    """
    Replay buffer with novelty-weighted sampling.

    Samples are prioritized by: weight = novelty × |TD_error|
    """

    def __init__(
        self,
        capacity=int(1e6),
        *,
        artifact_dir: Optional[str] = None,
        sampling_log_interval: int = 50,
    ):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.priorities: deque[float] = deque(maxlen=capacity)
        self.transition_metadata: deque[dict[str, Any]] = deque(maxlen=capacity)
        self.dispatch_by_episode: Dict[str, Dict[str, Any]] = {}
        self.receipt_feedback_by_episode: Dict[str, Dict[str, Any]] = {}
        self.last_sampling_artifact: Optional[Dict[str, Any]] = None
        self.sample_calls = 0
        self.sampling_log_interval = int(sampling_log_interval)
        self._artifact_path: Optional[Path] = None
        if artifact_dir:
            artifact_root = Path(artifact_dir)
            artifact_root.mkdir(parents=True, exist_ok=True)
            self._artifact_path = artifact_root / "online_sac_sampling.jsonl"

    def push(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        novelty=1.0,
        *,
        episode_id: Optional[str] = None,
        queue_dispatch: Optional[Mapping[str, Any]] = None,
        source_domain: Optional[str] = None,
        receipt_feedback: Optional[Mapping[str, Any]] = None,
        condition_vector: Optional[Any] = None,
        skill_mode: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        """Add transition to buffer."""
        self.buffer.append((obs, action, reward, next_obs, done, novelty))
        self.priorities.append(1.0)  # Initial priority (will update)
        transition_meta: dict[str, Any] = {
            "episode_id": str(episode_id or ""),
            "queue_dispatch": dict(queue_dispatch or {}),
            "source_domain": source_domain,
            "receipt_feedback": dict(receipt_feedback or {}),
            "condition_vector": condition_vector,
            "skill_mode": skill_mode,
            "metadata": dict(metadata or {}),
        }
        self.transition_metadata.append(transition_meta)
        episode_key = str(transition_meta.get("episode_id", "") or "")
        queue_payload = transition_meta.get("queue_dispatch")
        receipt_payload = transition_meta.get("receipt_feedback")
        if episode_key and isinstance(queue_payload, Mapping) and queue_payload:
            self.dispatch_by_episode[episode_key] = dict(queue_payload)
        if episode_key and isinstance(receipt_payload, Mapping) and receipt_payload:
            self.receipt_feedback_by_episode[episode_key] = dict(receipt_payload)

    def sample(self, batch_size, use_prioritization=True, return_metadata=False):
        """
        Sample batch with novelty-based prioritization.

        Returns:
            batch: Tuple of (obs, actions, rewards, next_obs, dones, novelties, indices)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if use_prioritization and len(self.priorities) > 0:
            priorities = np.array(self.priorities, dtype=np.float64)
            priorities = np.abs(priorities) + 1e-6
        else:
            priorities = np.ones(len(self.buffer), dtype=np.float64)

        queue_multipliers = np.array(
            [self._sampling_multiplier(index) for index in range(len(self.buffer))],
            dtype=np.float64,
        )
        effective_priorities = priorities * queue_multipliers
        total_priority = float(effective_priorities.sum())
        if total_priority <= 0.0:
            effective_priorities = np.ones(len(self.buffer), dtype=np.float64)
            total_priority = float(effective_priorities.sum())
        probs = effective_priorities / total_priority
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        batch = [self.buffer[i] for i in indices]
        sampled_metadata = [self._transition_metadata(int(i)) for i in indices]

        obs = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_obs = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        novelties = np.array([t[5] for t in batch])

        self.sample_calls += 1
        self.last_sampling_artifact = self._build_sampling_artifact(
            indices=indices,
            sample_probs=probs,
            queue_multipliers=queue_multipliers,
            sampled_metadata=sampled_metadata,
        )
        self._append_sampling_artifact(self.last_sampling_artifact)

        if return_metadata:
            return obs, actions, rewards, next_obs, dones, novelties, indices, sampled_metadata
        return obs, actions, rewards, next_obs, dones, novelties, indices

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def apply_queue_dispatch(self, dispatch_artifact: Mapping[str, Any]) -> None:
        entries = list(dispatch_artifact.get("entries", []) or [])
        self.dispatch_by_episode = {
            str(entry.get("episode_id", "")): dict(entry)
            for entry in entries
            if entry.get("episode_id")
        }

    def attach_receipt_feedback(self, feedback_by_episode: Mapping[str, Mapping[str, Any]]) -> None:
        for episode_id, payload in dict(feedback_by_episode or {}).items():
            if episode_id:
                self.receipt_feedback_by_episode[str(episode_id)] = dict(payload or {})

    def __len__(self):
        return len(self.buffer)

    def _sampling_multiplier(self, index: int) -> float:
        metadata = self._transition_metadata(index)
        episode_id = str(metadata.get("episode_id", "") or "")
        queue_dispatch = dict(metadata.get("queue_dispatch", {}) or {})
        if episode_id and episode_id in self.dispatch_by_episode:
            queue_dispatch = dict(self.dispatch_by_episode[episode_id])
        base_weight = float(queue_dispatch.get("base_weight", 1.0) or 1.0)
        adjusted_weight = float(queue_dispatch.get("adjusted_weight", base_weight) or base_weight)
        if bool(queue_dispatch.get("dropped", False)):
            return 1e-6
        if abs(base_weight) <= 1e-9:
            return max(1.0, adjusted_weight)
        return max(1e-6, adjusted_weight / base_weight)

    def _transition_metadata(self, index: int) -> Dict[str, Any]:
        if index >= len(self.transition_metadata):
            return {}
        metadata = dict(self.transition_metadata[index] or {})
        episode_id = str(metadata.get("episode_id", "") or "")
        if episode_id and episode_id in self.dispatch_by_episode and not metadata.get("queue_dispatch"):
            metadata["queue_dispatch"] = dict(self.dispatch_by_episode[episode_id])
        if episode_id and episode_id in self.receipt_feedback_by_episode and not metadata.get("receipt_feedback"):
            metadata["receipt_feedback"] = dict(self.receipt_feedback_by_episode[episode_id])
        return metadata

    def _build_sampling_artifact(
        self,
        *,
        indices,
        sample_probs: np.ndarray,
        queue_multipliers: np.ndarray,
        sampled_metadata,
    ) -> Dict[str, Any]:
        entries: Dict[str, Dict[str, Any]] = {}
        first_seen: Dict[str, int] = {}
        for index in range(len(self.buffer)):
            metadata = self._transition_metadata(index)
            episode_id = str(metadata.get("episode_id", "") or f"transition_{index:06d}")
            dispatch = dict(metadata.get("queue_dispatch", {}) or self.dispatch_by_episode.get(episode_id, {}) or {})
            first_seen.setdefault(episode_id, index)
            if episode_id not in entries:
                receipt_feedback = dict(metadata.get("receipt_feedback", {}) or self.receipt_feedback_by_episode.get(episode_id, {}) or {})
                entries[episode_id] = {
                    "episode_id": episode_id,
                    "original_rank": int(dispatch.get("original_rank", first_seen[episode_id])),
                    "adjusted_rank": int(dispatch.get("adjusted_rank", first_seen[episode_id])),
                    "reweight_factor": float(queue_multipliers[index]),
                    "reasons": list(dispatch.get("reasons", []) or []),
                    "evidence": dict(dispatch.get("evidence", {}) or {}),
                    "promotion_stage": str(dispatch.get("promotion_stage", "compare_only") or "compare_only"),
                    "influence_source": str(dispatch.get("influence_source", "heuristic") or "heuristic"),
                    "authority_class": str(dispatch.get("authority_class", "observational_only") or "observational_only"),
                    "decision_scope": str(
                        dispatch.get("decision_scope", "training_distribution_only")
                        or "training_distribution_only"
                    ),
                    "reward_math_mutation": bool(dispatch.get("reward_math_mutation", False)),
                    "source_domain": metadata.get("source_domain"),
                    "receipt_feedback": receipt_feedback,
                }
        sampled_episode_ids = [
            str(metadata.get("episode_id", "") or f"transition_{int(index):06d}")
            for index, metadata in zip(indices.tolist(), sampled_metadata)
        ]
        original_queue_order = [
            row["episode_id"]
            for row in sorted(entries.values(), key=lambda row: (row["original_rank"], row["episode_id"]))
        ]
        adjusted_queue_order = [
            row["episode_id"]
            for row in sorted(
                entries.values(),
                key=lambda row: (row["adjusted_rank"], -row["reweight_factor"], row["episode_id"]),
            )
        ]
        authority_class = "observational_only"
        if any(row.get("authority_class") == "bounded_authority" for row in entries.values()):
            authority_class = "bounded_authority"
        elif any(row.get("authority_class") == "ordering_only" for row in entries.values()):
            authority_class = "ordering_only"
        return {
            "receipt_kind": "online_replay_sampling_artifact_v1",
            "authority_class": authority_class,
            "decision_scope": "training_distribution_only",
            "reward_math_mutation": False,
            "sample_call": int(self.sample_calls),
            "buffer_size": len(self.buffer),
            "selected_transition_count": int(len(indices)),
            "selected_episode_ids": sampled_episode_ids,
            "original_queue_order": original_queue_order,
            "adjusted_queue_order": adjusted_queue_order,
            "reweight_factors": {
                row["episode_id"]: float(row["reweight_factor"])
                for row in entries.values()
            },
            "mean_sample_probability": float(np.mean(sample_probs[indices])) if len(indices) else 0.0,
            "entries": [
                dict(row)
                for row in sorted(entries.values(), key=lambda row: (row["adjusted_rank"], row["episode_id"]))
            ],
        }

    def _append_sampling_artifact(self, artifact: Mapping[str, Any]) -> None:
        if self._artifact_path is None:
            return
        if self.sample_calls % max(1, self.sampling_log_interval) != 0:
            return
        with self._artifact_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(artifact), sort_keys=True) + "\n")


class SACAgent:
    """
    Soft Actor-Critic agent with encoder and novelty weighting.

    Architecture:
    - Encoder f_ψ: obs → latent (128D)
    - Actor π_θ: latent → action (2D, [speed, care])
    - Critics Q_ϕ1, Q_ϕ2: (latent, action) → Q-value
    """

    def __init__(self, encoder, latent_dim=128, action_dim=2,
                 lr=3e-4, gamma=0.995, tau=5e-3,
                 buffer_capacity=int(1e6), batch_size=1024,
                 target_entropy=None, device='cpu',
                 contract_aware_adapter: Optional[object] = None,
                 sampling_artifact_dir: Optional[str] = None,
                 sampling_log_interval: int = 50):
        """
        Args:
            encoder: EncoderWithAuxiliaries instance
            latent_dim: Latent dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            buffer_capacity: Replay buffer size
            batch_size: Mini-batch size
            target_entropy: Target entropy for automatic tuning (-action_dim default)
            device: torch device
        """
        self.encoder = encoder.to(device)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)  # Ensure scalar
        self.tau = float(tau)  # Ensure scalar
        self.batch_size = batch_size
        self.device = device
        self.contract_aware_adapter = contract_aware_adapter

        # Actor
        self.actor = Actor(latent_dim, action_dim).to(device)

        # Critics
        self.critic = Critic(latent_dim, action_dim).to(device)
        self.critic_target = Critic(latent_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy temperature (automatic tuning)
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        # Optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay_buffer = NoveltyReplayBuffer(
            capacity=buffer_capacity,
            artifact_dir=sampling_artifact_dir,
            sampling_log_interval=sampling_log_interval,
        )

        # Training metrics
        self.training_steps = 0

    def select_action(self, obs, novelty=None, deterministic=False):
        """
        Select action from policy.

        Args:
            obs: Observation dict or tensor
            novelty: Novelty score (for buffer storage)
            deterministic: Use mean action (no sampling)

        Returns:
            action: numpy array [action_dim]
            novelty: novelty score (pass-through)
        """
        with torch.no_grad():
            if isinstance(obs, dict):
                obs_tensor = self._obs_to_tensor(obs)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            latent = self.encoder.encode(obs_tensor)
            action, _ = self.actor.sample(latent, deterministic=deterministic, return_log_prob=False)

        return action.cpu().numpy()[0], novelty

    def store_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        novelty=1.0,
        *,
        episode_id: Optional[str] = None,
        queue_dispatch: Optional[Mapping[str, Any]] = None,
        source_domain: Optional[str] = None,
        receipt_feedback: Optional[Mapping[str, Any]] = None,
        condition_vector: Optional[Any] = None,
        skill_mode: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        """Store transition in replay buffer."""
        # Convert obs dict to array
        if isinstance(obs, dict):
            obs = np.array([obs['t'], obs['completed'], obs['attempts'], obs['errors']])
        if isinstance(next_obs, dict):
            next_obs = np.array([next_obs['t'], next_obs['completed'],
                                next_obs['attempts'], next_obs['errors']])

        self.replay_buffer.push(
            obs,
            action,
            reward,
            next_obs,
            done,
            novelty,
            episode_id=episode_id,
            queue_dispatch=queue_dispatch,
            source_domain=source_domain,
            receipt_feedback=receipt_feedback,
            condition_vector=condition_vector,
            skill_mode=skill_mode,
            metadata=metadata,
        )

    def apply_queue_dispatch(self, dispatch_artifact: Mapping[str, Any]) -> None:
        self.replay_buffer.apply_queue_dispatch(dispatch_artifact)

    def attach_receipt_feedback(self, feedback_by_episode: Mapping[str, Mapping[str, Any]]) -> None:
        self.replay_buffer.attach_receipt_feedback(feedback_by_episode)

    def get_last_sampling_artifact(self) -> Optional[Dict[str, Any]]:
        return self.replay_buffer.last_sampling_artifact

    def update(self, aux_loss_weight={'consistency': 0.1, 'contrastive': 0.1}):
        """
        SAC update with encoder auxiliary losses.

        Returns:
            metrics: Dict of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        obs, actions, rewards, next_obs, dones, novelties, indices, sample_metadata = \
            self.replay_buffer.sample(self.batch_size, return_metadata=True)

        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        novelties_t = torch.FloatTensor(novelties).to(self.device)
        condition_batch = _condition_batch_from_metadata(sample_metadata)
        skill_modes = _skill_modes_from_metadata(sample_metadata)

        # Encode observations
        latent = self.encoder.encode(obs_t)
        next_latent = self.encoder.encode(next_obs_t)

        # --- Critic Update ---
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_logprobs = self.actor.sample(next_latent, return_log_prob=True)

            # Target Q-values (use minimum of twin critics)
            q1_target, q2_target = self.critic_target(next_latent, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # Bellman backup with entropy
            target_value = rewards_t + (1 - dones_t) * self.gamma * (q_target - self.alpha * next_logprobs)

        # Current Q-values
        q1, q2 = self.critic(latent.detach(), actions_t)  # Detach encoder gradients

        # Novelty-weighted critic loss
        weights = torch.clamp(novelties_t, 0.5, 2.0)
        weights = weights / (weights.mean() + 1e-6)

        critic_loss = (F.mse_loss(q1, target_value, reduction='none') * weights.unsqueeze(1)).mean()
        critic_loss += (F.mse_loss(q2, target_value, reduction='none') * weights.unsqueeze(1)).mean()
        contract_alignment_loss = torch.tensor(0.0, device=self.device)
        contract_aware_bundle = None
        contract_aware_metrics = {}
        if self.contract_aware_adapter is not None and getattr(self.contract_aware_adapter, "live_mode_enabled", False):
            contract_aware_bundle = self.contract_aware_adapter.compute_loss_bundle(
                latent_batch=latent.detach(),
                action_batch=actions_t.detach(),
                reward_batch=rewards_t.detach().squeeze(-1),
                done_batch=dones_t.detach().squeeze(-1),
                condition_batch=condition_batch,
                skill_modes=skill_modes,
                reference_scalar_predictions=torch.min(q1.detach(), q2.detach()).squeeze(-1),
            )
            if contract_aware_bundle.get("enabled", False):
                alignment_target = contract_aware_bundle["outputs"].compiled_scalar.detach().unsqueeze(-1)
                contract_alignment_loss = (
                    F.mse_loss(q1, alignment_target) + F.mse_loss(q2, alignment_target)
                )
                critic_alignment_weight = float(
                    getattr(getattr(self.contract_aware_adapter, "config", None), "critic_alignment_weight", 0.0) or 0.0
                )
                critic_loss = critic_loss + critic_alignment_weight * contract_alignment_loss

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update priorities
        td_errors = (target_value - q1).abs().detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors * novelties)

        if self.contract_aware_adapter is not None and getattr(self.contract_aware_adapter, "live_mode_enabled", False):
            if contract_aware_bundle is not None and contract_aware_bundle.get("enabled", False):
                contract_aware_metrics = self.contract_aware_adapter.optimize_loss_bundle(contract_aware_bundle)
                contract_aware_metrics["critic_alignment_loss"] = float(contract_alignment_loss.detach().item())
        elif self.contract_aware_adapter is not None:
            contract_aware_metrics = self.contract_aware_adapter.update_from_batch(
                latent_batch=latent.detach(),
                action_batch=actions_t.detach(),
                reward_batch=rewards_t.detach().squeeze(-1),
                done_batch=dones_t.detach().squeeze(-1),
                condition_batch=condition_batch,
                skill_modes=skill_modes,
            )

        # --- Actor Update ---
        # Sample actions from current policy
        new_actions, logprobs = self.actor.sample(latent, return_log_prob=True)

        # Q-values for new actions
        q1_new, q2_new = self.critic(latent.detach(), new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss (maximize Q - α*entropy)
        actor_loss = (self.alpha * logprobs - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Entropy Temperature Update ---
        alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # --- Encoder Update (with auxiliary losses) ---
        # Re-encode to get fresh gradients for encoder training
        latent_fresh = self.encoder.encode(obs_t)
        next_latent_fresh = self.encoder.encode(next_obs_t)

        encoder_loss = 0

        # Consistency loss (only for encoders that support it)
        if hasattr(self.encoder, 'use_consistency') and self.encoder.use_consistency:
            consistency_loss = self.encoder.compute_consistency_loss(latent_fresh, next_latent_fresh)
            encoder_loss += aux_loss_weight['consistency'] * consistency_loss
        else:
            consistency_loss = torch.tensor(0.0)

        # Contrastive loss (only for encoders that support it)
        if hasattr(self.encoder, 'use_contrastive') and self.encoder.use_contrastive:
            contrastive_loss = self.encoder.compute_contrastive_loss(latent_fresh)
            encoder_loss += aux_loss_weight['contrastive'] * contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0)

        if encoder_loss > 0:
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()

        # --- Soft Update Target Networks ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.training_steps += 1

        # Return metrics
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'mean_novelty': novelties.mean(),
            'mean_weight': weights.mean().item(),
            'q_mean': q_new.mean().item()
        }
        sampling_artifact = self.replay_buffer.last_sampling_artifact or {}
        if sampling_artifact:
            metrics['sampling_selected_episode_count'] = len(sampling_artifact.get('selected_episode_ids', []))
            metrics['sampling_queue_entry_count'] = len(sampling_artifact.get('entries', []))
            metrics['sampling_mean_probability'] = float(sampling_artifact.get('mean_sample_probability', 0.0))
        for key, value in contract_aware_metrics.items():
            metrics[f'contract_aware_{key}'] = value
        return metrics

    def _obs_to_tensor(self, obs_dict):
        """Convert observation dict to tensor."""
        features = np.array([obs_dict['t'], obs_dict['completed'],
                            obs_dict['attempts'], obs_dict['errors']])
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def save(self, path):
        """Save agent state."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'training_steps': self.training_steps,
            'contract_aware_adapter': (
                self.contract_aware_adapter.state_dict()
                if self.contract_aware_adapter is not None and hasattr(self.contract_aware_adapter, 'state_dict')
                else None
            ),
        }, path)

    def load(self, path):
        """Load agent state."""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        self.training_steps = checkpoint['training_steps']
        if (
            self.contract_aware_adapter is not None
            and checkpoint.get('contract_aware_adapter') is not None
            and hasattr(self.contract_aware_adapter, 'load_state_dict')
        ):
            self.contract_aware_adapter.load_state_dict(checkpoint['contract_aware_adapter'])


def _condition_batch_from_metadata(sample_metadata) -> Optional[np.ndarray]:
    rows: list[Optional[np.ndarray]] = []
    max_dim = 0
    for metadata in sample_metadata:
        condition = metadata.get("condition_vector")
        if condition is None:
            rows.append(None)
            continue
        if hasattr(condition, "to_vector"):
            values = np.asarray(condition.to_vector(), dtype=np.float32).reshape(-1)
        elif isinstance(condition, dict):
            try:
                values = np.asarray(list(condition.values()), dtype=np.float32).reshape(-1)
            except Exception:
                values = np.zeros(0, dtype=np.float32)
        else:
            values = np.asarray(condition, dtype=np.float32).reshape(-1)
        rows.append(values)
        max_dim = max(max_dim, int(values.shape[0]))
    if max_dim <= 0:
        return None
    padded: list[np.ndarray] = []
    for row_values in rows:
        if row_values is None:
            padded.append(np.zeros(max_dim, dtype=np.float32))
            continue
        if row_values.shape[0] < max_dim:
            row_values = np.pad(row_values, (0, max_dim - row_values.shape[0]))
        elif row_values.shape[0] > max_dim:
            row_values = row_values[:max_dim]
        padded.append(row_values.astype(np.float32))
    return np.stack(padded, axis=0)


def _skill_modes_from_metadata(sample_metadata) -> list[str]:
    return [
        str(metadata.get("skill_mode") or "efficiency_throughput")
        for metadata in sample_metadata
    ]
