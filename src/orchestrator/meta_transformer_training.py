"""
Meta-Transformer Training Infrastructure.

Implements cross-attention between VLA (affordance) and DINO (semantic) embeddings,
with semantic token supervision and authority prediction.

No actual training yet - provides dataloader, batching, forward pass, and eval.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from src.utils.json_safe import to_json_safe

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MetaTransformerSample:
    """
    Training sample for meta-transformer.

    Contains:
    - VLA embeddings (affordance/action-oriented)
    - DINO embeddings (semantic/visual features)
    - Semantic tokens (ontology-derived)
    - Ground truth authority (which stream should dominate)
    - Task context
    """
    sample_id: str
    vla_embedding: np.ndarray  # (vla_dim,) or (seq_len, vla_dim)
    dino_embedding: np.ndarray  # (dino_dim,) or (patch_len, dino_dim)
    semantic_tokens: List[str]  # ["drawer", "vase", "fragile", "grasp"]
    authority_gt: str  # "vla" or "dino" - ground truth which to trust
    confidence_vla: float  # VLA confidence score
    confidence_dino: float  # DINO confidence score
    task_context: Dict[str, Any]  # {"task_type": "drawer_open", "safety_critical": True}


@dataclass
class MetaTransformerBatch:
    """Batched samples for training."""
    vla_embeddings: Any  # (B, vla_dim) or (B, seq_len, vla_dim)
    dino_embeddings: Any  # (B, dino_dim) or (B, patch_len, dino_dim)
    semantic_token_ids: Any  # (B, max_tokens)
    authority_gt: Any  # (B,) - 0 for dino, 1 for vla
    confidences_vla: Any  # (B,)
    confidences_dino: Any  # (B,)


# =============================================================================
# Semantic Token Vocabulary
# =============================================================================

SEMANTIC_VOCAB = [
    "<pad>", "<unk>", "drawer", "vase", "fragile", "grasp", "open", "close",
    "avoid", "collision", "safe", "energy", "high_priority", "low_priority",
    "careful", "fast", "slow", "robot_arm", "gripper", "handle", "surface",
    "obstacle", "clearance", "retract", "approach", "skill", "primitive",
    "affordance", "semantic", "visual", "action", "state", "goal",
]
SEMANTIC_TOKEN_TO_IDX = {tok: i for i, tok in enumerate(SEMANTIC_VOCAB)}
IDX_TO_SEMANTIC_TOKEN = {i: tok for tok, i in SEMANTIC_TOKEN_TO_IDX.items()}


def encode_semantic_tokens(tokens: List[str], max_len: int = 16) -> np.ndarray:
    """Encode semantic tokens to indices with padding."""
    ids = []
    for tok in tokens[:max_len]:
        ids.append(SEMANTIC_TOKEN_TO_IDX.get(tok, SEMANTIC_TOKEN_TO_IDX["<unk>"]))
    # Pad
    while len(ids) < max_len:
        ids.append(SEMANTIC_TOKEN_TO_IDX["<pad>"])
    return np.array(ids, dtype=np.int64)


def decode_semantic_tokens(ids: np.ndarray) -> List[str]:
    """Decode token indices back to strings."""
    tokens = []
    for idx in ids:
        if idx == SEMANTIC_TOKEN_TO_IDX["<pad>"]:
            continue
        tokens.append(IDX_TO_SEMANTIC_TOKEN.get(int(idx), "<unk>"))
    return tokens


# =============================================================================
# PyTorch Dataset
# =============================================================================

class MetaTransformerDataset(Dataset if TORCH_AVAILABLE else object):
    """Dataset for meta-transformer training."""

    def __init__(self, samples: List[MetaTransformerSample], max_semantic_tokens: int = 16):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MetaTransformerDataset")

        self.samples = samples
        self.max_semantic_tokens = max_semantic_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors
        vla_emb = torch.from_numpy(sample.vla_embedding).float()
        dino_emb = torch.from_numpy(sample.dino_embedding).float()

        # Encode semantic tokens
        semantic_ids = encode_semantic_tokens(sample.semantic_tokens, self.max_semantic_tokens)
        semantic_ids = torch.from_numpy(semantic_ids).long()

        # Authority: 0 = dino, 1 = vla
        authority = 1 if sample.authority_gt == "vla" else 0

        return {
            "vla_embedding": vla_emb,
            "dino_embedding": dino_emb,
            "semantic_token_ids": semantic_ids,
            "authority_gt": authority,
            "confidence_vla": sample.confidence_vla,
            "confidence_dino": sample.confidence_dino,
        }


def collate_meta_transformer_batch(batch):
    """Collate samples into batched tensors."""
    return {
        "vla_embeddings": torch.stack([s["vla_embedding"] for s in batch]),
        "dino_embeddings": torch.stack([s["dino_embedding"] for s in batch]),
        "semantic_token_ids": torch.stack([s["semantic_token_ids"] for s in batch]),
        "authority_gt": torch.tensor([s["authority_gt"] for s in batch], dtype=torch.long),
        "confidences_vla": torch.tensor([s["confidence_vla"] for s in batch], dtype=torch.float32),
        "confidences_dino": torch.tensor([s["confidence_dino"] for s in batch], dtype=torch.float32),
    }


# =============================================================================
# Meta-Transformer Model with Cross-Attention
# =============================================================================

class MetaTransformerNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Meta-transformer with cross-attention between VLA and DINO streams.

    Architecture:
    1. Project VLA and DINO embeddings to shared space
    2. Cross-attention: VLA attends to DINO and vice versa
    3. Fuse attended representations
    4. Predict:
       - Shared policy state
       - Diffusion conditioning
       - Semantic tokens (from ontology)
       - Authority (which stream to trust)
    """

    def __init__(
        self,
        vla_dim: int = 128,
        dino_dim: int = 256,
        hidden_dim: int = 128,
        num_semantic_tokens: int = len(SEMANTIC_VOCAB),
        max_output_tokens: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_output_tokens = max_output_tokens

        # Input projections
        self.vla_proj = nn.Linear(vla_dim, hidden_dim)
        self.dino_proj = nn.Linear(dino_dim, hidden_dim)

        # Cross-attention layers
        self.vla_to_dino_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.dino_to_vla_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output heads
        # 1. Shared policy state
        self.policy_head = nn.Linear(hidden_dim, hidden_dim)

        # 2. Diffusion conditioning
        self.diffusion_head = nn.Linear(hidden_dim, hidden_dim)

        # 3. Semantic token prediction (sequence generation)
        self.token_head = nn.Linear(hidden_dim, num_semantic_tokens)

        # 4. Authority prediction (which stream to trust)
        self.authority_head = nn.Linear(hidden_dim, 2)  # Binary: dino vs vla

        # Learnable query for semantic token generation
        self.token_queries = nn.Parameter(torch.randn(max_output_tokens, hidden_dim))

    def forward(
        self,
        vla_emb: torch.Tensor,
        dino_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through meta-transformer.

        Args:
            vla_emb: (B, vla_dim) VLA embeddings
            dino_emb: (B, dino_dim) DINO embeddings

        Returns:
            outputs: dict with policy_state, diffusion_cond, token_logits, authority_logits
        """
        B = vla_emb.shape[0]

        # Project to shared space
        vla_h = self.vla_proj(vla_emb).unsqueeze(1)  # (B, 1, hidden)
        dino_h = self.dino_proj(dino_emb).unsqueeze(1)  # (B, 1, hidden)

        # Cross-attention: VLA attends to DINO
        vla_attended, _ = self.vla_to_dino_attn(vla_h, dino_h, dino_h)  # (B, 1, hidden)

        # Cross-attention: DINO attends to VLA
        dino_attended, _ = self.dino_to_vla_attn(dino_h, vla_h, vla_h)  # (B, 1, hidden)

        # Fuse
        fused = torch.cat([vla_attended.squeeze(1), dino_attended.squeeze(1)], dim=-1)  # (B, hidden*2)
        shared_repr = self.fusion_mlp(fused)  # (B, hidden)

        # Output heads
        policy_state = self.policy_head(shared_repr)  # (B, hidden)
        diffusion_cond = self.diffusion_head(shared_repr)  # (B, hidden)

        # Semantic token prediction: use learnable queries attending to shared repr
        token_queries = self.token_queries.unsqueeze(0).expand(B, -1, -1)  # (B, max_tokens, hidden)
        shared_repr_expanded = shared_repr.unsqueeze(1).expand(-1, self.max_output_tokens, -1)

        # Simple: just project each query conditioned on shared repr
        token_input = token_queries + shared_repr_expanded
        token_logits = self.token_head(token_input)  # (B, max_tokens, vocab_size)

        # Authority prediction
        authority_logits = self.authority_head(shared_repr)  # (B, 2)

        return {
            "policy_state": policy_state,
            "diffusion_cond": diffusion_cond,
            "token_logits": token_logits,
            "authority_logits": authority_logits,
            "shared_repr": shared_repr,
        }


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_meta_sample(seed: int = None) -> MetaTransformerSample:
    """Generate synthetic sample for meta-transformer training."""
    if seed is not None:
        np.random.seed(seed)

    # Random embeddings
    vla_dim = 128
    dino_dim = 256
    vla_embedding = np.random.randn(vla_dim).astype(np.float32) * 0.1
    dino_embedding = np.random.randn(dino_dim).astype(np.float32) * 0.1

    # Random semantic tokens
    possible_tokens = [
        "drawer", "vase", "fragile", "grasp", "open", "avoid", "collision",
        "safe", "energy", "careful", "fast", "robot_arm", "gripper", "handle",
    ]
    num_tokens = np.random.randint(3, 10)
    semantic_tokens = list(np.random.choice(possible_tokens, size=num_tokens, replace=False))

    # Determine authority based on tokens
    if "fragile" in semantic_tokens or "safe" in semantic_tokens or "avoid" in semantic_tokens:
        authority_gt = "dino"  # Semantic reasoning is critical
        confidence_dino = np.random.uniform(0.7, 1.0)
        confidence_vla = np.random.uniform(0.3, 0.7)
    elif "grasp" in semantic_tokens or "fast" in semantic_tokens:
        authority_gt = "vla"  # Action/affordance is critical
        confidence_vla = np.random.uniform(0.7, 1.0)
        confidence_dino = np.random.uniform(0.3, 0.7)
    else:
        # Balanced
        authority_gt = np.random.choice(["vla", "dino"])
        confidence_vla = np.random.uniform(0.4, 0.8)
        confidence_dino = np.random.uniform(0.4, 0.8)

    task_context = {
        "task_type": np.random.choice(["drawer_open", "drawer_close", "pick_place"]),
        "safety_critical": "fragile" in semantic_tokens or "safe" in semantic_tokens,
        "high_energy": "fast" in semantic_tokens,
    }

    return MetaTransformerSample(
        sample_id=f"meta_sample_{seed or np.random.randint(100000)}",
        vla_embedding=vla_embedding,
        dino_embedding=dino_embedding,
        semantic_tokens=semantic_tokens,
        authority_gt=authority_gt,
        confidence_vla=float(confidence_vla),
        confidence_dino=float(confidence_dino),
        task_context=task_context,
    )


def generate_meta_transformer_dataset(num_samples: int = 1000) -> List[MetaTransformerSample]:
    """Generate synthetic dataset for meta-transformer."""
    samples = []
    for i in range(num_samples):
        samples.append(generate_synthetic_meta_sample(seed=i))
    return samples


def sample_to_dict(sample: MetaTransformerSample) -> Dict[str, Any]:
    """Convert sample to JSON-serializable dict."""
    return {
        "sample_id": sample.sample_id,
        "vla_embedding": sample.vla_embedding.tolist(),
        "dino_embedding": sample.dino_embedding.tolist(),
        "semantic_tokens": sample.semantic_tokens,
        "authority_gt": sample.authority_gt,
        "confidence_vla": sample.confidence_vla,
        "confidence_dino": sample.confidence_dino,
        "task_context": sample.task_context,
    }


def save_meta_transformer_dataset(samples: List[MetaTransformerSample], path: str) -> None:
    """Save dataset to JSON."""
    data = [sample_to_dict(s) for s in samples]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(samples)} meta-transformer samples to {path}")


def load_meta_transformer_dataset(path: str) -> List[MetaTransformerSample]:
    """Load dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = MetaTransformerSample(
            sample_id=item["sample_id"],
            vla_embedding=np.array(item["vla_embedding"], dtype=np.float32),
            dino_embedding=np.array(item["dino_embedding"], dtype=np.float32),
            semantic_tokens=item["semantic_tokens"],
            authority_gt=item["authority_gt"],
            confidence_vla=item["confidence_vla"],
            confidence_dino=item["confidence_dino"],
            task_context=item["task_context"],
        )
        samples.append(sample)
    return samples


# =============================================================================
# Training Loop (Placeholder)
# =============================================================================

def forward_pass_test(
    model: MetaTransformerNet,
    batch_size: int = 4,
    include_shared_repr: bool = False,
) -> Dict[str, Any]:
    """
    Test forward pass through meta-transformer.

    Returns metrics about output shapes and values.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    model.eval()

    # Create synthetic batch
    vla_emb = torch.randn(batch_size, 128)
    dino_emb = torch.randn(batch_size, 256)

    with torch.no_grad():
        outputs = model(vla_emb, dino_emb)

    result = {
        "policy_state_shape": list(outputs["policy_state"].shape),
        "diffusion_cond_shape": list(outputs["diffusion_cond"].shape),
        "token_logits_shape": list(outputs["token_logits"].shape),
        "authority_logits_shape": list(outputs["authority_logits"].shape),
        "shared_repr_norm": float(outputs["shared_repr"].norm().item()),
        "authority_probs": outputs["authority_logits"].softmax(-1).mean(0).tolist(),
    }
    if include_shared_repr:
        result["shared_repr"] = to_json_safe(outputs["shared_repr"], include_tensors=True)
    return result


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training loss.

    Losses:
    1. Authority prediction (cross-entropy)
    2. Semantic token prediction (cross-entropy per position)
    """
    # Authority loss
    authority_loss = nn.functional.cross_entropy(
        outputs["authority_logits"], batch["authority_gt"]
    )

    # Semantic token loss (simplified: just predict first token correctly)
    # Full version would use teacher forcing or autoregressive decoding
    token_loss = nn.functional.cross_entropy(
        outputs["token_logits"][:, 0, :], batch["semantic_token_ids"][:, 0]
    )

    # Total loss
    total_loss = authority_loss + 0.5 * token_loss

    metrics = {
        "total_loss": total_loss.item(),
        "authority_loss": authority_loss.item(),
        "token_loss": token_loss.item(),
    }

    return total_loss, metrics


def evaluate_meta_transformer(
    model: MetaTransformerNet,
    dataloader: DataLoader,
) -> Dict[str, float]:
    """Evaluate meta-transformer on dataset."""
    model.eval()

    metrics = {
        "authority_acc": 0.0,
        "first_token_acc": 0.0,
        "total_loss": 0.0,
    }
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch["vla_embeddings"], batch["dino_embeddings"])

            # Authority accuracy
            pred_authority = outputs["authority_logits"].argmax(-1)
            metrics["authority_acc"] += (pred_authority == batch["authority_gt"]).sum().item()

            # First token accuracy
            pred_first_token = outputs["token_logits"][:, 0, :].argmax(-1)
            metrics["first_token_acc"] += (pred_first_token == batch["semantic_token_ids"][:, 0]).sum().item()

            # Loss
            loss, _ = compute_loss(outputs, batch)
            metrics["total_loss"] += loss.item()

            total_samples += batch["vla_embeddings"].shape[0]

    # Average
    metrics["authority_acc"] /= total_samples
    metrics["first_token_acc"] /= total_samples
    metrics["total_loss"] /= len(dataloader)

    return metrics
