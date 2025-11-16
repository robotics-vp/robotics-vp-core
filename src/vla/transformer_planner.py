"""
VLA Transformer Planner.

Maps language instructions + visual observations to skill sequences.

SIMA-style "L" head for hierarchical planning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass

from src.hrl.skills import SkillID, SkillParams


@dataclass
class VLAInput:
    """Input to VLA transformer."""
    instruction: str
    z_v: Optional[np.ndarray] = None  # (z_v_dim,) visual latent
    state: Optional[np.ndarray] = None  # (obs_dim,) optional state
    risk_map: Optional[np.ndarray] = None  # (H, W) optional risk map
    affordance_map: Optional[np.ndarray] = None  # (H, W) optional affordance


@dataclass
class VLAPlan:
    """Output from VLA transformer."""
    skill_sequence: List[int] = field(default_factory=list)
    skill_params: List[np.ndarray] = field(default_factory=list)
    timing_horizons: List[int] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    instruction: str = ""

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'skill_sequence': self.skill_sequence,
            'skill_params': [p.tolist() for p in self.skill_params],
            'timing_horizons': self.timing_horizons,
            'confidence': self.confidence,
            'instruction': self.instruction,
        }

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(
            skill_sequence=d['skill_sequence'],
            skill_params=[np.array(p) for p in d['skill_params']],
            timing_horizons=d['timing_horizons'],
            confidence=d['confidence'],
            instruction=d.get('instruction', ''),
        )

    def __str__(self):
        skills = [SkillID.name(sid) for sid in self.skill_sequence]
        return f"VLAPlan({' -> '.join(skills)})"


class SimpleTokenizer:
    """
    Simple word-level tokenizer for VLA.

    In production, use GPT-2/BERT tokenizer.
    """

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.next_id = 4

        # Pre-populate with common words
        common_words = [
            'open', 'close', 'the', 'drawer', 'vase', 'without', 'hitting',
            'carefully', 'avoid', 'fragile', 'top', 'bottom', 'left', 'right',
            'grasp', 'pull', 'push', 'handle', 'slowly', 'quickly', 'safe',
            'while', 'maintaining', 'clearance', 'from', 'and', 'to', 'a'
        ]
        for word in common_words:
            self._add_word(word)

    def _add_word(self, word):
        """Add word to vocabulary."""
        if word not in self.word_to_id and self.next_id < self.vocab_size:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1

    def encode(self, text, max_length=32):
        """
        Encode text to token IDs.

        Args:
            text: Input string
            max_length: Maximum sequence length

        Returns:
            token_ids: List of integers
        """
        words = text.lower().split()

        # Add unknown words to vocab
        for word in words:
            self._add_word(word)

        # Convert to IDs
        ids = [self.word_to_id.get('<START>')]
        for word in words[:max_length - 2]:
            ids.append(self.word_to_id.get(word, self.word_to_id['<UNK>']))
        ids.append(self.word_to_id.get('<END>'))

        # Pad to max_length
        while len(ids) < max_length:
            ids.append(self.word_to_id['<PAD>'])

        return ids[:max_length]

    def decode(self, token_ids):
        """
        Decode token IDs to text.

        Args:
            token_ids: List of integers

        Returns:
            text: Decoded string
        """
        words = []
        for tid in token_ids:
            word = self.id_to_word.get(tid, '<UNK>')
            if word not in ['<PAD>', '<START>', '<END>']:
                words.append(word)
        return ' '.join(words)


class VLATransformerPlanner(nn.Module if TORCH_AVAILABLE else object):
    """
    Vision-Language-Action Transformer.

    Maps language instructions + visual features to skill sequences.

    Architecture:
    - Text encoder (word embeddings + positional)
    - Vision encoder (project z_V to embedding space)
    - Transformer encoder (cross-attention between text and vision)
    - Autoregressive skill decoder
    """

    def __init__(
        self,
        vocab_size=1000,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=32,
        z_v_dim=128,
        obs_dim=13,
        num_skills=6,
        max_plan_length=10,
        map_dim=256  # 16x16 flattened
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VLATransformerPlanner")

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_skills = num_skills
        self.max_plan_length = max_plan_length
        self.max_seq_len = max_seq_len

        # Tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size)

        # Text embeddings
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Vision projections
        self.z_v_proj = nn.Linear(z_v_dim, embed_dim)
        self.state_proj = nn.Linear(obs_dim, embed_dim)
        self.risk_proj = nn.Linear(map_dim, embed_dim)
        self.affordance_proj = nn.Linear(map_dim, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Skill sequence decoder
        # Autoregressive: each position predicts next skill
        self.skill_start_token = nn.Parameter(torch.randn(1, embed_dim))
        self.skill_embedding = nn.Embedding(num_skills + 1, embed_dim)  # +1 for EOS

        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2
        )

        # Output heads
        self.skill_head = nn.Linear(embed_dim, num_skills + 1)  # +1 for EOS
        self.param_head = nn.Linear(embed_dim, 5)  # Skill parameters
        self.timing_head = nn.Linear(embed_dim, 1)  # Timing horizon
        self.confidence_head = nn.Linear(embed_dim, 1)  # Confidence score

        # EOS token ID
        self.eos_id = num_skills

    def encode_instruction(self, token_ids):
        """
        Encode text instruction.

        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            text_features: (batch, seq_len, embed_dim)
        """
        seq_len = token_ids.shape[1]
        device = token_ids.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        text_emb = self.text_embedding(token_ids)  # (batch, seq_len, embed_dim)
        pos_emb = self.pos_embedding(pos_ids)      # (1, seq_len, embed_dim)

        return text_emb + pos_emb

    def encode_vision(self, z_v=None, state=None, risk_map=None, affordance_map=None):
        """
        Encode visual and state features.

        Args:
            z_v: (batch, z_v_dim) visual latent
            state: (batch, obs_dim) state observation
            risk_map: (batch, H, W) risk map
            affordance_map: (batch, H, W) affordance map

        Returns:
            vision_tokens: (batch, num_tokens, embed_dim)
        """
        tokens = []
        batch_size = None

        if z_v is not None:
            batch_size = z_v.shape[0]
            z_v_token = self.z_v_proj(z_v).unsqueeze(1)  # (batch, 1, embed_dim)
            tokens.append(z_v_token)

        if state is not None:
            if batch_size is None:
                batch_size = state.shape[0]
            state_token = self.state_proj(state).unsqueeze(1)
            tokens.append(state_token)

        if risk_map is not None:
            if batch_size is None:
                batch_size = risk_map.shape[0]
            risk_flat = risk_map.view(batch_size, -1)
            risk_token = self.risk_proj(risk_flat).unsqueeze(1)
            tokens.append(risk_token)

        if affordance_map is not None:
            if batch_size is None:
                batch_size = affordance_map.shape[0]
            aff_flat = affordance_map.view(batch_size, -1)
            aff_token = self.affordance_proj(aff_flat).unsqueeze(1)
            tokens.append(aff_token)

        if len(tokens) == 0:
            # No vision features, return empty
            return torch.zeros(1, 0, self.embed_dim)

        vision_tokens = torch.cat(tokens, dim=1)  # (batch, num_tokens, embed_dim)
        return vision_tokens

    def forward(self, token_ids, z_v=None, state=None, risk_map=None, affordance_map=None):
        """
        Generate skill plan.

        Args:
            token_ids: (batch, seq_len) text token IDs
            z_v: Optional visual latent
            state: Optional state observation
            risk_map: Optional risk map
            affordance_map: Optional affordance map

        Returns:
            skill_logits: (batch, max_plan_length, num_skills+1)
            skill_params: (batch, max_plan_length, 5)
            timing: (batch, max_plan_length)
            confidence: (batch, max_plan_length)
        """
        batch_size = token_ids.shape[0]
        device = token_ids.device

        # Encode text
        text_features = self.encode_instruction(token_ids)  # (batch, seq_len, embed_dim)

        # Encode vision
        vision_tokens = self.encode_vision(z_v, state, risk_map, affordance_map)

        # Concatenate text and vision
        if vision_tokens.shape[1] > 0:
            encoder_input = torch.cat([text_features, vision_tokens], dim=1)
        else:
            encoder_input = text_features

        # Transformer encoder
        memory = self.transformer(encoder_input)  # (batch, seq_len+num_vis, embed_dim)

        # Autoregressive decoding
        # Start with start token
        decoder_input = self.skill_start_token.expand(batch_size, 1, -1)

        all_skill_logits = []
        all_params = []
        all_timing = []
        all_confidence = []

        for step in range(self.max_plan_length):
            # Decode
            output = self.decoder(decoder_input, memory)  # (batch, step+1, embed_dim)

            # Get predictions from last position
            last_output = output[:, -1, :]  # (batch, embed_dim)

            skill_logit = self.skill_head(last_output)  # (batch, num_skills+1)
            params = torch.sigmoid(self.param_head(last_output))  # (batch, 5)
            timing = F.relu(self.timing_head(last_output)).squeeze(-1) * 100  # (batch,)
            conf = torch.sigmoid(self.confidence_head(last_output)).squeeze(-1)  # (batch,)

            all_skill_logits.append(skill_logit)
            all_params.append(params)
            all_timing.append(timing)
            all_confidence.append(conf)

            # Prepare next decoder input (teacher forcing with predicted skill)
            predicted_skill = skill_logit.argmax(dim=-1)  # (batch,)
            skill_emb = self.skill_embedding(predicted_skill).unsqueeze(1)  # (batch, 1, embed_dim)
            decoder_input = torch.cat([decoder_input, skill_emb], dim=1)

        # Stack outputs
        skill_logits = torch.stack(all_skill_logits, dim=1)  # (batch, max_plan_length, num_skills+1)
        skill_params = torch.stack(all_params, dim=1)  # (batch, max_plan_length, 5)
        timing = torch.stack(all_timing, dim=1)  # (batch, max_plan_length)
        confidence = torch.stack(all_confidence, dim=1)  # (batch, max_plan_length)

        return skill_logits, skill_params, timing, confidence

    def plan(self, vla_input: VLAInput, device='cpu'):
        """
        Generate plan from VLAInput.

        Args:
            vla_input: VLAInput object
            device: Compute device

        Returns:
            plan: VLAPlan object
        """
        # Tokenize instruction
        token_ids = self.tokenizer.encode(vla_input.instruction)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Prepare optional inputs
        z_v = None
        state = None
        risk_map = None
        affordance_map = None

        if vla_input.z_v is not None:
            z_v = torch.FloatTensor(vla_input.z_v).unsqueeze(0).to(device)

        if vla_input.state is not None:
            state = torch.FloatTensor(vla_input.state).unsqueeze(0).to(device)

        if vla_input.risk_map is not None:
            risk_map = torch.FloatTensor(vla_input.risk_map).unsqueeze(0).to(device)

        if vla_input.affordance_map is not None:
            affordance_map = torch.FloatTensor(vla_input.affordance_map).unsqueeze(0).to(device)

        # Generate plan
        self.eval()
        with torch.no_grad():
            skill_logits, skill_params, timing, confidence = self.forward(
                token_ids, z_v, state, risk_map, affordance_map
            )

        # Extract plan (stopping at EOS)
        skill_sequence = []
        param_list = []
        timing_list = []
        conf_list = []

        for i in range(self.max_plan_length):
            skill_id = skill_logits[0, i].argmax().item()

            if skill_id == self.eos_id:
                break

            skill_sequence.append(skill_id)
            param_list.append(skill_params[0, i].cpu().numpy())
            timing_list.append(int(timing[0, i].item()))
            conf_list.append(float(confidence[0, i].item()))

        plan = VLAPlan(
            skill_sequence=skill_sequence,
            skill_params=param_list,
            timing_horizons=timing_list,
            confidence=conf_list,
            instruction=vla_input.instruction
        )

        return plan

    def compute_loss(self, token_ids, gt_skill_sequence, gt_params=None,
                     z_v=None, state=None, risk_map=None, affordance_map=None):
        """
        Compute training loss.

        Args:
            token_ids: (batch, seq_len) input tokens
            gt_skill_sequence: (batch, plan_len) ground truth skills
            gt_params: Optional (batch, plan_len, 5) ground truth params
            z_v, state, risk_map, affordance_map: Optional features

        Returns:
            loss: Scalar loss
            metrics: dict with individual losses
        """
        skill_logits, pred_params, pred_timing, pred_conf = self.forward(
            token_ids, z_v, state, risk_map, affordance_map
        )

        # Skill prediction loss (cross-entropy)
        batch_size = token_ids.shape[0]
        plan_len = gt_skill_sequence.shape[1]

        # Pad ground truth to max_plan_length with EOS
        gt_padded = torch.full(
            (batch_size, self.max_plan_length),
            self.eos_id,
            device=token_ids.device,
            dtype=torch.long
        )
        gt_padded[:, :plan_len] = gt_skill_sequence

        # Cross-entropy loss
        skill_loss = F.cross_entropy(
            skill_logits.view(-1, self.num_skills + 1),
            gt_padded.view(-1)
        )

        metrics = {'skill_loss': skill_loss.item()}
        total_loss = skill_loss

        # Parameter loss (if provided)
        if gt_params is not None:
            # Only compute for non-EOS positions
            param_loss = F.mse_loss(pred_params[:, :plan_len], gt_params)
            metrics['param_loss'] = param_loss.item()
            total_loss = total_loss + param_loss * 0.1

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_skills': self.num_skills,
            'max_plan_length': self.max_plan_length,
            'tokenizer_word_to_id': self.tokenizer.word_to_id,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['embed_dim'],
            num_skills=checkpoint['num_skills'],
            max_plan_length=checkpoint['max_plan_length']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.tokenizer.word_to_id = checkpoint['tokenizer_word_to_id']
        model.tokenizer.id_to_word = {v: k for k, v in checkpoint['tokenizer_word_to_id'].items()}
        model.to(device)
        return model


class RuleBasedVLAPlanner:
    """
    Rule-based VLA planner for baseline comparison.

    Parses instruction text to generate skill sequences.
    """

    def __init__(self):
        self.default_plan = [
            SkillID.LOCATE_DRAWER,
            SkillID.LOCATE_VASE,
            SkillID.PLAN_SAFE_APPROACH,
            SkillID.GRASP_HANDLE,
            SkillID.OPEN_WITH_CLEARANCE,
            SkillID.RETRACT_SAFE,
        ]

    def plan(self, vla_input: VLAInput):
        """
        Generate plan from instruction using rules.

        Args:
            vla_input: VLAInput object

        Returns:
            plan: VLAPlan object
        """
        instruction = vla_input.instruction.lower()

        # Parse instruction for modifiers
        skill_sequence = []
        skill_params = []
        timing = []
        confidence = []

        # Default: standard drawer+vase task
        if 'drawer' in instruction:
            # Check for safety modifiers
            if 'carefully' in instruction or 'safe' in instruction:
                clearance = 0.2  # Larger clearance
            else:
                clearance = 0.15

            # Check for speed modifiers
            if 'quickly' in instruction or 'fast' in instruction:
                speed = 0.9
            elif 'slowly' in instruction:
                speed = 0.4
            else:
                speed = 0.6

            # Build plan
            for skill_id in self.default_plan:
                skill_sequence.append(skill_id)

                params = SkillParams.default_for_skill(skill_id)
                params.target_clearance = clearance
                if skill_id == SkillID.OPEN_WITH_CLEARANCE:
                    params.pull_speed = speed
                skill_params.append(params.to_array())

                timing.append(params.timeout_steps)
                confidence.append(0.8)  # Fixed confidence

        else:
            # Unknown task, return empty plan
            pass

        return VLAPlan(
            skill_sequence=skill_sequence,
            skill_params=skill_params,
            timing_horizons=timing,
            confidence=confidence,
            instruction=vla_input.instruction
        )
