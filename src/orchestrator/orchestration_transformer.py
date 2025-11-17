"""
Stub orchestration transformer that selects tool calls given context + instruction.

Scaffolding only; outputs tool logits over domain-specific tool names and a small
argument vector. No integration with RL/econ math.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

from src.orchestrator.toolspecs import ToolName, ToolCall, OrchestrationStep
from src.orchestrator.context import OrchestratorContext, OrchestratorResult


TOOL_NAMES: List[ToolName] = [
    "SET_ENERGY_PROFILE",
    "SET_OBJECTIVE_PRESET",
    "SET_BACKEND",
    "SET_DATA_MIX",
    "QUERY_DATAPACKS",
    "QUERY_ENERGY_SURFACE",
    "CALL_VLA_SINGLE_STEP",
    "CALL_VLA_FOR_DATAPACK_CLASS",
]


class OrchestrationTransformer(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden: int = 96, ctx_dim: int = 36):
        super().__init__()
        self.instr_embed = nn.Embedding(vocab_size, hidden)
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.tool_head = nn.Linear(hidden, len(TOOL_NAMES))
        self.arg_head = nn.Linear(hidden, 12)   # a few continuous knobs as stub

    def forward(self, instr_tokens: torch.Tensor, ctx_vec: torch.Tensor):
        """
        instr_tokens: (B, T) token ids (stub)
        ctx_vec: (B, ctx_dim) flattened context features
        """
        instr_emb = self.instr_embed(instr_tokens).mean(dim=1)
        ctx_emb = self.ctx_proj(ctx_vec)
        h = torch.cat([instr_emb, ctx_emb], dim=-1)
        h = self.mlp(h)
        tool_logits = self.tool_head(h)
        arg_vec = self.arg_head(h)
        return tool_logits, arg_vec


def decode_tool(logits: torch.Tensor) -> ToolName:
    idx = int(torch.argmax(logits, dim=-1).item())
    return TOOL_NAMES[idx % len(TOOL_NAMES)]


def _hash_tokens(instr: str, vocab_size: int, max_len: int = 16) -> torch.Tensor:
    toks = instr.lower().split()
    ids = [(abs(hash(t)) % (vocab_size - 1)) + 1 for t in toks[:max_len]]
    if not ids:
        ids = [0]
    return torch.tensor([ids], dtype=torch.long)


def _encode_ctx(ctx: OrchestratorContext) -> np.ndarray:
    # One-hots for engine/task/customer as simple hashes into fixed slots
    def oh(val, size, offset):
        vec = np.zeros(size, dtype=np.float32)
        vec[(abs(hash(val)) % size)] = 1.0
        return vec

    engine = oh(ctx.engine_type, 4, 0)
    task = oh(ctx.task_type, 4, 0)
    customer = oh(ctx.customer_segment, 4, 0)

    base_profile = []
    for name in ["BASE", "BOOST", "SAVER", "SAFE"]:
        prof = ctx.profile_summaries.get(name, {})
        base_profile.extend([
            prof.get("mpl", 0.0),
            prof.get("error", 0.0),
            prof.get("energy_Wh", 0.0),
        ])

    vec = np.concatenate([
        engine,
        task,
        customer,
        np.array(ctx.objective_vector, dtype=np.float32),
        np.array([ctx.wage_human, ctx.energy_price_kWh], dtype=np.float32),
        np.array([ctx.mean_delta_mpl, ctx.mean_delta_error, ctx.mean_delta_j, ctx.mean_trust, ctx.mean_w_econ], dtype=np.float32),
        np.array(base_profile, dtype=np.float32),
    ])
    return vec.astype(np.float32)


def propose_orchestrated_plan(model: OrchestrationTransformer, ctx: OrchestratorContext, instruction: str, steps: int = 4) -> OrchestratorResult:
    vocab = model.instr_embed.num_embeddings
    ctx_vec = torch.from_numpy(_encode_ctx(ctx))
    # Pad/trim ctx to expected dim
    if ctx_vec.numel() < model.ctx_proj.in_features:
        pad = torch.zeros(model.ctx_proj.in_features - ctx_vec.numel())
        ctx_vec = torch.cat([ctx_vec, pad])
    elif ctx_vec.numel() > model.ctx_proj.in_features:
        ctx_vec = ctx_vec[: model.ctx_proj.in_features]
    ctx_vec = ctx_vec.unsqueeze(0)
    steps_out = []
    chosen_backend = ctx.engine_type
    energy_mix = {"BASE": 0.25, "BOOST": 0.25, "SAVER": 0.25, "SAFE": 0.25}
    objective_preset = "balanced"
    data_mix = {"real": 0.6, "synthetic": 0.3, "hybrid": 0.1}
    delta_mpl = 0.0
    delta_error = 0.0
    delta_energy = 0.0

    instr_tokens = _hash_tokens(instruction, vocab)
    for _ in range(steps):
        logits, arg_vec = model(instr_tokens, ctx_vec)
        arg_vec = arg_vec.detach()
        tool = decode_tool(logits)
        args = {}
        if tool == "SET_BACKEND":
            args["backend"] = ctx.engine_type
            chosen_backend = ctx.engine_type
        elif tool == "SET_OBJECTIVE_PRESET":
            # map arg_vec[0] to preset
            presets = ["throughput", "energy_saver", "safety_first", "balanced"]
            idx = int(abs(arg_vec[0, 0].item()) * 10) % len(presets)
            objective_preset = presets[idx]
            args["preset"] = objective_preset
        elif tool == "SET_ENERGY_PROFILE":
            # softmax over first 4 args
            weights = torch.softmax(arg_vec[0, :4], dim=0).cpu().numpy()
            energy_mix = {k: float(v) for k, v in zip(["BASE", "BOOST", "SAVER", "SAFE"], weights)}
            args["profile_mix"] = energy_mix
        elif tool == "SET_DATA_MIX":
            weights = torch.softmax(arg_vec[0, 4:7], dim=0).cpu().numpy()
            data_mix = {k: float(v) for k, v in zip(["real", "synthetic", "hybrid"], weights)}
            args["data_mix"] = data_mix
        elif tool == "QUERY_ENERGY_SURFACE":
            args["profile_query"] = True
        elif tool == "QUERY_DATAPACKS":
            args["filter"] = {"env": ctx.env_name, "engine": ctx.engine_type}
        steps_out.append(
            OrchestrationStep(
                instruction=instruction,
                objective_vector=ctx.objective_vector,
                backend_id=ctx.engine_type,
                env_name=ctx.env_name,
                tool_call=tool_call_from(tool, args),
                observation=None,
            )
        )

    # Estimate expected deltas from profile mix
    for name, w in energy_mix.items():
        prof = ctx.profile_summaries.get(name, {})
        delta_mpl += w * prof.get("mpl", 0.0)
        delta_error += w * prof.get("error", 0.0)
        delta_energy += w * prof.get("energy_Wh", 0.0)

    return OrchestratorResult(
        steps=steps_out,
        chosen_backend=chosen_backend,
        energy_profile_weights=energy_mix,
        objective_preset=objective_preset,
        data_mix_weights=data_mix,
        expected_delta_mpl=delta_mpl,
        expected_delta_error=delta_error,
        expected_delta_energy_Wh=delta_energy,
    )


def tool_call_from(name: ToolName, args: dict) -> ToolCall:
    return ToolCall(name=name, args=args)
