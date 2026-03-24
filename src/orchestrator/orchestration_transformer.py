"""
Stub orchestration transformer that selects tool calls given context + instruction.

Scaffolding only; outputs tool logits over domain-specific tool names and a small
argument vector. No integration with RL/econ math.
"""
from typing import List

import torch
import torch.nn as nn
import numpy as np

from src.evidence.preconditions import build_execution_preconditions, build_execution_work_order
from src.orchestrator.toolspecs import ToolName, ToolCall, OrchestrationStep
from src.orchestrator.context import OrchestratorContext, OrchestratorResult
from src.orchestrator.semantic_transformer_bridge import (
    ORCHESTRATION_CTX_DIM,
    build_semantic_orchestration_plan,
    build_semantic_world_model_summary,
    build_tool_biases,
    derive_backend,
    derive_data_mix_weights,
    derive_energy_profile_mix,
    derive_objective_preset,
    encode_semantic_world_model_features,
    estimate_expected_deltas,
)


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
    def __init__(self, vocab_size: int = 128, hidden: int = 96, ctx_dim: int = ORCHESTRATION_CTX_DIM):
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

    base_vec = np.concatenate([
        engine,
        task,
        customer,
        np.array(ctx.objective_vector, dtype=np.float32),
        np.array([ctx.wage_human, ctx.energy_price_kWh], dtype=np.float32),
        np.array([ctx.mean_delta_mpl, ctx.mean_delta_error, ctx.mean_delta_j, ctx.mean_trust, ctx.mean_w_econ], dtype=np.float32),
        np.array(base_profile, dtype=np.float32),
    ])
    semantic_summary = build_semantic_world_model_summary(context=ctx)
    semantic_vec = encode_semantic_world_model_features(semantic_summary)
    vec = np.concatenate([base_vec, semantic_vec])
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
    instr_tokens = _hash_tokens(instruction, vocab)
    logits, _ = model(instr_tokens, ctx_vec)
    semantic_summary = build_semantic_world_model_summary(context=ctx)
    econ_signals = {
        "mpl_urgency": max(0.0, 1.0 - max(ctx.mean_delta_mpl, 0.0) / 60.0),
        "error_urgency": max(0.0, min(1.0, ctx.mean_delta_error * 10.0)),
        "energy_urgency": max(0.0, min(1.0, ctx.energy_price_kWh / 0.25)),
    }
    datapack_signals = {
        "data_coverage_score": max(0.0, min(1.0, ctx.mean_trust)),
        "embedding_diversity": max(0.0, min(1.0, abs(ctx.mean_w_econ))),
        "vla_annotation_fraction": 1.0 if semantic_summary.get("fusion_bridge", 0.0) >= 0.5 else 0.0,
        "guidance_annotation_fraction": 1.0 if semantic_summary.get("stage2_bridge", 0.0) >= 0.5 else 0.0,
        "data_gaps": list(ctx.semantic_metadata.get("data_gaps", []) or []),
    }
    objective_preset = derive_objective_preset(
        semantic_summary,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
        instruction=instruction,
    )
    energy_mix = derive_energy_profile_mix(
        semantic_summary,
        econ_signals=econ_signals,
        objective_preset=objective_preset,
    )
    data_mix = derive_data_mix_weights(
        semantic_summary,
        datapack_signals=datapack_signals,
    )
    chosen_backend = derive_backend(
        semantic_summary,
        econ_signals=econ_signals,
        current_backend=ctx.engine_type,
    )
    deltas = estimate_expected_deltas(
        semantic_summary,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
    )
    orchestration_plan = build_semantic_orchestration_plan(
        semantic_summary,
        objective_preset=objective_preset,
        data_mix_weights=data_mix,
        energy_profile_weights=energy_mix,
        datapack_signals=datapack_signals,
    )
    tool_biases = build_tool_biases(
        semantic_summary,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
        instruction=instruction,
    )
    tool_scores = {}
    for index, tool_name in enumerate(TOOL_NAMES):
        tool_scores[tool_name] = float(logits[0, index].item()) * 0.1 + float(tool_biases.get(tool_name, 0.0))
    ranked_tools = sorted(
        TOOL_NAMES,
        key=lambda name: (tool_scores.get(name, 0.0), -TOOL_NAMES.index(name)),
        reverse=True,
    )
    selected_set = set(ranked_tools[: max(steps, 1)])
    preferred_order = [
        "SET_OBJECTIVE_PRESET",
        "SET_ENERGY_PROFILE",
        "SET_DATA_MIX",
        "SET_BACKEND",
        "QUERY_DATAPACKS",
        "QUERY_ENERGY_SURFACE",
        "CALL_VLA_FOR_DATAPACK_CLASS",
        "CALL_VLA_SINGLE_STEP",
    ]
    chosen_tools = [tool_name for tool_name in preferred_order if tool_name in selected_set]
    for tool_name in ranked_tools:
        if tool_name not in chosen_tools:
            chosen_tools.append(tool_name)
        if len(chosen_tools) >= max(steps, 1):
            break
    steps_out = []
    for tool in chosen_tools[: max(steps, 1)]:
        args = {}
        if tool == "SET_BACKEND":
            args["backend"] = chosen_backend
        elif tool == "SET_OBJECTIVE_PRESET":
            args["preset"] = objective_preset
        elif tool == "SET_ENERGY_PROFILE":
            args["profile_mix"] = energy_mix
        elif tool == "SET_DATA_MIX":
            args["data_mix"] = data_mix
        elif tool == "QUERY_ENERGY_SURFACE":
            args["profile_query"] = True
        elif tool == "QUERY_DATAPACKS":
            args["filter"] = {"env": ctx.env_name, "engine": chosen_backend, "focus": datapack_signals["data_gaps"]}
        elif tool == "CALL_VLA_FOR_DATAPACK_CLASS":
            args["class_filter"] = list(semantic_summary.get("top_object_labels", []) or [])
        elif tool == "CALL_VLA_SINGLE_STEP":
            args["focus_meta_node"] = next(iter(semantic_summary.get("top_meta_nodes", []) or []), "")
        steps_out.append(
            OrchestrationStep(
                instruction=instruction,
                objective_vector=ctx.objective_vector,
                backend_id=chosen_backend,
                env_name=ctx.env_name,
                tool_call=tool_call_from(tool, args),
                observation=None,
            )
        )
    readiness = build_execution_preconditions(
        subject_id=str(semantic_summary.get("task_id") or ctx.task_type or ctx.env_name),
        subject_kind="orchestration_transformer",
        artifact_refs={"semantic_world_model_id": semantic_summary.get("world_model_id")},
        required_artifact_refs=["semantic_world_model_id"],
        signal_values={
            "semantic_present": 1.0 if semantic_summary.get("present") else 0.0,
            "object_count": semantic_summary.get("object_count", 0.0),
            "meta_node_count": semantic_summary.get("meta_node_count", 0.0),
            "capability_mean": semantic_summary.get("capability_mean", 0.0),
            "risk_reasoning": semantic_summary.get("risk_reasoning", 0.0),
        },
        min_signal_thresholds={
            "object_count": 1.0,
            "capability_mean": 0.15,
        },
        required_boolean_signals={"semantic_present": True},
        soft_min_signal_thresholds={
            "meta_node_count": 1.0,
            "risk_reasoning": 0.2,
        },
        metadata={
            "semantic_world_model_summary": semantic_summary,
            "tool_biases": tool_biases,
            "semantic_plan": orchestration_plan,
        },
    )
    execution_mode = "bounded_execution" if readiness.ready else "advisory"
    activation_plan = {
        "mode": execution_mode,
        "bounded_actions": [
            "set_objective_preset",
            "set_energy_profile",
            "set_data_mix",
            "set_backend",
        ],
        "tool_sequence": [step.tool_call.name for step in steps_out],
        "semantic_plan": orchestration_plan,
    }
    activation_work_order = build_execution_work_order(
        order_type="transformer_routing",
        subject_id=str(semantic_summary.get("task_id") or ctx.task_type or ctx.env_name),
        subject_kind="orchestration_transformer",
        decision="activate_orchestration_transformer",
        priority=float(max(semantic_summary.get("capability_mean", 0.0), 0.1)),
        recommended_mode=execution_mode,
        readiness=readiness,
        reasons=list(activation_plan["bounded_actions"]),
        artifact_refs={"semantic_world_model_id": semantic_summary.get("world_model_id")},
        metadata={"semantic_world_model_summary": semantic_summary},
    ).to_dict()

    return OrchestratorResult(
        steps=steps_out,
        chosen_backend=chosen_backend,
        energy_profile_weights=energy_mix,
        objective_preset=objective_preset,
        data_mix_weights=data_mix,
        expected_delta_mpl=deltas["expected_delta_mpl"],
        expected_delta_error=deltas["expected_delta_error"],
        expected_delta_energy_Wh=deltas["expected_delta_energy_Wh"],
        execution_mode=execution_mode,
        activation_plan=activation_plan,
        activation_work_order=activation_work_order,
        metadata={
            "semantic_world_model_summary": semantic_summary,
            "tool_biases": tool_biases,
            "semantic_plan": orchestration_plan,
            "execution_preconditions": readiness.to_dict(),
        },
    )


def tool_call_from(name: ToolName, args: dict) -> ToolCall:
    return ToolCall(name=name, args=args)
