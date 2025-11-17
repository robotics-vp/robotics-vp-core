"""
Orchestration Transformer Training Dataset Generator.

Builds (context_features, target_tool_sequence) pairs from:
- Datapacks (ObjectiveProfile, GuidanceProfile, energy metrics)
- Energy interventions / EnergyResponseNet outputs
- Offline solver outputs (best configs per objective + constraints)

Uses heuristic "teacher" policy to generate target tool sequences.

This is the supervised learning signal for the orchestration transformer.
No Phase B / RL / reward math changes - purely advisory.
"""

import json
import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.orchestrator.context import OrchestratorContext
from src.orchestrator.orchestration_transformer import TOOL_NAMES, _encode_ctx
from src.orchestrator.toolspecs import ToolCall
from src.valuation.datapack_schema import DataPackMeta
from src.orchestrator.semantic_metrics import SemanticMetrics


@dataclass
class EconSemanticDecisionSummary:
    """
    Auxiliary supervision target: econ/semantic decision summary.

    This is the "ground truth" from econ/datapack/semantic feedback loop
    that the orchestration transformer should learn to predict.
    """
    chosen_profile: str  # "SAFE" | "SAVER" | "BASE" | "BOOST"
    objective_preset: str  # "balanced" | "safety" | "throughput" | "energy_saver"
    pareto_classification: str  # "energy_tight" | "mpl_tight" | "balanced" | "safety_focused"
    urgency_level: str  # "none" | "moderate" | "high" | "critical"
    recommended_focus: str  # "throughput_demonstrations" | "safety_edge_cases" | etc.
    semantic_priority_fraction: float  # Fraction of tasks with high/critical priority
    data_coverage_score: float  # From DatapackSignals
    wage_parity: float  # From EconSignals


@dataclass
class OrchestrationSample:
    """Single training sample: context -> tool sequence."""

    context: OrchestratorContext
    context_features: np.ndarray  # Flattened feature vector
    target_tool_sequence: List[ToolCall]  # Heuristic-generated sequence
    heuristic_rationale: List[str]  # Why each tool was selected
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Econ/semantic decision summary (auxiliary supervision target)
    econ_semantic_summary: Optional[EconSemanticDecisionSummary] = None
    # Source type: "heuristic" or "econ_semantic"
    source_type: str = "heuristic"


@dataclass
class HeuristicDecision:
    """Decision made by heuristic teacher."""

    tool: str
    args: Dict[str, Any]
    rationale: str


def heuristic_energy_profile_decision(ctx: OrchestratorContext) -> HeuristicDecision:
    """
    Decide energy profile based on energy price and market conditions.

    Rules:
    - High energy price -> SAVER or SAFE profile
    - Low energy price -> BASE or BOOST profile
    - Safety-focused customer -> SAFE profile
    - Throughput-focused customer -> BOOST profile
    """
    energy_price = ctx.energy_price_kWh

    if energy_price > 0.20:  # High energy cost
        profile = "SAVER"
        rationale = f"High energy price (${energy_price}/kWh) -> SAVER profile"
    elif energy_price > 0.15:
        if "safety" in ctx.customer_segment.lower():
            profile = "SAFE"
            rationale = f"Moderate energy + safety customer -> SAFE profile"
        else:
            profile = "SAVER"
            rationale = f"Moderate energy price -> SAVER profile"
    else:  # Low energy cost
        if "throughput" in ctx.customer_segment.lower():
            profile = "BOOST"
            rationale = f"Low energy + throughput customer -> BOOST profile"
        else:
            profile = "BASE"
            rationale = f"Low energy price -> BASE profile"

    # Safety override
    if "safety" in ctx.customer_segment.lower() or ctx.objective_vector[3] > 2.0:
        profile = "SAFE"
        rationale = f"Safety-focused objective (w_safety={ctx.objective_vector[3]}) -> SAFE profile"

    profile_weights = {"BASE": 0.0, "BOOST": 0.0, "SAVER": 0.0, "SAFE": 0.0}
    profile_weights[profile] = 1.0

    return HeuristicDecision(
        tool="SET_ENERGY_PROFILE",
        args={"profile_mix": profile_weights},
        rationale=rationale,
    )


def heuristic_objective_preset_decision(ctx: OrchestratorContext) -> HeuristicDecision:
    """
    Decide objective preset based on context metrics.

    Rules:
    - MPL below human baseline + low error -> throughput preset
    - High error rate -> safety preset
    - High energy consumption -> energy_saver preset
    - Default to balanced
    """
    mpl_gap = ctx.mean_delta_mpl
    error_trend = ctx.mean_delta_error
    w_mpl, w_error, w_energy, w_safety, _ = ctx.objective_vector

    # Priority: check for specific conditions
    if error_trend > 0.05:  # Error rate increasing
        preset = "safety"
        rationale = f"Error increasing (Δerror={error_trend:.3f}) -> safety preset"
    elif w_safety > 2.0:  # High safety weight in objective
        preset = "safety"
        rationale = f"High safety weight (w_safety={w_safety}) -> safety preset"
    elif mpl_gap < -5.0 and error_trend <= 0:  # MPL low but errors ok
        preset = "throughput"
        rationale = f"Low MPL (ΔMPL={mpl_gap:.2f}), errors stable -> throughput preset"
    elif w_energy > 2.0:  # High energy weight
        preset = "energy_saver"
        rationale = f"High energy weight (w_energy={w_energy}) -> energy_saver preset"
    elif w_mpl > 2.0:  # High throughput weight
        preset = "throughput"
        rationale = f"High MPL weight (w_mpl={w_mpl}) -> throughput preset"
    else:
        preset = "balanced"
        rationale = "No strong signal -> balanced preset"

    return HeuristicDecision(
        tool="SET_OBJECTIVE_PRESET",
        args={"preset": preset},
        rationale=rationale,
    )


def heuristic_backend_decision(ctx: OrchestratorContext) -> HeuristicDecision:
    """
    Decide which physics backend to use.

    Rules:
    - If Isaac is available and env supports it -> ISAAC (more accurate energy)
    - Otherwise -> PYBULLET
    - For energy analysis tasks -> prefer ISAAC
    """
    if ctx.engine_type == "isaac":
        backend = "isaac"
        rationale = "Isaac backend available -> use for accurate energy metrics"
    else:
        # Check if context suggests we need better energy modeling
        w_energy = ctx.objective_vector[2]
        if w_energy > 1.5:
            backend = "isaac"  # Would prefer, but may not be available
            rationale = f"High energy weight (w_energy={w_energy}) -> prefer Isaac for accuracy"
        else:
            backend = "pybullet"
            rationale = "Standard PyBullet backend sufficient"

    return HeuristicDecision(
        tool="SET_BACKEND",
        args={"backend": backend},
        rationale=rationale,
    )


def heuristic_data_mix_decision(ctx: OrchestratorContext) -> HeuristicDecision:
    """
    Decide data mix weights for training.

    Rules:
    - Low trust / low coverage -> favor synthetic data
    - High trust + high w_econ -> favor real data
    - Negative bucket heavy -> increase synthetic for balance
    """
    trust = ctx.mean_trust
    w_econ = ctx.mean_w_econ
    delta_j = ctx.mean_delta_j

    if trust < 0.5:
        # Low trust: need more synthetic for exploration
        mix = {"real": 0.3, "synthetic": 0.6, "hybrid": 0.1}
        rationale = f"Low trust ({trust:.2f}) -> favor synthetic data"
    elif w_econ < 0.3:
        # Low economic weight: current data not useful
        mix = {"real": 0.4, "synthetic": 0.5, "hybrid": 0.1}
        rationale = f"Low w_econ ({w_econ:.2f}) -> increase synthetic"
    elif delta_j < 0:
        # Negative J trend: diversify with synthetic
        mix = {"real": 0.5, "synthetic": 0.4, "hybrid": 0.1}
        rationale = f"Negative ΔJ ({delta_j:.3f}) -> balance with synthetic"
    else:
        # Good data quality: favor real
        mix = {"real": 0.7, "synthetic": 0.2, "hybrid": 0.1}
        rationale = f"Good trust ({trust:.2f}) + w_econ ({w_econ:.2f}) -> favor real data"

    return HeuristicDecision(
        tool="SET_DATA_MIX",
        args={"data_mix": mix},
        rationale=rationale,
    )


def generate_heuristic_tool_sequence(ctx: OrchestratorContext) -> List[HeuristicDecision]:
    """
    Generate target tool sequence using heuristic policy.

    This is the "teacher" that the transformer learns to imitate.

    Returns:
        List of HeuristicDecision objects (tool + args + rationale)
    """
    decisions = []

    # 1. Set backend (if relevant)
    backend_decision = heuristic_backend_decision(ctx)
    decisions.append(backend_decision)

    # 2. Set objective preset (based on context)
    objective_decision = heuristic_objective_preset_decision(ctx)
    decisions.append(objective_decision)

    # 3. Set energy profile (based on price and safety)
    energy_decision = heuristic_energy_profile_decision(ctx)
    decisions.append(energy_decision)

    # 4. Set data mix (based on trust and coverage)
    data_mix_decision = heuristic_data_mix_decision(ctx)
    decisions.append(data_mix_decision)

    return decisions


def context_to_sample(ctx: OrchestratorContext) -> OrchestrationSample:
    """
    Convert OrchestratorContext to training sample with heuristic labels.

    Args:
        ctx: OrchestratorContext from datapacks or synthetic

    Returns:
        OrchestrationSample with features and target sequence
    """
    # Encode context to features
    context_features = _encode_ctx(ctx)

    # Generate target sequence using heuristic
    decisions = generate_heuristic_tool_sequence(ctx)

    # Convert to ToolCall format
    target_sequence = [
        ToolCall(name=d.tool, args=d.args) for d in decisions
    ]

    rationales = [d.rationale for d in decisions]

    return OrchestrationSample(
        context=ctx,
        context_features=context_features,
        target_tool_sequence=target_sequence,
        heuristic_rationale=rationales,
        metadata={
            "num_tools": len(target_sequence),
            "tools_used": [d.tool for d in decisions],
        },
    )


def generate_synthetic_context(seed: int = None) -> OrchestratorContext:
    """
    Generate synthetic OrchestratorContext for dataset augmentation.

    Randomizes:
    - Objective vector weights
    - Economic parameters (energy price, wage)
    - Customer segment
    - Datapack aggregates (trust, w_econ, deltas)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Random objective vector
    w_mpl = random.uniform(0.5, 3.0)
    w_error = random.uniform(0.5, 3.0)
    w_energy = random.uniform(0.3, 3.0)
    w_safety = random.uniform(0.5, 5.0)
    w_novelty = 0.0
    objective_vector = [w_mpl, w_error, w_energy, w_safety, w_novelty]

    # Random economic context
    energy_price_kWh = random.uniform(0.08, 0.30)
    wage_human = random.uniform(15.0, 25.0)

    # Random customer segment
    segments = ["balanced", "throughput_focused", "premium_safety", "energy_saver"]
    customer_segment = random.choice(segments)

    # Random datapack aggregates
    mean_delta_mpl = random.uniform(-10.0, 10.0)
    mean_delta_error = random.uniform(-0.1, 0.1)
    mean_delta_j = random.uniform(-5.0, 5.0)
    mean_trust = random.uniform(0.3, 1.0)
    mean_w_econ = random.uniform(0.2, 1.0)

    # Random profile summaries
    profile_summaries = {
        "BASE": {
            "mpl": random.uniform(50, 70),
            "error": random.uniform(0.02, 0.08),
            "energy_Wh": random.uniform(10, 30),
            "risk": random.uniform(0.1, 0.3),
        },
        "BOOST": {
            "mpl": random.uniform(60, 85),
            "error": random.uniform(0.05, 0.12),
            "energy_Wh": random.uniform(20, 50),
            "risk": random.uniform(0.2, 0.5),
        },
        "SAVER": {
            "mpl": random.uniform(40, 60),
            "error": random.uniform(0.02, 0.06),
            "energy_Wh": random.uniform(5, 20),
            "risk": random.uniform(0.1, 0.3),
        },
        "SAFE": {
            "mpl": random.uniform(30, 50),
            "error": random.uniform(0.01, 0.04),
            "energy_Wh": random.uniform(8, 25),
            "risk": random.uniform(0.05, 0.15),
        },
    }

    return OrchestratorContext(
        env_name="drawer_vase",
        engine_type=random.choice(["pybullet", "isaac"]),
        task_type="drawer_open",
        customer_segment=customer_segment,
        market_region=random.choice(["US", "EU", "APAC"]),
        objective_vector=objective_vector,
        wage_human=wage_human,
        energy_price_kWh=energy_price_kWh,
        mean_delta_mpl=mean_delta_mpl,
        mean_delta_error=mean_delta_error,
        mean_delta_j=mean_delta_j,
        mean_trust=mean_trust,
        mean_w_econ=mean_w_econ,
        profile_summaries=profile_summaries,
    )


def build_training_dataset(
    num_samples: int = 1000,
    real_datapacks: Optional[List[DataPackMeta]] = None,
    save_path: Optional[str] = None,
) -> List[OrchestrationSample]:
    """
    Build complete training dataset for orchestration transformer.

    Args:
        num_samples: Number of samples to generate
        real_datapacks: Optional list of real datapacks to extract contexts from
        save_path: Optional path to save dataset as JSON

    Returns:
        List of OrchestrationSample objects
    """
    samples = []

    # Use real datapacks if provided
    if real_datapacks:
        for dp in real_datapacks[:num_samples // 2]:
            if dp.objective_profile:
                ctx = OrchestratorContext(
                    env_name=dp.objective_profile.env_name or dp.task_name,
                    engine_type=dp.objective_profile.engine_type or dp.env_type,
                    task_type=dp.objective_profile.task_type or "unknown",
                    customer_segment=dp.objective_profile.customer_segment or "balanced",
                    market_region=dp.objective_profile.market_region or "US",
                    objective_vector=dp.objective_profile.objective_vector,
                    wage_human=dp.objective_profile.wage_human,
                    energy_price_kWh=dp.objective_profile.energy_price_kWh,
                    mean_delta_mpl=dp.attribution.delta_mpl,
                    mean_delta_error=dp.attribution.delta_error,
                    mean_delta_j=dp.attribution.delta_J,
                    mean_trust=dp.attribution.trust_score,
                    mean_w_econ=dp.attribution.w_econ,
                    profile_summaries={},  # Would need energy interventions
                )
                samples.append(context_to_sample(ctx))

    # Fill rest with synthetic
    remaining = num_samples - len(samples)
    for i in range(remaining):
        ctx = generate_synthetic_context(seed=i)
        samples.append(context_to_sample(ctx))

    # Save if path provided
    if save_path:
        save_dataset(samples, save_path)

    return samples


def sample_to_dict(sample: OrchestrationSample) -> Dict[str, Any]:
    """Convert OrchestrationSample to JSON-serializable dict."""
    result = {
        "context": asdict(sample.context),
        "context_features": sample.context_features.tolist(),
        "target_tool_sequence": [
            {"name": tc.name, "args": tc.args} for tc in sample.target_tool_sequence
        ],
        "heuristic_rationale": sample.heuristic_rationale,
        "metadata": sample.metadata,
        "source_type": sample.source_type,
    }
    if sample.econ_semantic_summary is not None:
        result["econ_semantic_summary"] = asdict(sample.econ_semantic_summary)
    return result


def save_dataset(samples: List[OrchestrationSample], path: str) -> None:
    """Save training dataset to JSON file."""
    data = [sample_to_dict(s) for s in samples]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(samples)} samples to {path}")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load training dataset from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {path}")
    return data


def dataset_to_tensors(
    samples: List[OrchestrationSample],
) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """
    Convert samples to numpy arrays for training.

    Returns:
        X: (N, ctx_dim) context features
        Y: (N, max_seq_len) tool indices (padded)
        tool_names: List of tool sequences (for reference)
    """
    X = np.stack([s.context_features for s in samples])

    # Convert tool names to indices
    tool_to_idx = {name: i for i, name in enumerate(TOOL_NAMES)}
    max_seq_len = max(len(s.target_tool_sequence) for s in samples)

    Y = np.zeros((len(samples), max_seq_len), dtype=np.int64)
    tool_names = []

    for i, sample in enumerate(samples):
        seq = []
        for j, tc in enumerate(sample.target_tool_sequence):
            if j < max_seq_len:
                Y[i, j] = tool_to_idx.get(tc.name, 0)
                seq.append(tc.name)
        tool_names.append(seq)

    return X, Y, tool_names


# ==============================================================================
# Econ/Semantic-Derived Training Samples
# ==============================================================================

def classify_pareto_frontier(
    econ_signals: Dict[str, Any],
    datapack_signals: Dict[str, Any],
) -> str:
    """
    Classify the Pareto frontier configuration based on signals.

    Returns one of: "energy_tight", "mpl_tight", "balanced", "safety_focused"
    """
    energy_urgency = econ_signals.get("energy_urgency", 0.0)
    mpl_urgency = econ_signals.get("mpl_urgency", 0.0)
    error_urgency = econ_signals.get("error_urgency", 0.0)

    if error_urgency > 0.7:
        return "safety_focused"
    elif energy_urgency > 0.5 and energy_urgency > mpl_urgency:
        return "energy_tight"
    elif mpl_urgency > 0.5 and mpl_urgency > energy_urgency:
        return "mpl_tight"
    else:
        return "balanced"


def derive_urgency_level(econ_signals: Dict[str, Any]) -> str:
    """Derive urgency level from econ signals."""
    max_urgency = max(
        econ_signals.get("mpl_urgency", 0.0),
        econ_signals.get("error_urgency", 0.0),
        econ_signals.get("energy_urgency", 0.0),
    )
    if max_urgency > 0.7:
        return "critical"
    elif max_urgency > 0.5:
        return "high"
    elif max_urgency > 0.3:
        return "moderate"
    else:
        return "none"


def derive_chosen_profile_from_signals(
    econ_signals: Dict[str, Any],
    datapack_signals: Dict[str, Any],
) -> str:
    """
    Derive the best energy profile based on econ/datapack signals.

    This is the "ground truth" from the economic feedback loop, not heuristics.
    """
    error_urgency = econ_signals.get("error_urgency", 0.0)
    energy_urgency = econ_signals.get("energy_urgency", 0.0)
    mpl_urgency = econ_signals.get("mpl_urgency", 0.0)
    wage_parity = econ_signals.get("wage_parity", 1.0)

    # Safety-first: if error rate is high, go SAFE
    if error_urgency > 0.6:
        return "SAFE"

    # Energy optimization: if energy costs are high
    if energy_urgency > 0.5:
        return "SAVER"

    # Throughput focus: if MPL is low and we need to catch up
    if mpl_urgency > 0.5 and wage_parity < 0.8:
        return "BOOST"

    # Default to BASE
    return "BASE"


def derive_objective_preset_from_signals(
    econ_signals: Dict[str, Any],
    datapack_signals: Dict[str, Any],
) -> str:
    """
    Derive objective preset from econ/datapack signals.

    This is ground truth from the economic layer.
    """
    error_urgency = econ_signals.get("error_urgency", 0.0)
    energy_urgency = econ_signals.get("energy_urgency", 0.0)
    mpl_urgency = econ_signals.get("mpl_urgency", 0.0)

    if error_urgency > 0.5:
        return "safety"
    elif energy_urgency > 0.5:
        return "energy_saver"
    elif mpl_urgency > 0.5:
        return "throughput"
    else:
        return "balanced"


def build_econ_semantic_sample(
    ctx: OrchestratorContext,
    econ_signals: Dict[str, Any],
    datapack_signals: Dict[str, Any],
    semantic_metrics: Optional[Dict[str, Any]] = None,
) -> OrchestrationSample:
    """
    Build training sample using econ/semantic signals as ground truth.

    Unlike heuristic samples, these use the actual EconomicController and
    DatapackEngine outputs to derive the target decisions.

    Args:
        ctx: OrchestratorContext with economic/datapack context
        econ_signals: EconSignals.to_dict() output
        datapack_signals: DatapackSignals.to_dict() output
        semantic_metrics: Optional SemanticMetrics data

    Returns:
        OrchestrationSample with econ/semantic-derived labels
    """
    # Encode context features
    context_features = _encode_ctx(ctx)

    # Derive decisions from econ/semantic signals (not heuristics!)
    chosen_profile = derive_chosen_profile_from_signals(econ_signals, datapack_signals)
    objective_preset = derive_objective_preset_from_signals(econ_signals, datapack_signals)
    pareto_class = classify_pareto_frontier(econ_signals, datapack_signals)
    urgency_level = derive_urgency_level(econ_signals)
    recommended_focus = datapack_signals.get("recommended_collection_focus", "balanced")

    # Compute semantic priority fraction
    semantic_priority_fraction = 0.0
    if semantic_metrics:
        # This would come from SemanticOrchestrator's task graph analysis
        semantic_priority_fraction = semantic_metrics.get("high_priority_task_fraction", 0.0)

    # Build econ/semantic summary (auxiliary supervision target)
    econ_semantic_summary = EconSemanticDecisionSummary(
        chosen_profile=chosen_profile,
        objective_preset=objective_preset,
        pareto_classification=pareto_class,
        urgency_level=urgency_level,
        recommended_focus=recommended_focus,
        semantic_priority_fraction=semantic_priority_fraction,
        data_coverage_score=datapack_signals.get("data_coverage_score", 0.0),
        wage_parity=econ_signals.get("wage_parity", 1.0),
    )

    # Build target tool sequence using econ/semantic decisions
    decisions = []

    # 1. Set backend (based on energy needs)
    backend = "isaac" if econ_signals.get("energy_urgency", 0) > 0.5 else "pybullet"
    decisions.append(HeuristicDecision(
        tool="SET_BACKEND",
        args={"backend": backend},
        rationale=f"Econ-driven: energy_urgency={econ_signals.get('energy_urgency', 0):.2f}",
    ))

    # 2. Set objective preset from econ signals
    decisions.append(HeuristicDecision(
        tool="SET_OBJECTIVE_PRESET",
        args={"preset": objective_preset},
        rationale=f"Econ-driven: pareto_class={pareto_class}, urgency={urgency_level}",
    ))

    # 3. Set energy profile from econ signals
    profile_weights = {p: 0.0 for p in ["BASE", "BOOST", "SAVER", "SAFE"]}
    profile_weights[chosen_profile] = 1.0
    decisions.append(HeuristicDecision(
        tool="SET_ENERGY_PROFILE",
        args={"profile_mix": profile_weights},
        rationale=f"Econ-driven: chosen_profile={chosen_profile} based on urgencies",
    ))

    # 4. Set data mix based on datapack signals
    tier2_frac = datapack_signals.get("tier2_fraction", 0.1)
    if tier2_frac < 0.1:  # Low frontier data
        data_mix = {"real": 0.4, "synthetic": 0.5, "hybrid": 0.1}
    elif datapack_signals.get("data_coverage_score", 0.5) < 0.5:
        data_mix = {"real": 0.5, "synthetic": 0.4, "hybrid": 0.1}
    else:
        data_mix = {"real": 0.6, "synthetic": 0.3, "hybrid": 0.1}

    decisions.append(HeuristicDecision(
        tool="SET_DATA_MIX",
        args={"data_mix": data_mix},
        rationale=f"Datapack-driven: tier2_frac={tier2_frac:.2f}, coverage={datapack_signals.get('data_coverage_score', 0):.2f}",
    ))

    # Convert to ToolCall format
    target_sequence = [ToolCall(name=d.tool, args=d.args) for d in decisions]
    rationales = [d.rationale for d in decisions]

    return OrchestrationSample(
        context=ctx,
        context_features=context_features,
        target_tool_sequence=target_sequence,
        heuristic_rationale=rationales,
        metadata={
            "num_tools": len(target_sequence),
            "tools_used": [d.tool for d in decisions],
            "pareto_classification": pareto_class,
            "urgency_level": urgency_level,
        },
        econ_semantic_summary=econ_semantic_summary,
        source_type="econ_semantic",
    )


def generate_synthetic_econ_semantic_context(seed: int = None) -> Tuple[OrchestratorContext, Dict[str, Any], Dict[str, Any]]:
    """
    Generate synthetic context with matching econ/datapack signals.

    Returns:
        Tuple of (context, econ_signals, datapack_signals)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate base context
    ctx = generate_synthetic_context(seed)

    # Generate corresponding econ signals
    econ_signals = {
        "current_mpl": random.uniform(30.0, 70.0),
        "baseline_mpl_human": 60.0,
        "mpl_delta": random.uniform(-5.0, 10.0),
        "error_rate": random.uniform(0.01, 0.15),
        "error_trend": random.uniform(-0.05, 0.05),
        "damage_cost_total": random.uniform(0.0, 100.0),
        "implied_wage": random.uniform(10.0, 25.0),
        "human_wage": 18.0,
        "wage_parity": random.uniform(0.6, 1.2),
        "energy_Wh_per_unit": random.uniform(0.05, 0.15),
        "energy_cost_per_unit": random.uniform(0.01, 0.05),
        "energy_urgency": random.uniform(0.0, 1.0),
        "mpl_urgency": random.uniform(0.0, 1.0),
        "error_urgency": random.uniform(0.0, 1.0),
        "customer_segment": ctx.customer_segment,
        "rebate_pct": random.uniform(0.0, 0.2),
        "attributable_spread_capture": random.uniform(0.0, 0.3),
        "data_premium": random.uniform(0.0, 0.1),
    }

    # Generate corresponding datapack signals
    datapack_signals = {
        "total_datapacks": random.randint(10, 500),
        "positive_fraction": random.uniform(0.5, 0.9),
        "negative_fraction": random.uniform(0.1, 0.5),
        "mean_novelty": random.uniform(0.2, 0.8),
        "max_novelty": random.uniform(0.5, 1.0),
        "novelty_variance": random.uniform(0.0, 0.3),
        "tier0_fraction": random.uniform(0.2, 0.5),
        "tier1_fraction": random.uniform(0.3, 0.5),
        "tier2_fraction": random.uniform(0.05, 0.3),
        "data_coverage_score": random.uniform(0.3, 0.9),
        "embedding_diversity": random.uniform(0.3, 0.8),
        "vla_annotation_fraction": random.uniform(0.3, 0.9),
        "guidance_annotation_fraction": random.uniform(0.4, 0.9),
        "semantic_tag_diversity": random.randint(5, 30),
        "mean_rebate_pct": random.uniform(0.0, 0.15),
        "mean_spread_capture": random.uniform(0.0, 0.2),
        "mean_data_premium": random.uniform(0.0, 0.08),
        "recommended_collection_focus": random.choice([
            "safety_edge_cases",
            "throughput_demonstrations",
            "energy_efficient_trajectories",
            "frontier_cases",
            "balanced",
        ]),
    }

    return ctx, econ_signals, datapack_signals


def build_mixed_training_dataset(
    num_heuristic: int = 500,
    num_econ_semantic: int = 500,
    save_path: Optional[str] = None,
) -> Tuple[List[OrchestrationSample], Dict[str, Any]]:
    """
    Build mixed dataset with both heuristic and econ/semantic samples.

    Args:
        num_heuristic: Number of heuristic-only samples
        num_econ_semantic: Number of econ/semantic-derived samples
        save_path: Optional path to save dataset

    Returns:
        Tuple of (samples, dataset_stats)
    """
    samples = []

    # Generate heuristic samples
    for i in range(num_heuristic):
        ctx = generate_synthetic_context(seed=i)
        sample = context_to_sample(ctx)
        sample.source_type = "heuristic"
        samples.append(sample)

    # Generate econ/semantic samples
    for i in range(num_econ_semantic):
        ctx, econ_sig, datapack_sig = generate_synthetic_econ_semantic_context(seed=i + num_heuristic)
        sample = build_econ_semantic_sample(ctx, econ_sig, datapack_sig)
        samples.append(sample)

    # Shuffle
    random.shuffle(samples)

    # Compute stats
    heuristic_count = sum(1 for s in samples if s.source_type == "heuristic")
    econ_semantic_count = sum(1 for s in samples if s.source_type == "econ_semantic")

    stats = {
        "total_samples": len(samples),
        "heuristic_count": heuristic_count,
        "econ_semantic_count": econ_semantic_count,
        "heuristic_fraction": heuristic_count / len(samples),
        "econ_semantic_fraction": econ_semantic_count / len(samples),
    }

    # Count econ/semantic summaries
    if econ_semantic_count > 0:
        profiles = {}
        presets = {}
        pareto_classes = {}
        urgencies = {}

        for s in samples:
            if s.econ_semantic_summary:
                p = s.econ_semantic_summary.chosen_profile
                profiles[p] = profiles.get(p, 0) + 1

                preset = s.econ_semantic_summary.objective_preset
                presets[preset] = presets.get(preset, 0) + 1

                pc = s.econ_semantic_summary.pareto_classification
                pareto_classes[pc] = pareto_classes.get(pc, 0) + 1

                urg = s.econ_semantic_summary.urgency_level
                urgencies[urg] = urgencies.get(urg, 0) + 1

        stats["profile_distribution"] = profiles
        stats["preset_distribution"] = presets
        stats["pareto_classification_distribution"] = pareto_classes
        stats["urgency_distribution"] = urgencies

    # Save if path provided
    if save_path:
        save_dataset(samples, save_path)
        # Also save stats
        stats_path = save_path.replace(".json", "_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved dataset stats to {stats_path}")

    return samples, stats


def split_dataset_by_source(
    samples: List[OrchestrationSample],
) -> Tuple[List[OrchestrationSample], List[OrchestrationSample]]:
    """
    Split dataset into heuristic-only and econ/semantic-derived subsets.

    Returns:
        Tuple of (heuristic_samples, econ_semantic_samples)
    """
    heuristic = [s for s in samples if s.source_type == "heuristic"]
    econ_semantic = [s for s in samples if s.source_type == "econ_semantic"]
    return heuristic, econ_semantic
