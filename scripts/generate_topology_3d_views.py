#!/usr/bin/env python3
"""Generate interactive 3D topology charts for robotics-vp-core architecture.

Outputs:
- artifacts/topology_3d/base_typed_graph_3d.json
- artifacts/topology_3d/chart_a_dependency_dag_3d.html
- artifacts/topology_3d/chart_b_runtime_feedback_3d.html
- artifacts/topology_3d/chart_c_ontology_overlay_3d.html
- artifacts/topology_3d/index.html

The generated HTML files are standalone viewers (Plotly via CDN) with:
- 3D nodes and directed edges
- edge/node semantic metadata in hover text
- legends for node/edge types
- cycle summaries
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "topology_3d"


NODE_TYPE_COLORS = {
    "compute": "#1f77b4",
    "interface": "#2ca02c",
    "data": "#ff7f0e",
    "governance": "#d62728",
}

EDGE_TYPE_COLORS = {
    "dataflow": "#1f77b4",
    "control": "#9467bd",
    "objective": "#2ca02c",
    "telemetry": "#17becf",
    "constraint": "#d62728",
}


BASE_NODES = {
    "N1": {
        "name": "WorkcellPhysicsEnv",
        "node_type": "compute",
        "purpose": "World dynamics and actuation stepping.",
        "representation_space": "S_t continuous scene state (poses, time, contact stats)",
        "modules": [
            "src/envs/workcell_env/env.py",
            "src/envs/workcell_env/physics/physics_adapter.py",
            "src/envs/workcell_env/physics/simple_physics.py",
        ],
    },
    "N2": {
        "name": "WorkcellObservationBuilder",
        "node_type": "interface",
        "purpose": "Build raw observations from scene state.",
        "representation_space": "O_raw object lists + state_vector",
        "modules": [
            "src/envs/workcell_env/observations/obs_builder.py",
        ],
    },
    "N3": {
        "name": "ObservationAdapter",
        "node_type": "interface",
        "purpose": "Canonical observation and condition-vector composition.",
        "representation_space": "O typed slices + optional ConditionVector",
        "modules": [
            "src/observation/adapter.py",
        ],
    },
    "N4": {
        "name": "EncoderStack",
        "node_type": "compute",
        "purpose": "Observation encoding into latent embeddings.",
        "representation_space": "Z latent tensors/features",
        "modules": [
            "src/encoders/builder.py",
            "src/encoders/video_encoder.py",
            "src/encoders/mlp_encoder.py",
        ],
    },
    "N5": {
        "name": "PolicyStack",
        "node_type": "compute",
        "purpose": "Policy inference and action generation.",
        "representation_space": "A action distribution/manifold",
        "modules": [
            "src/rl/sac.py",
            "src/inference/demo_policy.py",
        ],
    },
    "N6": {
        "name": "RewardEngineLogger",
        "node_type": "compute",
        "purpose": "Reward decomposition and telemetry stream construction.",
        "representation_space": "R scalar reward + component stream",
        "modules": [
            "src/rl/reward_shaping.py",
            "src/economics/reward_engine.py",
            "src/logging/episode_logger.py",
        ],
    },
    "N7": {
        "name": "ObjectiveEconBridge",
        "node_type": "compute",
        "purpose": "Objective tensor scalarization and objective->econ mapping.",
        "representation_space": "T_obj ObjectiveTensor and T_econ EconTensor",
        "modules": [
            "src/objectives/tensor.py",
            "src/objectives/compiler.py",
            "src/economics/functor.py",
            "src/economics/econ_tensor.py",
        ],
    },
    "N8": {
        "name": "OntologyStore",
        "node_type": "data",
        "purpose": "Persistent JSONL ontology/event/objective/econ store.",
        "representation_space": "DB_ont persistent JSONL records",
        "modules": [
            "src/ontology/store.py",
        ],
    },
    "N9": {
        "name": "DataPackRepo",
        "node_type": "data",
        "purpose": "Datapack corpus storage/query and schema enforcement.",
        "representation_space": "D_pack DataPackMeta corpus",
        "modules": [
            "src/valuation/datapack_repo.py",
            "src/valuation/datapack_schema.py",
        ],
    },
    "N10": {
        "name": "EconDataControllers",
        "node_type": "governance",
        "purpose": "Economic/data signals and semantic advisory generation.",
        "representation_space": "Sigma_econ + Sigma_data + advisories",
        "modules": [
            "src/orchestrator/economic_controller.py",
            "src/orchestrator/datapack_engine.py",
            "src/orchestrator/semantic_orchestrator_v2.py",
        ],
    },
    "N11": {
        "name": "RLSamplerTrainer",
        "node_type": "compute",
        "purpose": "Descriptor sampling and policy training update execution.",
        "representation_space": "D_train descriptors + replay + checkpoints",
        "modules": [
            "src/rl/episode_sampling.py",
            "src/rl/sac.py",
        ],
    },
    "N12": {
        "name": "RegalDeployLedger",
        "node_type": "governance",
        "purpose": "Regality verification, deploy gate, and immutable ledger writes.",
        "representation_space": "G deploy decisions + ledger records",
        "modules": [
            "src/training/regal_training_runner.py",
            "src/deployment/deploy_gate.py",
            "src/valuation/value_ledger.py",
        ],
    },
    "N13": {
        "name": "ConstraintSet",
        "node_type": "governance",
        "purpose": "Constraint synthesis and action-manifold projection metadata.",
        "representation_space": "C hard/soft bounds and invariants",
        "modules": [
            "src/constraints/constraint_set.py",
        ],
    },
}


CHART_A_COORDS: Dict[str, Tuple[float, float, float]] = {
    "N2": (-8.0, 2.2, 0.0),
    "N3": (-6.0, 2.0, 0.8),
    "N4": (-4.5, 2.2, 1.8),
    "N7": (-4.2, 0.2, 2.2),
    "N8": (-2.4, 0.2, 1.2),
    "N9": (-2.2, -2.2, 1.1),
    "N13": (-1.0, 2.4, 1.4),
    "N1": (0.0, 3.0, 0.2),
    "N5": (2.0, 2.2, 2.0),
    "N6": (2.0, 0.2, 1.1),
    "N10": (4.1, 0.1, 1.4),
    "N11": (6.2, 1.0, 2.0),
    "N12": (8.3, 1.1, 3.0),
}


CHART_B_COORDS: Dict[str, Tuple[float, float, float]] = {
    "N1": (0.0, 0.0, 0.1),
    "N2": (2.0, 1.2, 0.2),
    "N3": (4.2, 0.4, 0.3),
    "N4": (6.2, -0.8, 0.4),
    "N5": (8.0, 0.2, 0.3),
    "N6": (2.0, -2.0, 1.2),
    "N7": (4.1, -3.0, 1.6),
    "N8": (3.5, -5.2, 2.2),
    "N9": (0.7, -5.0, 2.3),
    "N10": (6.3, -5.0, 2.5),
    "N11": (8.3, -3.0, 1.9),
    "N12": (10.4, -2.0, 3.2),
    "N13": (7.0, -6.4, 2.9),
}


CHART_C_COORDS: Dict[str, Tuple[float, float, float]] = {
    "N1": (0.0, 0.0, 0.5),
    "N2": (2.0, 1.0, 1.0),
    "N3": (4.0, 1.2, 1.6),
    "N4": (6.3, 1.1, 3.0),
    "N5": (8.5, 0.2, 2.6),
    "N6": (2.2, -1.4, 2.1),
    "N7": (4.4, -1.5, 3.6),
    "N8": (4.4, -3.5, 1.2),
    "N9": (6.3, -3.2, 1.8),
    "N10": (8.4, -3.0, 2.8),
    "N11": (10.2, -2.0, 2.7),
    "N12": (12.2, -1.0, 3.3),
    "N13": (9.0, -5.2, 3.1),
}


CHART_A_EDGES = [
    {
        "id": "A1",
        "source": "N2",
        "target": "N1",
        "edge_type": "dataflow",
        "domain": "scene_state",
        "codomain": "raw_obs",
        "invariant": "deterministic object ordering and state_vector emission",
        "annotation": "interface=WorkcellObservationBuilder.build; map=scene_state->state_vector",
    },
    {
        "id": "A2",
        "source": "N3",
        "target": "N5",
        "edge_type": "dataflow",
        "domain": "Observation slices",
        "codomain": "policy tensor",
        "invariant": "canonical flattening order remains stable",
        "annotation": "interface=Observation contract; map=slices->policy features",
    },
    {
        "id": "A3",
        "source": "N4",
        "target": "N5",
        "edge_type": "dataflow",
        "domain": "obs tensor",
        "codomain": "latent z",
        "invariant": "latent_dim fixed by encoder config",
        "annotation": "interface=nn.Module encoder; map=image/state->latent",
    },
    {
        "id": "A4",
        "source": "N7",
        "target": "N6",
        "edge_type": "objective",
        "domain": "reward components",
        "codomain": "objective/econ tensor artifacts",
        "invariant": "objective axis count and schema checks pass",
        "annotation": "interface=ObjectiveTensor+Compiler; map=reward->objective/econ tensors",
    },
    {
        "id": "A5",
        "source": "N8",
        "target": "N6",
        "edge_type": "telemetry",
        "domain": "episode/events",
        "codomain": "persisted ontology rows",
        "invariant": "append order for events, keyed upserts for episodes",
        "annotation": "interface=OntologyStore JSONL API; map=events/econ/objective->rows",
    },
    {
        "id": "A6",
        "source": "N9",
        "target": "N10",
        "edge_type": "telemetry",
        "domain": "datapack corpus",
        "codomain": "DatapackSignals",
        "invariant": "fraction and tier signals clamped to [0,1]",
        "annotation": "interface=DataPackMeta schema; map=datapacks->signals",
    },
    {
        "id": "A7",
        "source": "N8",
        "target": "N10",
        "edge_type": "telemetry",
        "domain": "econ/event history",
        "codomain": "EconSignals",
        "invariant": "urgency metrics computed from persisted data only",
        "annotation": "interface=OntologyStore queries; map=history->econ urgencies",
    },
    {
        "id": "A8",
        "source": "N13",
        "target": "N10",
        "edge_type": "constraint",
        "domain": "semantic evidence",
        "codomain": "bounded constraint tags",
        "invariant": "hard bounds and confidence thresholds are clamped",
        "annotation": "interface=ConstraintSet v1; map=evidence->constraint fields",
    },
    {
        "id": "A9",
        "source": "N1",
        "target": "N11",
        "edge_type": "dataflow",
        "domain": "env transition tuple",
        "codomain": "training transition stream",
        "invariant": "transition shape matches replay expectations",
        "annotation": "interface=env reset/step protocol; map=action->transition",
    },
    {
        "id": "A10",
        "source": "N5",
        "target": "N11",
        "edge_type": "control",
        "domain": "policy outputs",
        "codomain": "optimizer inputs",
        "invariant": "action bounds preserved under tanh policy",
        "annotation": "interface=SAC actor/critic APIs; map=latent->action/Q",
    },
    {
        "id": "A11",
        "source": "N6",
        "target": "N11",
        "edge_type": "objective",
        "domain": "reward components",
        "codomain": "training targets",
        "invariant": "reward mode contract remains mpl_ep_error",
        "annotation": "interface=reward component dict; map=metrics->targets",
    },
    {
        "id": "A12",
        "source": "N10",
        "target": "N11",
        "edge_type": "objective",
        "domain": "advisory signal bundle",
        "codomain": "sampler/trainer overrides",
        "invariant": "advisory-only, no direct reward math mutation",
        "annotation": "interface=Econ/Data advisory contracts; map=urgency->sampling",
    },
    {
        "id": "A13",
        "source": "N11",
        "target": "N12",
        "edge_type": "telemetry",
        "domain": "audit/regal artifacts",
        "codomain": "deploy gate inputs and ledger",
        "invariant": "typed deploy inputs and deterministic decision sha",
        "annotation": "interface=DeployGateInputsV1; map=training artifacts->gate decision",
    },
]


RUNTIME_EDGES = [
    {
        "id": "E1",
        "source": "N1",
        "target": "N2",
        "edge_type": "dataflow",
        "domain": "S_t",
        "codomain": "O_raw",
        "invariant": "observation extraction deterministic",
        "annotation": "sync <=1 tick",
    },
    {
        "id": "E2",
        "source": "N2",
        "target": "N3",
        "edge_type": "dataflow",
        "domain": "O_raw",
        "codomain": "O",
        "invariant": "canonical field ordering",
        "annotation": "sync <=1 tick",
    },
    {
        "id": "E3",
        "source": "N3",
        "target": "N4",
        "edge_type": "dataflow",
        "domain": "O",
        "codomain": "Z",
        "invariant": "encoder latent schema stable",
        "annotation": "sync <=1 tick",
    },
    {
        "id": "E4",
        "source": "N4",
        "target": "N5",
        "edge_type": "dataflow",
        "domain": "Z",
        "codomain": "A",
        "invariant": "policy input dimensionality fixed",
        "annotation": "sync <=1 tick",
    },
    {
        "id": "E5",
        "source": "N5",
        "target": "N1",
        "edge_type": "control",
        "domain": "A x S_t",
        "codomain": "S_{t+1}",
        "invariant": "state transition bounded by spatial/time constraints",
        "annotation": "sync; latency=WorkcellEnvConfig.time_step_s",
    },
    {
        "id": "E6",
        "source": "N1",
        "target": "N6",
        "edge_type": "telemetry",
        "domain": "S_{t+1}, info",
        "codomain": "R",
        "invariant": "reward component schema preserved",
        "annotation": "sync per step",
    },
    {
        "id": "E7",
        "source": "N6",
        "target": "N7",
        "edge_type": "objective",
        "domain": "R",
        "codomain": "T_obj, T_econ",
        "invariant": "objective axis/schema checks",
        "annotation": "sync per step + async at episode end",
    },
    {
        "id": "E8",
        "source": "N7",
        "target": "N8",
        "edge_type": "telemetry",
        "domain": "tensor/events",
        "codomain": "DB_ont",
        "invariant": "event append order and episode keyed upserts",
        "annotation": "async episode end",
    },
    {
        "id": "E9",
        "source": "N8",
        "target": "N9",
        "edge_type": "dataflow",
        "domain": "ontology rows",
        "codomain": "datapack corpus",
        "invariant": "schema-versioned datapack artifacts",
        "annotation": "async export window",
    },
    {
        "id": "E10",
        "source": "N9",
        "target": "N10",
        "edge_type": "telemetry",
        "domain": "D_pack",
        "codomain": "Sigma_data",
        "invariant": "coverage/tier/novelty metrics bounded",
        "annotation": "async controller window",
    },
    {
        "id": "E11",
        "source": "N8",
        "target": "N10",
        "edge_type": "telemetry",
        "domain": "DB_ont",
        "codomain": "Sigma_econ",
        "invariant": "urgency metrics derived from persisted events",
        "annotation": "async controller window",
    },
    {
        "id": "E12",
        "source": "N10",
        "target": "N11",
        "edge_type": "objective",
        "domain": "Sigma_econ, Sigma_data",
        "codomain": "D_train",
        "invariant": "advisory-only signal path",
        "annotation": "async curriculum interval",
    },
    {
        "id": "E13",
        "source": "N9",
        "target": "N11",
        "edge_type": "dataflow",
        "domain": "D_pack",
        "codomain": "D_train",
        "invariant": "descriptor normalization and validation",
        "annotation": "async sampling interval",
    },
    {
        "id": "E14",
        "source": "N11",
        "target": "N5",
        "edge_type": "control",
        "domain": "optimizer state",
        "codomain": "updated policy",
        "invariant": "policy parameter update integrity",
        "annotation": "async optimizer step",
    },
    {
        "id": "E15",
        "source": "N11",
        "target": "N12",
        "edge_type": "telemetry",
        "domain": "training artifacts",
        "codomain": "deploy inputs + ledger",
        "invariant": "manifest/audit/regal artifacts emitted",
        "annotation": "async training window",
    },
    {
        "id": "E16",
        "source": "N12",
        "target": "N5",
        "edge_type": "control",
        "domain": "deploy decision",
        "codomain": "active policy selection",
        "invariant": "deterministic gate decision for fixed inputs",
        "annotation": "async deploy event",
    },
    {
        "id": "E17",
        "source": "N10",
        "target": "N13",
        "edge_type": "constraint",
        "domain": "semantic/econ signals",
        "codomain": "constraint set",
        "invariant": "hard/soft bounds remain safe and clamped",
        "annotation": "async constraint synthesis",
    },
    {
        "id": "E18",
        "source": "N13",
        "target": "N5",
        "edge_type": "constraint",
        "domain": "constraint set",
        "codomain": "policy action manifold",
        "invariant": "constraint projection applied pre-actuation",
        "annotation": "sync at inference",
    },
]


CHART_C_RUNTIME_EDGE_MAPPINGS = {
    "E1": "equivariant_projection (state->ordered obs)",
    "E2": "canonicalization (raw obs->typed slices)",
    "E3": "tokenization_embedding (obs->latent)",
    "E4": "policy_pushforward (latent->action)",
    "E5": "closed_loop_transition (action->next state)",
    "E6": "measurement_operator (state/info->reward terms)",
    "E7": "scalarization_lift (reward terms->objective/econ tensors)",
    "E8": "calibration_preserving_serialization",
    "E9": "ontology_to_datapack_adapter",
    "E10": "novelty_coverage_aggregation",
    "E11": "econ_history_aggregation",
    "E12": "advisory_weight_functor",
    "E13": "descriptor_ingestion_map",
    "E14": "optimizer_update_operator",
    "E15": "regality_audit_map",
    "E16": "deploy_gate_morphism",
    "E17": "constraint_synthesis_from_signals",
    "E18": "projection_to_action_manifold",
}


RUNTIME_CYCLES = [
    {
        "id": "C1",
        "label": "Control loop",
        "path": ["N1", "N2", "N3", "N4", "N5", "N1"],
        "description": "world state -> observation -> policy -> actuation -> new world state",
    },
    {
        "id": "C2",
        "label": "Training/telemetry loop",
        "path": ["N1", "N6", "N7", "N8", "N10", "N11", "N5", "N1"],
        "description": "telemetry -> objective update -> policy update -> deployment to runtime",
    },
    {
        "id": "C3",
        "label": "Data flywheel",
        "path": ["N9", "N10", "N11", "N12", "N5", "N1", "N6", "N7", "N8", "N9"],
        "description": "datapacks -> advisories -> train -> govern -> runtime -> new datapacks",
    },
]


CHART_C_NODE_FIBERS = {
    "N1": {
        "entropy_metric": "collision_count, constraint_error",
        "objective_tensor_components": "throughput,error,safety,energy pass-through",
        "safety_metadata": "max_steps, spatial_bounds, physics_mode",
    },
    "N2": {
        "entropy_metric": "missing object/type rate",
        "objective_tensor_components": "pass-through",
        "safety_metadata": "sort_ids deterministic ordering",
    },
    "N3": {
        "entropy_metric": "trust and recap confidence slices",
        "objective_tensor_components": "econ/objective context carry-through",
        "safety_metadata": "typed slice schema + gated fields",
    },
    "N4": {
        "entropy_metric": "latent std and temporal smoothness",
        "objective_tensor_components": "pass-through",
        "safety_metadata": "fixed latent_dim and checkpoint constraints",
    },
    "N5": {
        "entropy_metric": "policy entropy alpha",
        "objective_tensor_components": "scalarization target profile",
        "safety_metadata": "tanh action clamp + constraint projection",
    },
    "N6": {
        "entropy_metric": "reward component variance",
        "objective_tensor_components": "mpl/error/energy/wage terms",
        "safety_metadata": "mpl_ep_error mode contract",
    },
    "N7": {
        "entropy_metric": "uncertainty_discount and violation_count",
        "objective_tensor_components": "ObjectiveCompiler axis weights/maximize flags",
        "safety_metadata": "objective axis/shape validation",
    },
    "N8": {
        "entropy_metric": "coverage and missing-field entropy",
        "objective_tensor_components": "persisted objective/econ history",
        "safety_metadata": "append-only events + keyed upsert",
    },
    "N9": {
        "entropy_metric": "trust_score and novelty variance",
        "objective_tensor_components": "objective_profile/objective_vector",
        "safety_metadata": "schema_version and validator checks",
    },
    "N10": {
        "entropy_metric": "urgency/confidence estimates",
        "objective_tensor_components": "objective adjustment vectors",
        "safety_metadata": "advisory-only boundary",
    },
    "N11": {
        "entropy_metric": "td-error and audit confidence spread",
        "objective_tensor_components": "sampler weights + optimizer objective",
        "safety_metadata": "descriptor normalization/validation",
    },
    "N12": {
        "entropy_metric": "regal_degraded flag",
        "objective_tensor_components": "deploy threshold vectors",
        "safety_metadata": "deterministic decision SHA and gate checks",
    },
    "N13": {
        "entropy_metric": "map quality and disagreement margins",
        "objective_tensor_components": "constraint_penalty coupling",
        "safety_metadata": "hard bounds and safety invariants",
    },
}


def _assemble_nodes(coords: Dict[str, Tuple[float, float, float]], with_fibers: bool = False) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for node_id, meta in BASE_NODES.items():
        x, y, z = coords[node_id]
        node_record = {
            "id": node_id,
            "x": x,
            "y": y,
            "z": z,
            **meta,
        }
        if with_fibers:
            node_record["fiber"] = CHART_C_NODE_FIBERS[node_id]
        out.append(node_record)
    return out


def _chart_a_view() -> Dict[str, object]:
    return {
        "view_id": "chart_a_dependency_dag_3d",
        "title": "Chart A: Dependency DAG (3D)",
        "description": "Acyclic compile-time/repo dependency view with explicit interface mappings.",
        "strict_partial_order": [
            "N2",
            "N3",
            "N4",
            "N7",
            "N8",
            "N9",
            "N13",
            "N1",
            "N5",
            "N6",
            "N10",
            "N11",
            "N12",
        ],
        "nodes": _assemble_nodes(CHART_A_COORDS),
        "edges": CHART_A_EDGES,
        "cycles": [],
    }


def _chart_b_view() -> Dict[str, object]:
    return {
        "view_id": "chart_b_runtime_feedback_3d",
        "title": "Chart B: Runtime Control + Feedback Loops (3D)",
        "description": "Runtime control loop and telemetry/training feedback loop with latency annotations.",
        "nodes": _assemble_nodes(CHART_B_COORDS),
        "edges": RUNTIME_EDGES,
        "cycles": RUNTIME_CYCLES,
    }


def _chart_c_view() -> Dict[str, object]:
    mapped_edges = []
    for edge in RUNTIME_EDGES:
        mapped = dict(edge)
        mapping_label = CHART_C_RUNTIME_EDGE_MAPPINGS.get(edge["id"], "semantic mapping")
        mapped["annotation"] = f"semantic_mapping={mapping_label}"
        mapped_edges.append(mapped)

    return {
        "view_id": "chart_c_ontology_overlay_3d",
        "title": "Chart C: Ontology Overlay (3D)",
        "description": "Runtime graph with representation fibers, entropy metrics, objective tensor components, and safety metadata.",
        "nodes": _assemble_nodes(CHART_C_COORDS, with_fibers=True),
        "edges": mapped_edges,
        "cycles": RUNTIME_CYCLES,
    }


def _legend_html() -> str:
    node_rows = "".join(
        f"<tr><td>{k}</td><td><span style='color:{v};font-weight:700'>\u25a0</span></td></tr>"
        for k, v in NODE_TYPE_COLORS.items()
    )
    edge_rows = "".join(
        f"<tr><td>{k}</td><td><span style='color:{v};font-weight:700'>\u25a0</span></td></tr>"
        for k, v in EDGE_TYPE_COLORS.items()
    )
    return (
        "<h3>Legend</h3>"
        "<p><strong>Node type colors</strong></p>"
        f"<table>{node_rows}</table>"
        "<p><strong>Edge type colors</strong></p>"
        f"<table>{edge_rows}</table>"
        "<p><strong>Hover details</strong> include domain->codomain, invariants, and per-view annotations.</p>"
    )


def _panel_html(view: Dict[str, object]) -> str:
    nodes = view["nodes"]
    edges = view["edges"]
    cycles = view["cycles"]

    node_rows = []
    for n in nodes:
        modules = ", ".join(n["modules"])
        fiber = n.get("fiber")
        fiber_blob = ""
        if fiber:
            fiber_blob = (
                "<br/><small>"
                f"entropy={fiber['entropy_metric']}"
                f"; omega={fiber['objective_tensor_components']}"
                f"; safety={fiber['safety_metadata']}"
                "</small>"
            )
        node_rows.append(
            "<tr>"
            f"<td>{n['id']}</td>"
            f"<td>{n['name']}</td>"
            f"<td>{n['node_type']}</td>"
            f"<td>{n['representation_space']}{fiber_blob}</td>"
            f"<td><small>{modules}</small></td>"
            "</tr>"
        )

    edge_rows = []
    for e in edges:
        edge_rows.append(
            "<tr>"
            f"<td>{e['id']}</td>"
            f"<td>{e['edge_type']}</td>"
            f"<td>{e['source']} -&gt; {e['target']}</td>"
            f"<td>{e['domain']} -&gt; {e['codomain']}</td>"
            f"<td>{e['invariant']}</td>"
            f"<td>{e['annotation']}</td>"
            "</tr>"
        )

    cycle_rows = []
    for c in cycles:
        cycle_rows.append(
            "<tr>"
            f"<td>{c['id']}</td>"
            f"<td>{c['label']}</td>"
            f"<td>{' -> '.join(c['path'])}</td>"
            f"<td>{c['description']}</td>"
            "</tr>"
        )

    cycle_section = ""
    if cycle_rows:
        cycle_section = (
            "<h3>Cycles</h3>"
            "<table>"
            "<thead><tr><th>ID</th><th>Label</th><th>Path</th><th>Description</th></tr></thead>"
            f"<tbody>{''.join(cycle_rows)}</tbody>"
            "</table>"
        )

    return (
        f"<h2>{view['title']}</h2>"
        f"<p>{view['description']}</p>"
        f"{_legend_html()}"
        "<h3>Nodes</h3>"
        "<table>"
        "<thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Representation</th><th>Modules</th></tr></thead>"
        f"<tbody>{''.join(node_rows)}</tbody>"
        "</table>"
        "<h3>Edges</h3>"
        "<table>"
        "<thead><tr><th>ID</th><th>Type</th><th>Endpoints</th><th>Domain->Codomain</th><th>Invariant</th><th>Annotation</th></tr></thead>"
        f"<tbody>{''.join(edge_rows)}</tbody>"
        "</table>"
        f"{cycle_section}"
    )


def _html_template(view: Dict[str, object]) -> str:
    view_json = json.dumps(view)
    panel_html = _panel_html(view)
    node_colors = json.dumps(NODE_TYPE_COLORS)
    edge_colors = json.dumps(EDGE_TYPE_COLORS)

    template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>__TITLE__</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: #0b1220;
      color: #e7ecf3;
    }
    .layout {
      display: grid;
      grid-template-columns: 62vw 38vw;
      min-height: 100vh;
    }
    #plot {
      width: 100%;
      height: 100vh;
      border-right: 1px solid #202a3a;
    }
    .panel {
      padding: 12px 14px 80px 14px;
      overflow: auto;
      max-height: 100vh;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 14px;
      font-size: 12px;
    }
    th, td {
      border: 1px solid #2a364a;
      padding: 6px;
      vertical-align: top;
      text-align: left;
    }
    thead th {
      background: #141d2e;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    a { color: #7fc8ff; }
    .footer {
      position: fixed;
      bottom: 0;
      right: 0;
      left: 62vw;
      background: rgba(8, 12, 21, 0.95);
      border-top: 1px solid #202a3a;
      padding: 8px 12px;
      font-size: 12px;
    }
    @media (max-width: 1200px) {
      .layout { grid-template-columns: 1fr; }
      #plot { height: 62vh; border-right: none; border-bottom: 1px solid #202a3a; }
      .footer { left: 0; }
    }
  </style>
</head>
<body>
  <div class=\"layout\">
    <div id=\"plot\"></div>
    <div class=\"panel\">__PANEL_HTML__</div>
  </div>
  <div class=\"footer\">Use mouse to orbit/zoom. Hover nodes/edges for typed semantics. Directed edges are shown with cones near targets.</div>

  <script>
    const view = __VIEW_JSON__;
    const nodeTypeColors = __NODE_COLORS__;
    const edgeTypeColors = __EDGE_COLORS__;

    const nodesById = Object.fromEntries(view.nodes.map(n => [n.id, n]));
    const traces = [];

    const nodeX = [];
    const nodeY = [];
    const nodeZ = [];
    const nodeText = [];
    const nodeColor = [];
    const nodeSize = [];

    for (const n of view.nodes) {
      const modules = (n.modules || []).join('<br/>');
      const fiber = n.fiber
        ? `<br/><b>fiber entropy</b>: ${n.fiber.entropy_metric}<br/><b>fiber omega</b>: ${n.fiber.objective_tensor_components}<br/><b>fiber safety</b>: ${n.fiber.safety_metadata}`
        : '';
      nodeX.push(n.x);
      nodeY.push(n.y);
      nodeZ.push(n.z);
      nodeText.push(
        `<b>${n.id} ${n.name}</b><br/>` +
        `<b>type</b>: ${n.node_type}<br/>` +
        `<b>purpose</b>: ${n.purpose}<br/>` +
        `<b>repr</b>: ${n.representation_space}<br/>` +
        `<b>modules</b>:<br/>${modules}` + fiber
      );
      nodeColor.push(nodeTypeColors[n.node_type] || '#9aa4b2');
      nodeSize.push(n.node_type === 'governance' ? 10 : 8);
    }

    traces.push({
      type: 'scatter3d',
      mode: 'markers+text',
      x: nodeX,
      y: nodeY,
      z: nodeZ,
      text: view.nodes.map(n => n.id),
      textposition: 'top center',
      textfont: { size: 11, color: '#d9e2ef' },
      hovertext: nodeText,
      hoverinfo: 'text',
      marker: { size: nodeSize, color: nodeColor, opacity: 0.95 },
      name: 'nodes',
      showlegend: false,
    });

    const legendNodeTypes = Array.from(new Set(view.nodes.map(n => n.node_type)));
    for (const nt of legendNodeTypes) {
      traces.push({
        type: 'scatter3d',
        mode: 'markers',
        x: [null],
        y: [null],
        z: [null],
        marker: { size: 8, color: nodeTypeColors[nt] || '#9aa4b2' },
        name: `node:${nt}`,
        hoverinfo: 'skip',
        showlegend: true,
      });
    }

    const legendEdgeTypes = Array.from(new Set(view.edges.map(e => e.edge_type)));
    for (const et of legendEdgeTypes) {
      traces.push({
        type: 'scatter3d',
        mode: 'lines',
        x: [null, null],
        y: [null, null],
        z: [null, null],
        line: { width: 5, color: edgeTypeColors[et] || '#9aa4b2' },
        name: `edge:${et}`,
        hoverinfo: 'skip',
        showlegend: true,
      });
    }

    for (const e of view.edges) {
      const s = nodesById[e.source];
      const t = nodesById[e.target];
      if (!s || !t) continue;

      const color = edgeTypeColors[e.edge_type] || '#9aa4b2';

      traces.push({
        type: 'scatter3d',
        mode: 'lines',
        x: [s.x, t.x],
        y: [s.y, t.y],
        z: [s.z, t.z],
        line: { width: 4, color },
        hovertext: (
          `<b>${e.id}</b> ${e.edge_type}<br/>` +
          `<b>endpoints</b>: ${e.source} -> ${e.target}<br/>` +
          `<b>domain->codomain</b>: ${e.domain} -> ${e.codomain}<br/>` +
          `<b>invariant</b>: ${e.invariant}<br/>` +
          `<b>annotation</b>: ${e.annotation}`
        ),
        hoverinfo: 'text',
        showlegend: false,
      });

      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const dz = t.z - s.z;
      const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (norm > 1e-9) {
        const ux = dx / norm;
        const uy = dy / norm;
        const uz = dz / norm;
        const back = 0.35;
        const px = t.x - ux * back;
        const py = t.y - uy * back;
        const pz = t.z - uz * back;
        traces.push({
          type: 'cone',
          x: [px],
          y: [py],
          z: [pz],
          u: [ux * 0.26],
          v: [uy * 0.26],
          w: [uz * 0.26],
          anchor: 'tip',
          sizemode: 'absolute',
          sizeref: 0.16,
          showscale: false,
          hoverinfo: 'skip',
          colorscale: [[0, color], [1, color]],
          showlegend: false,
          opacity: 0.95,
        });
      }
    }

    const layout = {
      title: { text: view.title, font: { color: '#e7ecf3', size: 18 } },
      paper_bgcolor: '#0b1220',
      plot_bgcolor: '#0b1220',
      margin: { l: 0, r: 0, t: 48, b: 0 },
      legend: {
        bgcolor: 'rgba(10,15,28,0.85)',
        bordercolor: '#24324a',
        borderwidth: 1,
        font: { color: '#d6deea', size: 11 },
      },
      scene: {
        bgcolor: '#0b1220',
        xaxis: { title: 'X', color: '#d6deea', gridcolor: '#1d2a3e', zerolinecolor: '#2d3c55' },
        yaxis: { title: 'Y', color: '#d6deea', gridcolor: '#1d2a3e', zerolinecolor: '#2d3c55' },
        zaxis: { title: 'Z', color: '#d6deea', gridcolor: '#1d2a3e', zerolinecolor: '#2d3c55' },
        camera: { eye: { x: 1.5, y: 1.2, z: 1.1 } },
      },
    };

    Plotly.newPlot('plot', traces, layout, { responsive: true, displaylogo: false });
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", str(view["title"]))
        .replace("__PANEL_HTML__", panel_html)
        .replace("__VIEW_JSON__", view_json)
        .replace("__NODE_COLORS__", node_colors)
        .replace("__EDGE_COLORS__", edge_colors)
    )


def _write_index(views: List[Dict[str, object]], out_dir: Path) -> None:
    links = []
    for view in views:
        file_name = f"{view['view_id']}.html"
        links.append(
            f"<li><a href='{file_name}'>{view['title']}</a> - {view['description']}</li>"
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Topology 3D Views</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: #0b1220;
      color: #e7ecf3;
      padding: 28px;
    }}
    a {{ color: #7fc8ff; }}
    code {{ background: #141d2e; padding: 2px 4px; border-radius: 4px; }}
    .card {{
      background: #121b2c;
      border: 1px solid #24324a;
      border-radius: 8px;
      padding: 16px;
      max-width: 980px;
    }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>3D Topology Views</h1>
    <p>Generated artifacts for typed architecture graph.</p>
    <ul>
      {''.join(links)}
    </ul>
    <p>Machine-readable base spec: <code>base_typed_graph_3d.json</code></p>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chart_a = _chart_a_view()
    chart_b = _chart_b_view()
    chart_c = _chart_c_view()

    payload = {
        "schema_version": "topology_3d_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "node_type_legend": NODE_TYPE_COLORS,
        "edge_type_legend": EDGE_TYPE_COLORS,
        "base_nodes": [
            {
                "id": node_id,
                **meta,
            }
            for node_id, meta in BASE_NODES.items()
        ],
        "views": {
            chart_a["view_id"]: chart_a,
            chart_b["view_id"]: chart_b,
            chart_c["view_id"]: chart_c,
        },
    }

    (OUT_DIR / "base_typed_graph_3d.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    for view in (chart_a, chart_b, chart_c):
        html = _html_template(view)
        (OUT_DIR / f"{view['view_id']}.html").write_text(html, encoding="utf-8")

    _write_index([chart_a, chart_b, chart_c], OUT_DIR)

    print(f"Wrote 3D topology artifacts to: {OUT_DIR}")


if __name__ == "__main__":
    main()
