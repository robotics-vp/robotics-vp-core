# Semantic Authority Promotion

Date: 2026-03-24

## Intent

The semantic world model should not remain permanently advisory. It also should not become a sovereign controller. The target shape is a bounded control plane:

- semantic world model = canonical routing/state substrate
- transformers = learned policy over that substrate
- meta-nodes = bounded executor lanes
- execution preconditions = promotion gate
- frozen Phase B baseline = fallback floor

## Promotion Stages

### Stage A: Advisory Packet

Current additive packet family:

- `SemanticWorldModelState`
- `SemanticSnapshot`
- `OrchestratorAdvisory`
- transformer outputs/results

These surfaces must remain typed, serializable, replayable, and evidence-backed.

### Stage B: Preconditioned Execution

Promote advisory outputs into executable packets only when readiness is explicit.

Required packet fields:

- `execution_mode`
- `bounded_actions`
- `execution_preconditions`
- `execution_work_order`

This is the first safe non-advisory step because execution remains narrow and auditable.

### Stage C: Bounded Meta-Node Authority

Let selected semantic/meta-node lanes act directly:

- sampler and curriculum routing
- objective-preset selection
- energy-profile selection
- backend selection
- semantic-memory refresh requests
- Stage-2 enrichment requests
- risk/recovery routing

Do not grant blanket planner/controller sovereignty.

### Stage D: Learned Control Plane

Once real semantic grounding and execution evidence are dense enough:

- learned transformers stop imitating heuristics only
- transformer outputs become primary bounded-routing decisions
- shell activation backlogs promote specific transformer lanes from advisory to executable

## Order Of Promotion

1. `MetaTransformer`
   Promote from suggestion packet to bounded routing packet for objective/backend/energy/data-mix decisions.
2. `OrchestrationTransformer`
   Promote from context encoder + stub tool sampler to bounded tool-plan packet over semantic state.
3. `SemanticOrchestratorV2`
   Keep as the meta-node routing shell, but make more consumers execute its bounded actions.
4. `PipelineManager`
   Stop treating transformer output as optional decoration; carry execution packets through iteration planning.
5. sampler/curriculum/replay
   Expand bounded consumers before any attempt at broader planner authority.

## What Landed In This Pass

- shared semantic-WM transformer bridge in `src/orchestrator/semantic_transformer_bridge.py`
- `MetaTransformer.propose_plan(...)` now consumes semantic world-model state directly and emits:
  - semantic-aware preset/backend/energy/data-mix decisions
  - bounded orchestration steps
  - execution preconditions
  - execution work order
- `OrchestrationTransformer` now:
  - encodes semantic world-model features into context
  - biases tool selection from semantic/econ/data state
  - emits execution mode, activation plan, execution preconditions, and activation work order
- `run_pipeline_step_with_causal_order(...)` now carries the meta-transformer execution packet instead of swallowing it as a soft suggestion only

## Next Promotions

1. compile transformer bounded actions into typed executor-specific work orders instead of generic metadata payloads
2. add shell-activation backlog items for transformer lanes once real execution evidence accumulates
3. bind replay/receipt promotion evidence to successful transformer work-order execution
4. replace the heuristic recommendation rules with learned routing over the same packet shape

## Guardrails

- keep all promotions additive
- preserve the frozen Phase B baseline as canonical fallback
- require explicit readiness before non-advisory execution
- keep actions bounded, replayable, and attributable
- treat semantic state as control-plane substrate, not as unchecked truth
