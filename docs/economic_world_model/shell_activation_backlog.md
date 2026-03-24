# Shell Activation Backlog

Source of truth:
- `scripts/SHELL_ACTIVATION_BACKLOG.json`
- `src/orchestrator/shell_activation.py`

This backlog separates two classes of higher-shell promotion:

1. Present-tense bounded activations
- `semantic_orchestrator_preconditioned_routing`
- `pipeline_manager_preconditioned_iteration`
- `phase_h_advisory_integration_preconditioned_routing`
- `phase_h_controller_cycle_executor`
- `phase_h_economic_learner_budget_activation`

These entries auto-activate once the shared execution-precondition summary reports:
- enough reports exist
- enough reports are ready
- blocked count is zero
- mean readiness passes the item threshold

When active, the shell stops being merely advisory and emits a bounded activation plan plus a typed shell work order.

2. Future-training-only backlog
- `semantic_orchestrator_closed_loop_training`
- `pipeline_manager_training_run_executor`
- `phase_h_portfolio_autonomy`

These entries do not auto-execute. They stay in the backlog until future training runs surface stronger evidence such as:
- non-stub SceneTracks grounding
- live teacher-runtime grounding
- replay roundtrip completeness
- promotion trace completeness
- runtime manifests / settlement artifacts

Once those preconditions appear in the shared readiness summaries as satisfied checks, the backlog evaluator will move the item from `future_pending` to `activation_ready`, but manual review is still expected before granting direct training-run authority.
