---
name: roadmap-execution-companion
description: Use when detecting bottlenecks in the multi-WM roadmap, surfacing next-highest-leverage work, auditing doc claims against code/test/artifact reality, or comparing upstream approaches for adoption decisions.
---

# Roadmap Execution Companion

Use this skill when the task is to identify roadmap bottlenecks, recommend next actions, audit claims vs reality, compare upstream approaches, or prepare experiment matrices for subsystems.

## Read First

Before proposing any action, read these in order:

- `docs/economic_world_model/multi_wm_architecture_plan.md`
- `docs/economic_world_model/roadmap.md`
- `docs/economic_world_model/progress_log.md`
- `docs/economic_world_model/nightly_audit.md`
- `docs/economic_world_model/phase1_closure_standard.md`
- `docs/economic_world_model/phase2_closure_standard.md`
- `AGENTS.md`
- `scripts/TRAINING_MIGRATION_BACKLOG.json`
- `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json`
- `results/run_registry/` (if populated)
- `.agent/runs/` (recent runs, most recent first)

## Operating Rules

1. Always ground recommendations in current repo state (files, tests, artifacts), not memory or assumptions. Run verification commands before claiming status.
2. Distinguish structural work (schemas, contracts, wiring, tests) from data/GPU/asset work (weights, runtime installs, sensor corpora). Only structural work is actionable without external resources.
3. Respect phase sequencing discipline. Do not recommend pulling implementation priority away from the current phase before it is honestly closed. Read `.agent/claude_copilot.md` for the current implementation priority.
4. Prefer concrete verification commands over vague recommendations. Every recommendation must include a command the agent or developer can run to verify the recommendation is valid and the work is complete.
5. When comparing upstream approaches, always specify: what to borrow, what to adapt, what to ignore, and why. Use the three-bucket classification: `copy directly`, `adapt carefully`, `do not import`.
6. Every recommendation must include:
   - **What**: specific action (file, module, schema, test)
   - **Why now**: what makes this the highest-leverage next step
   - **Unblocks**: what downstream work this enables
   - **Verify**: command to confirm the work is done and correct
   - **Do NOT**: explicit anti-pattern or scope boundary
7. Keep outputs bounded. Produce a ranked list of 3-5 items, not unbounded essays. If more than 5 items are relevant, rank and truncate with a note about what was deprioritized and why.
8. Mark each recommendation with:
   - **Confidence**: `high` / `medium` / `low` (based on how well the repo state supports the recommendation)
   - **Blocking**: `blocks-phase-exit` / `blocks-downstream` / `nice-to-have`
9. Do not recommend work that is already done (check code and tests, not just docs) or work that is externally blocked (GPU, weights, sensor data) unless the recommendation is to document the blocker.
10. Prefer bounded artifact outputs over elegant commentary. If uncertainty cannot yet be reduced into a bounded artifact shape, explicitly state what information is missing.
11. When a recommendation implies remote execution, specify the expected `run_class` and `epistemic_status` so the resulting run is decision-legible rather than merely executable.

## Preferred Output Types

### 1. `ranked_next_actions`

Ranked next 3-5 highest-leverage tasks.

```
## Ranked Next Actions (YYYY-MM-DD)

### 1. [Action title]
- **What**: ...
- **Why now**: ...
- **Unblocks**: ...
- **Verify**: `command`
- **Do NOT**: ...
- **Confidence**: high | medium | low
- **Blocking**: blocks-phase-exit | blocks-downstream | nice-to-have
```

### 2. `bottleneck_report`

Ranked list of bottlenecks with severity and suggested resolution.

```
## Bottleneck Report (YYYY-MM-DD)

| Rank | Bottleneck | Severity | Type | Suggested Resolution |
|------|-----------|----------|------|---------------------|
| 1    | ...       | high     | structural | ... |
| 2    | ...       | medium   | external   | ... |
```

### 3. `claim_vs_code_audit`

Comparison of doc claims vs code/test/artifact reality.

```
## Claim vs Code Audit (YYYY-MM-DD)

| Claim (source doc) | Code/Test Evidence | Status |
|--------------------|-------------------|--------|
| "X is implemented" (roadmap.md) | `src/x.py` exists, `tests/test_x.py` passes | verified |
| "Y is wired" (progress_log.md) | no test, stub only | unverified |
```

### 4. `run_comparison_summary`

Bounded summary of a benchmark-, promotion-, or deployment-oriented run family.

```
## Run Comparison Summary (YYYY-MM-DD)

- **Baseline**: ...
- **Candidate runs**: ...
- **What changed**: ...
- **What improved**: ...
- **What regressed**: ...
- **Confidence level**: low | medium | high
- **Promotion implication**: ...
- **Roadmap implication**: ...
- **Next recommended action**: ...
```

### 5. `upstream_comparison`

Structured comparison of an upstream approach vs repo-native approach.

### 6. `experiment_matrix`

Proposed experiment matrix for a specific subsystem.

### 7. `refactor_recommendation`

Specific refactor with verification commands.

## Invocation Pattern

Run this companion:

- **After major Codex tranches**: to detect what the tranche left unfinished or misaligned
- **Before planning sessions**: to surface the actual highest-leverage next work
- **When bottleneck detection is needed**: when progress stalls or priorities are unclear
- **Weekly strategic review**: to audit claims vs reality and adjust the roadmap

## Queue-Prioritization Awareness

When comparing multiple runnable ideas, prefer or surface fields such as:

- `wm`
- `subsystem`
- `blocker`
- `run_class`
- `epistemic_status`
- `expected_value`
- `estimated_cost_usd`
- `dependency_chain`
- `urgency`

These do not turn the companion into a scheduler. They make its outputs legible once there are multiple concurrent GPU windows.

## Verification

After producing any output, run the repo verification loop to confirm the repo state matches the companion's understanding:

```bash
python3 -m compileall src/ && pytest tests/ -v
```
