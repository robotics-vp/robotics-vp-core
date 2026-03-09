# Nightly Audit Runbook

## Purpose

The nightly loop exists to move the repo forward one disciplined additive step at a time. It should behave like a staff engineer:

- re-read the roadmap and progress log
- rescan the repo for drift and already-landed work
- choose one highest-value additive task
- verify before claiming progress
- update status without spamming or pretending work happened

## Preferred Autonomous Path

- Primary: Codex app automation using the repo-local skill and prompt in `docs/economic_world_model/AUTOMATION_SPEC.md`
- Secondary: local CLI runner
- Fallback: GitHub/cloud runner when app or local execution is unavailable

## Repo-Implemented Nightly Paths

- Local actual execution: `scripts/economic_world_model/run_nightly_codex_task.sh --mode cli`
- Local queued execution: `scripts/economic_world_model/run_nightly_codex_task.sh --queue-only`
- GitHub/cloud audit: `.github/workflows/economic-world-model-nightly.yml`
- Optional GitHub/cloud Codex execution: same workflow, but only when `CODEX_API_KEY` is configured

## Audit Inputs

Every nightly pass should read:

- `docs/economic_world_model/architecture_gap_analysis.md`
- `docs/economic_world_model/roadmap.md`
- `docs/economic_world_model/progress_log.md`
- `docs/economic_world_model/implementation_notes.md`
- `scripts/TRAINING_MIGRATION_BACKLOG.json`
- `AGENTS.md`

## Default Verification

The audit script runs these by default:

- `./scripts/agent/verify.sh`
- `python3 -m compileall src scripts/economic_world_model -q`
- `python3 -m pytest -q tests/test_runtime_packets.py tests/embodiment/test_registry.py tests/test_objective_runtime_builder.py tests/test_constraint_set.py tests/test_pricing_sentinel.py tests/test_value_ledger.py`

## Drift Rules

Nightly drift should be flagged when:

- required docs, skill files, workflow files, or new middleware scaffolds are missing
- the training backlog is newer than the progress log
- core roadmap documents do not mention scaffolds that already landed
- verification starts failing

## Task Selection Policy

The nightly pass should prefer, in order:

1. docs or automation substrate fixes that unblock the roadmap
2. scaffolding code that increases runtime legibility
3. additive sidecar wiring into existing shadow/replay paths
4. tests and verification hardening
5. behavior-changing work only behind explicit flags and only after the previous layers exist

For the active video-world-model subset, the nightly pass should further prefer:

1. Week 6.5 real-video grounding and reconstruction sidecars
2. Week 6.5 teacher-runtime hardening and explicit fallback semantics
3. Week 6.75 governed supervision/value-target wiring
4. test and smoke coverage for the above
5. training-backlog refresh only after the previous items are materially landed

## Skip Rules

- Do not update the GitHub status issue when the audit digest is unchanged.
- Do not run Codex automatically when the audit does not mark a task as safe to execute.
- Do not modify the stable Phase B checkpoint, legacy baseline world-model math, trust-net, `w_econ`, or lambda controller math.
- Additive successor modules in `src/world_model/` are allowed only when they preserve the stable baseline as the rollback anchor and remain advisory/governed.
- Do not claim GitHub/cloud or app automation execution unless the relevant credentials or UI automation actually exist.

## Current Baseline

- Canonical packet scaffold: present
- Embodiment registry scaffold: present
- Packet sidecar wiring in shadow runtime/replay: not present yet
- Actual local nightly Codex runner: present
- Actual GitHub/cloud Codex runner: present but credential-gated
- App automation: manual setup required
