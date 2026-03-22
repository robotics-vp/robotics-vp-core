---
name: economic-world-model-roadmap
description: Use when advancing or auditing the repo's economic-world-model readiness roadmap, including nightly audits, local or cloud Codex execution, and additive middleware work around packets, embodiment normalization, event/evidence/econ/governance scaffolding.
---

# Economic World Model Roadmap

Use this skill when the task is to advance the roadmap, run the nightly audit, or choose the next additive economic-world-model preparation step.

## Read First

- `docs/economic_world_model/architecture_gap_analysis.md`
- `docs/economic_world_model/roadmap.md`
- `docs/economic_world_model/progress_log.md`
- `docs/economic_world_model/nightly_audit.md`
- `docs/economic_world_model/implementation_notes.md`
- `docs/economic_world_model/AUTOMATION_SPEC.md`
- `scripts/TRAINING_MIGRATION_BACKLOG.json`
- `AGENTS.md`

## Operating Rules

- Keep VLA and foundation-model paths external, pluggable, and sidecar/advisory.
- Prefer docs, scaffolding, tests, and additive wiring before invasive rewrites.
- Preserve the stable Phase B checkpoint and legacy baseline math as frozen unless explicitly authorized.
- Additive successor modules in `src/world_model/` are allowed only when they preserve the stable baseline as the rollback anchor and stay advisory/governed.
- Land one highest-value additive step per pass.
- Update `docs/economic_world_model/progress_log.md` and `docs/economic_world_model/implementation_notes.md` whenever a roadmap change lands.
- Run verification before closing the pass.
- Do not leave automation commits local-only. If you create a commit, publish it to `origin/main` when safe or push a timestamped feature branch and report the exact published ref.

## Nightly Loop

1. Run `python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md`.
2. Read the audit summary and choose the single highest-value additive next step.
3. If the audit says no safe task should run, stop after refreshing the summary and note why.
4. If a safe task exists, make a scoped change only.
5. Re-run verification, update the progress log and implementation notes, publish any created commit via `bash scripts/economic_world_model/publish_codex_change.sh --base-branch main --feature-prefix codex/ewm-nightly`, and leave a concise summary with the published ref or push blocker.

## Real Execution Paths

- Preferred: Codex app automation using the prompt in `docs/economic_world_model/AUTOMATION_SPEC.md`
- Local CLI execution: `bash scripts/economic_world_model/run_nightly_codex_task.sh --mode cli`
- Local queueing: `bash scripts/economic_world_model/run_nightly_codex_task.sh --queue-only`
- GitHub/cloud execution: `.github/workflows/economic-world-model-nightly.yml` when `CODEX_API_KEY` is configured

If credentials are missing, report that the execution path is unavailable. Do not claim the automation ran when it did not.
