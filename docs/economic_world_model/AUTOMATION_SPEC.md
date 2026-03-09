# Automation Spec

## What Is Repo-Implemented Versus Manual

Repo-implemented now:

- `scripts/economic_world_model/nightly_audit.py`
- `scripts/economic_world_model/run_nightly_codex_task.sh`
- `scripts/economic_world_model/update_status_issue.py`
- `.github/workflows/economic-world-model-nightly.yml`

Manual setup still required:

- creating a Codex app automation in the UI
- configuring `CODEX_API_KEY` in GitHub secrets if you want GitHub/cloud Codex execution
- optionally configuring `CODEX_CLOUD_ENV_ID` if you want the local runner to use Codex cloud mode instead of local CLI mode

## Codex App Automation

This is the preferred autonomous path for this roadmap.

Recommended schedule:

- nightly at 02:17 Europe/Madrid

Prompt to paste into the Codex app automation UI:

```text
Follow [$economic-world-model-roadmap](/Users/amarmurray/robotics-vp-core/codex_skills/economic-world-model-roadmap/SKILL.md).
Read docs/economic_world_model/architecture_gap_analysis.md, docs/economic_world_model/roadmap.md, docs/economic_world_model/progress_log.md, docs/economic_world_model/nightly_audit.md, and docs/economic_world_model/implementation_notes.md.
Run python3 scripts/economic_world_model/nightly_audit.py --output-json artifacts/economic_world_model/nightly_audit_summary.json --output-markdown artifacts/economic_world_model/nightly_audit_summary.md.
Choose the single highest-value additive next task.
Prefer docs, scaffolding, tests, and sidecars before invasive rewrites.
Do not modify the stable Phase B checkpoint, legacy baseline world-model math, trust_net, w_econ lattice math, lambda controller equations, or src/controllers/synthetic_weight_controller.py core logic.
Additive successor modules inside src/world_model/ are allowed only when they preserve the stable baseline as the rollback anchor and stay advisory/governed.
If a safe task exists, implement it, run verification, update docs/economic_world_model/progress_log.md and docs/economic_world_model/implementation_notes.md, and leave a concise summary suitable for a GitHub issue comment.
If no safe task exists, refresh the audit summary and explain why execution was skipped.
```

Expected output behavior:

- refresh `artifacts/economic_world_model/nightly_audit_summary.json`
- refresh `artifacts/economic_world_model/nightly_audit_summary.md`
- make one scoped additive change or skip explicitly
- update `docs/economic_world_model/progress_log.md`
- update `docs/economic_world_model/implementation_notes.md`

Current autonomous priority for the video-world-model subset:

1. real-video reconstruction/calibration sidecars
2. teacher-runtime hardening and explicit fallback metadata
3. governed video supervision bundles with counterfactual/value targets
4. only then refresh the learned video-state training backlog or training scaffolds

The automation should not jump to model-training work while those prior stages remain open.

## Local Cron / Launchd Path

Use this only when app automation is unavailable or you want a parallel fallback.

CLI command:

```bash
cd /Users/amarmurray/robotics-vp-core && bash scripts/economic_world_model/run_nightly_codex_task.sh --mode cli
```

Cron example:

```cron
17 2 * * * /bin/zsh -lc 'cd /Users/amarmurray/robotics-vp-core && bash scripts/economic_world_model/run_nightly_codex_task.sh --mode cli >> /tmp/economic_world_model_nightly.log 2>&1'
```

Cloud-mode variant:

```bash
cd /Users/amarmurray/robotics-vp-core && bash scripts/economic_world_model/run_nightly_codex_task.sh --mode cloud --env "$CODEX_CLOUD_ENV_ID"
```

## GitHub / Cloud Runner Path

Use this when you want cloud-hosted fallback automation or artifact generation outside the app.

To enable Codex execution on GitHub runners:

1. Add repository secret `CODEX_API_KEY`.
2. Keep `.github/workflows/economic-world-model-nightly.yml` enabled.
3. Review uploaded artifacts from the `codex_execute` job:
   - `.agent/runs`
   - `artifacts/economic_world_model/codex-nightly.patch` when a diff is generated

Without `CODEX_API_KEY`, the workflow still performs the nightly audit and issue update, but it will not pretend Codex executed.
