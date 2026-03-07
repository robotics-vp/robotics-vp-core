# Running The Economic World Model Skill

## Purpose

The repo-local skill in `codex_skills/economic-world-model-roadmap/SKILL.md` is the operating guide for nightly audits and roadmap work. It keeps Codex focused on one additive step at a time instead of rewriting the stack.

## Codex App

This is the preferred autonomous path.

The repo cannot create app automations from git content alone, but it is now prepared for app-first autonomous use:

- repo-local skill: `codex_skills/economic-world-model-roadmap/SKILL.md`
- roadmap docs: `docs/economic_world_model/*`
- audit artifacts: `artifacts/economic_world_model/*`
- exact app setup instructions: `docs/economic_world_model/AUTOMATION_SPEC.md`

For one-off app usage without automation:

- open the repo in the app
- tell Codex to follow `codex_skills/economic-world-model-roadmap/SKILL.md`
- point it at the roadmap docs and the latest audit summary under `artifacts/economic_world_model/`

## Local CLI

Use the real local actuation path when you want Codex to perform a nightly pass against the checked-out repo:

```bash
python3 scripts/economic_world_model/nightly_audit.py
bash scripts/economic_world_model/run_nightly_codex_task.sh --mode cli
```

Requirements:

- Codex CLI installed
- `CODEX_API_KEY` or `OPENAI_API_KEY` set
- repo dependencies installed for verification

Queue instead of execute:

```bash
bash scripts/economic_world_model/run_nightly_codex_task.sh --mode cli --queue-only
```

## GitHub / Cloud

The scheduled workflow `.github/workflows/economic-world-model-nightly.yml` always performs the repo audit and updates a single status issue.

It only runs Codex itself when:

- `CODEX_API_KEY` is configured in repository secrets
- the audit marks the next task as safe to execute

The cloud job runs Codex CLI on the GitHub runner, captures `.agent/runs`, and uploads any generated diff as an artifact. This is a real execution path, not a placeholder.

## Recommended Human Loop

1. Review `docs/economic_world_model/progress_log.md`.
2. Run the audit.
3. Let the nightly runner execute the single next additive step.
4. Review the changed files and verification results.
