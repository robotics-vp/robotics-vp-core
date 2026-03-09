# Quality Ratcheting

The repository still carries legacy lint and type debt, so the short-term goal is
to prevent regressions while cleanup proceeds in slices.

## Baseline

The active baselines live in `config/quality_ratchet.json` and are intended to
move downward over time, never upward without an explicit decision.

Current categories:

- `ruff`: total `python3 -m ruff check .` findings
- `mypy`: total `python3 -m mypy src/` errors

Intentional exception:

- `E402` is ignored for direct-execution files under `scripts/`, `tests/`, and
  `experiments/` because those entrypoints bootstrap the repo root onto
  `sys.path` before importing `src.*`.

## Commands

Run the ratchet checks directly:

```bash
python3 scripts/ci/check_ruff_ratchet.py
python3 scripts/ci/check_mypy_ratchet.py
```

Run the broader verification loop when a change is expected to be clean:

```bash
python3 -m compileall src scripts tests -q
python3 -m pytest tests/ -m "not slow and not no_mujoco" -q
```

## Cleanup Strategy

Use two sweep types:

1. Repo-level sweeps for tooling, docs, and enforcement gaps.
2. Targeted code sweeps for one module family at a time, keeping behavior stable.

Preferred order:

1. Fix root scripts, `experiments/`, and low-risk utility files.
2. Fix concentrated module families with repeated issues such as `src/vision/`,
   `src/orchestrator/`, and `src/envs/`.
3. Once counts reach zero, replace ratchets with full hard gates.

## Rules

- Do not raise the baseline to mask new debt.
- Keep changes additive around stable frozen zones; for Phase B that means preserving the stable checkpoint and baseline math while allowing additive successor scaffolding beside them.
- When a cleanup slice lands, update `config/quality_ratchet.json` only after the
  lower counts are verified and intended to become the new floor.

See also: `docs/quality_debt_followups.md` for the current hotspot map and the build-ergonomics notes from the latest sweep.
