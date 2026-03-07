# Quality Cleanup Plan (Ruff + Mypy)

This repository uses a ratchet model:
- Do not let quality regress.
- Reduce debt in scoped slices.
- Keep feature velocity.

## Sacred Lane
The `golden_path_contract` CI job is a non-negotiable lane:
- compile checks
- `golden_path` tests
- minimal golden-path smoke run

This protects core architecture contracts while other cleanup continues.

## Ruff Phases
1. Phase 1 (auto-fix slices)
   - Run `ruff check <package> --fix`
   - Commit one package at a time.
2. Phase 2 (touched-files enforcement)
   - Keep CI strict for changed files in active work.
3. Phase 3 (repo-wide enforcement)
   - Lower baseline until full pass.

## Mypy Phases
1. Phase 1 (baseline ratchet)
   - Keep `config/quality_ratchet.json` count as the upper bound.
2. Phase 2 (subsystem cleanup)
   - Start with golden path and economics interfaces.
   - Then valuation/orchestrator.
   - Then broader RL/vision areas.
3. Phase 3 (strict package-by-package)
   - Enable stricter typing in selected packages when counts are near zero.

## Ratchet Commands
```bash
python3 scripts/ci/check_ruff_ratchet.py
python3 scripts/ci/check_mypy_ratchet.py
```

## Suggested Commit Sequence
1. `chore: add golden-path contract CI lane`
2. `chore: add ruff and mypy ratchet checks`
3. `chore: ruff auto-fix economics package`
4. `chore: ruff auto-fix valuation package`
5. `chore: mypy cleanup golden-path interfaces`
