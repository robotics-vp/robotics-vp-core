# Agent Guidelines

This document defines how AI agents operate in this repository. It is tool-agnostic and applies to any agent (Claude, Codex, Cursor, etc.).

## Repository Purpose

Economics-first robotics stack: robots priced like labor, not software. Full-stack econ + data valuation for video-to-policy, HRL, VLA, and synthetic flywheels. Target flywheel: better data → better robots → better economics → better data.

## Build / Test / Lint / Format

| Action | Command |
|--------|---------|
| Install dependencies | `pip install -e .` |
| Test | `pytest` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Type check | `mypy src/` |
| Compile check | `python3 -m compileall src/` |

### Quick verification loop

```bash
# Run this after any change to verify nothing is broken
python3 -m compileall src/ && pytest tests/ -v
```

### Key smoke tests

```bash
# Feature extractor sanity
python3 scripts/test_episode_features.py

# Dishwashing smoke + summaries
python3 scripts/smoke_test_dishwashing_sac.py --episodes 5 --econ-preset toy

# Phase C HRL/VLA smoke
python3 scripts/smoke_test_phase_c_hrl_vla.py --episodes 3

# Workcell env suite
python3 scripts/smoke_workcell_env.py
```

## Definition of Done

A task is complete when:

1. **Tests pass** - `pytest tests/ -v` passes
2. **No regressions** - Existing functionality works as before
3. **Compiles** - `python3 -m compileall src/` has no errors
4. **Phase B frozen** - Do NOT modify world model, trust_net, w_econ_lattice, or λ controller math
5. **Docs updated** - README or module docs updated if behavior changed
6. **Commits are clean** - Small, focused commits with clear messages

## Architecture Constraints

### Phase B is FROZEN

Do not modify:
- World model math (`src/world_model/`)
- `checkpoints/stable_world_model.pt`
- Trust net, w_econ lattice, λ controller equations
- `src/controllers/synthetic_weight_controller.py` core logic

### Additive-only zones

- Energy bench: experimental, additive only
- Orchestrator: advisory, no reward/weight changes
- Phase C scaffolding: HRL/VLA/SIMA for drawer+vase

## PR Hygiene

- **Small commits**: Each commit should do one thing
- **Clear messages**: Use conventional commits format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- **Atomic PRs**: One logical change per PR when possible
- **Tests included**: New features and bug fixes include tests
- **No commented code**: Remove dead code, don't comment it out

## Safety and Secrets

### Never commit secrets

- API keys, tokens, passwords
- `.env` files with real values
- Private keys, certificates
- Database connection strings with credentials

### Safe patterns

- Use environment variables for secrets
- Use `.env.example` with placeholder values
- Add secrets to `.gitignore`
- Use secret managers in production

## CLI-First Development

Whenever feasible, build and verify changes using CLI commands:

```bash
# 1. Make change
# 2. Run verification
python3 -m compileall src/ && pytest tests/ -v
# 3. Run relevant smoke test
python3 scripts/smoke_test_dishwashing_sac.py --episodes 2
# 4. Iterate
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/config/econ_params.py` | Economic parameters |
| `src/envs/` | Environments (dishwashing, drawer_vase, workcell) |
| `src/rl/reward_shaping.py` | MPL/EP/error + wage penalty |
| `src/controllers/` | Synthetic weight, λ budget controllers |
| `src/valuation/` | Data valuation, datapacks, w_econ lattice |
| `src/world_model/` | World model (FROZEN) |
| `src/hrl/`, `src/vla/`, `src/sima/` | Phase C scaffolding |

## Getting Help

- Check `docs/` for detailed documentation
- Check `CLAUDE_LEGACY.md` for historical context
- Check `.agent/` for agent-specific configuration
- Run `./scripts/docs/list.sh` to see available documentation
