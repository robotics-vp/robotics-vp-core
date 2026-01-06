# Agent Guidelines

This document defines how AI agents operate in this repository. It is tool-agnostic and applies to any agent (Claude, Codex, Cursor, etc.).

## Repository Purpose

<!-- CUSTOMIZE: Replace with actual repo description -->
[Describe the purpose of this repository in 1-2 sentences.]

## Build / Test / Lint / Format

<!-- CUSTOMIZE: Replace placeholders with actual commands discovered from package.json, pyproject.toml, Makefile, or CI -->

| Action | Command |
|--------|---------|
| Install dependencies | `npm install` or `pip install -e .` |
| Build | `npm run build` or `python -m build` |
| Test | `npm test` or `pytest` |
| Lint | `npm run lint` or `ruff check .` |
| Format | `npm run format` or `ruff format .` |
| Type check | `npm run typecheck` or `mypy .` |

### Quick verification loop

```bash
# Run this after any change to verify nothing is broken
./scripts/verify.sh  # or: npm test && npm run lint
```

## Definition of Done

A task is complete when:

1. **Tests pass** - All existing tests pass, new functionality has tests
2. **No regressions** - Existing functionality works as before
3. **Linting passes** - No lint errors or warnings
4. **Types check** - No type errors (if applicable)
5. **Docs updated** - README, API docs, or comments updated if behavior changed
6. **Commits are clean** - Small, focused commits with clear messages

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

### Destructive operations

Before running any destructive command:
1. Verify you're in the correct directory
2. Verify you're on the correct branch
3. Prefer dry-run flags when available (`--dry-run`, `-n`)
4. Never force-push to main/master without explicit approval

## CLI-First Development

Whenever feasible, build and verify changes using CLI commands:

1. **Build a minimal test harness first** - Before implementing features, create a way to run and verify them from the command line
2. **Prefer scriptable interfaces** - CLIs over GUIs, JSON output over human-only output
3. **Close the loop fast** - The path from "change" to "verified working" should be one command

### Example workflow

```bash
# 1. Make change
# 2. Run verification
npm test -- --grep "feature-name"
# 3. Check output
# 4. Iterate
```

## Agent-Friendly Structure

To help agents navigate this repo efficiently:

- **Docs in `docs/`** - Keep documentation in a predictable location
- **Scripts in `scripts/`** - Runnable utilities organized by purpose
- **Predictable naming** - Use clear, conventional file names
- **Module READMEs** - Complex modules should have their own README
- **Type hints** - Use type annotations for better code understanding

## Getting Help

- Check `docs/` for detailed documentation
- Check `.agent/` for agent-specific configuration and guidance
- Run `./scripts/docs/list.sh` to see available documentation
