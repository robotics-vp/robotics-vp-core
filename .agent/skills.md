# Agent Skills and Working Style

## Quality Bar

- **Correctness first** - Working code over clever code
- **Minimal changes** - Do exactly what's asked, no more
- **Verify before claiming done** - Run tests, check output
- **Leave code better** - But only in files you're touching

## Where to Look First

1. `README.md` - Project overview and quick start
2. `docs/` - Detailed documentation
3. `package.json` / `pyproject.toml` - Available scripts and dependencies
4. `.github/workflows/` - CI configuration shows how tests are run
5. `tests/` or `__tests__/` - Existing test patterns

## Working Style

### Start small, verify often

```
1. Understand the request
2. Locate relevant code
3. Make minimal change
4. Run tests
5. Iterate or complete
```

### Ask before assuming

If requirements are ambiguous:
- Ask clarifying questions
- Don't guess at business logic
- Check for existing patterns in codebase

### Commit hygiene

- Commit when a logical unit is complete
- Use conventional commit format
- Keep commits atomic and reversible

## Repo Commands

<!-- CUSTOMIZE: Replace with actual discovered commands -->

| Purpose | Command |
|---------|---------|
| Install | `npm install` |
| Dev server | `npm run dev` |
| Test | `npm test` |
| Test (watch) | `npm test -- --watch` |
| Lint | `npm run lint` |
| Lint (fix) | `npm run lint -- --fix` |
| Build | `npm run build` |
| Type check | `npm run typecheck` |

## Agent-Friendly Structure

This repo follows conventions that help agents work efficiently:

- **`docs/`** - All documentation lives here
- **`scripts/`** - Utility scripts organized by purpose
- **`src/`** or **`lib/`** - Source code with clear module boundaries
- **`tests/`** - Tests mirror source structure
- **Predictable names** - Files named after what they export/do

### Navigation tips

```bash
# List all docs
./scripts/docs/list.sh

# Find a file
find . -name "*.ts" | grep -i "component"

# Search code
grep -r "functionName" src/
```

## Quality Checklist

Before marking any task complete:

- [ ] Tests pass (`npm test`)
- [ ] No lint errors (`npm run lint`)
- [ ] Types check (`npm run typecheck`)
- [ ] No regressions in related functionality
- [ ] Commit message is clear and follows format
