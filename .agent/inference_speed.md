# Inference-Speed Workflow

Actionable rules adapted from "Shipping at Inference-Speed" for agent-driven development.

## Core Principles

### 1. Queue work, don't block

**Problem**: Waiting for one task to complete before starting the next wastes time.

**Solution**: Use queues and background execution.

```bash
# Bad: Blocking
./scripts/codex/run.sh "task 1"  # Wait...
./scripts/codex/run.sh "task 2"  # Wait...

# Good: Queue and continue
./scripts/codex/enqueue.sh "task 1"
./scripts/codex/enqueue.sh "task 2"
./scripts/codex/worker.sh &  # Process in background
# Continue with other work
```

### 2. Keep prompts short and iterative

**Problem**: Giant prompts are hard to parse and often misunderstood.

**Solution**: Ask, align, then build.

```
# Bad: One-shot mega-prompt
"Implement complete user authentication with login, logout,
password reset, email verification, 2FA, session management..."

# Good: Iterative
1. "What auth patterns exist in this codebase?"
2. "Plan: add email/password login. Review before building."
3. "Implement the plan we discussed."
```

### 3. CLI-first verification

**Problem**: No way to verify changes without manual testing.

**Solution**: Build CLI harnesses for quick feedback.

```bash
# Every feature should be testable with a command
npm test -- --grep "feature-name"
./scripts/smoke.sh
python -c "from mymod import feature; feature.test()"
```

### 4. Engineer the repo for agents

**Problem**: Scattered docs, inconsistent naming, hidden conventions.

**Solution**: Predictable structure.

```
docs/           # All docs here
  README.md     # Index of docs
  api.md        # API reference
  setup.md      # Setup guide
scripts/        # All scripts here
  test.sh
  build.sh
src/            # Clear module boundaries
tests/          # Mirror src/ structure
```

### 5. Treat compaction as review

**Problem**: Context gets lost after compaction/restart.

**Solution**: Write summaries that survive compaction.

```
# After completing work, summarize:
"Completed: Added login endpoint in src/auth/login.ts
Tests: 3 new tests in tests/auth/login.test.ts
Next: Need to add password reset endpoint"
```

### 6. Avoid huge one-shot prompts

**Problem**: Token limits, lost context, misunderstandings.

**Solution**: Composable steps.

```
# Refactor pattern
1. Inspect: "Show me the current auth implementation"
2. Plan: "Plan refactoring auth to use sessions"
3. Build: "Implement step 1 of the plan"
4. Verify: "Run auth tests"
5. Repeat for remaining steps
```

## Suspicion Triggers

Be suspicious when:

| Signal | Likely Issue | Action |
|--------|--------------|--------|
| Task "should be quick" but drags | Scope too large | Split into smaller tasks |
| No events for 5+ minutes | Silent failure | Check logs, consider restart |
| Agent seems confused | Context lost | Re-read key files, re-state goal |
| Same error repeated | Stuck in loop | Different approach needed |
| Tests pass but behavior wrong | Wrong tests | Verify test actually tests feature |

## Auto-Splitting Rules

When a task stalls, split by:

1. **Directory/module**: "Do `src/auth/` first, then `src/api/`"
2. **Deliverable type**: "Write tests first, then implementation"
3. **Size**: "Do first half of files, then second half"
4. **Dependency order**: "Do core utils first, then consumers"

## Time Boxing

Set expectations:

| Task Type | Expected Duration | Split If Exceeds |
|-----------|-------------------|------------------|
| Simple fix | 2-5 minutes | 10 minutes |
| Single feature | 10-30 minutes | 1 hour |
| Multi-file refactor | 30-60 minutes | 2 hours |
| Large feature | 1-2 hours | 4 hours |

## Quick Reference

```bash
# Start task queue
./scripts/codex/worker.sh &

# Add task
./scripts/codex/enqueue.sh "task description"

# Check progress
./scripts/codex/worker_status.sh

# If stuck, diagnose
./scripts/codex/diagnose.sh

# List docs (for context recovery)
./scripts/docs/list.sh
```

## Anti-Patterns

- Writing 500+ word prompts
- Not running tests after changes
- Ignoring timeout warnings
- Re-trying the same failing approach
- Not reading error messages
- Assuming instead of verifying
