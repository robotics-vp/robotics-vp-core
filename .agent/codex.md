# Codex Sub-Agent Contract

## Overview

Codex can be used as a sub-agent to handle background tasks. This document defines the contract for Codex integration, ensuring reliable operation with proper observability and failure handling.

## Dual-Path Execution

Codex can be invoked via two paths:

### Path 1: MCP (Model Context Protocol)

```bash
# Uses MCP server if configured
./scripts/codex/run_mcp.sh "task description"
```

### Path 2: CLI (Command Line)

```bash
# Uses codex exec directly
./scripts/codex/run_cli.sh "task description"
```

### Router (Automatic Selection)

```bash
# Prefers MCP if available, falls back to CLI
./scripts/codex/run.sh "task description"

# Force a specific mode
CODEX_MODE=cli ./scripts/codex/run.sh "task description"
CODEX_MODE=mcp ./scripts/codex/run.sh "task description"
```

## Task Contract

Every Codex job MUST:

### 1. Acknowledge start (within 60 seconds)

- Job created
- Log file created at `.agent/runs/<job_id>/`
- First event received

### 2. Emit heartbeats (every 5 minutes minimum)

- Any event stream activity counts as heartbeat
- Reading files counts (Codex may read silently for several minutes)
- Explicit progress messages preferred

### 3. Produce structured output

Final output MUST include:

```json
{
  "summary": "Brief description of what was done",
  "files_changed": ["path/to/file1", "path/to/file2"],
  "commands_run": ["npm test", "npm run lint"],
  "tests": {
    "run": true,
    "passed": 42,
    "failed": 0
  },
  "risks": ["Potential issues or concerns"],
  "next_steps": ["Suggested follow-up actions"]
}
```

### 4. Handle failures explicitly

- Missing output = failure
- Timeout = failure (retry with smaller scope)
- Error = failure (with diagnostics)

## Watchdog Behavior

The watchdog (`scripts/codex/watchdog.sh`) monitors running jobs:

| Condition | Action |
|-----------|--------|
| No ack within 60s | Mark failed, retry |
| No heartbeat for 5min | Mark stalled |
| No heartbeat for 10min | Mark failed, auto-split task |
| Task completes | Verify output, mark done |

### Auto-splitting

When a task times out, the watchdog splits it:

1. By directory/module (if applicable)
2. By deliverable (tests first, then feature)
3. By size (half the files)

Maximum 3 retry attempts before escalating.

## Output Locations

All Codex runs write to `.agent/runs/<job_id>/`:

```
.agent/runs/
└── 20240115-143022-abc123/
    ├── meta.json       # Job metadata (timestamps, mode, command, PID)
    ├── events.jsonl    # Event stream (from --json flag)
    ├── stdout.log      # Standard output
    ├── stderr.log      # Standard error
    ├── final.json      # Structured output (if schema enforced)
    └── final.md        # Human-readable summary
```

## CLI Flags Reference

### Essential flags for automation

```bash
codex exec \
  --json \                    # Stream events as JSON (essential for monitoring)
  --output-schema schema.json # Enforce structured output
  -o output.txt \             # Write final message to file
  "task description"
```

### Resume sessions

```bash
# Resume last session
codex exec resume --last

# Resume specific session
codex exec resume <session-id>
```

### Approval policies

```bash
# Minimal permissions (recommended for automation)
codex exec --approval-policy=deny-all "task"

# Allow reads only
codex exec --approval-policy=read-only "task"
```

## Queue System

For non-blocking task delegation:

```bash
# Add to queue
./scripts/codex/enqueue.sh "implement feature X"

# Start worker (runs in background)
./scripts/codex/worker.sh &

# Check status
./scripts/codex/worker_status.sh
```

Queue stored in `.agent/queue/codex.jsonl`.

## Diagnostics

### Quick health check

```bash
./scripts/codex/diagnose.sh
```

This will:
1. Test CLI path (120s timeout)
2. Test MCP path (if configured)
3. Report findings

### Check job status

```bash
./scripts/codex/status.sh          # Latest job
./scripts/codex/status.sh <job_id> # Specific job
```

### Cancel running job

```bash
./scripts/codex/cancel.sh          # Cancel current job
./scripts/codex/cancel.sh <job_id> # Cancel specific job
```

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No output for hours | Missing --json flag | Use --json for event streaming |
| Silent failure | Output limit hit | Increase limits in config |
| Auth errors | Missing OPENAI_API_KEY | Set environment variable |
| Timeout | Task too large | Split into smaller tasks |

See `docs/codex_postmortem.md` for detailed diagnostics.
