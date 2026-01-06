# Codex CLI Guide

This document covers installation, configuration, and usage of the Codex CLI for agent-driven development.

## Installation

### Via npm (recommended)

```bash
npm install -g @openai/codex
```

### Via Homebrew (macOS)

```bash
brew install codex
```

### Verify Installation

```bash
codex --version
```

## Authentication

### Set API Key (Automation / CI)

```bash
export CODEX_API_KEY="your-codex-api-key"
```

Add to your shell profile (`~/.zshrc`, `~/.bashrc`) for persistence.

`CODEX_API_KEY` is supported in `codex exec` and is the preferred env var for explicit credentials in non-interactive flows.

### OAuth Login (Interactive)

```bash
codex login
```

Codex stores OAuth tokens in your configured credential store (see below).

### Verify Authentication

```bash
codex whoami
```

### Getting an API Key

1. Go to https://platform.openai.com/api-keys
2. Create a new key
3. Ensure your account has Codex access

### Credential Storage (OAuth Tokens)

Codex credential storage is configurable via:

```toml
cli_auth_credentials_store = "keyring" | "file" | "auto"
```

- `keyring`: store tokens in the system keychain (recommended on macOS)
- `auto`: choose the best available store
- `file`: store tokens in `~/.codex/auth.json`

If you use `file`, treat `~/.codex/auth.json` like a password: do not commit or share it.

## Basic Usage

### Execute a Task

```bash
codex exec "implement a function that calculates fibonacci numbers"
```

### Execute with JSON Output (Recommended for Automation)

```bash
codex exec --json "your task"
```

This streams events as JSON, enabling:
- Progress monitoring
- Event logging
- Failure detection

### Write Output to File

```bash
codex exec -o output.txt "your task"
```

### Use Output Schema (Structured Output)

```bash
codex exec --output-schema schema.json "your task"
```

Example schema:
```json
{
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "files_changed": {"type": "array", "items": {"type": "string"}},
    "tests_passed": {"type": "boolean"}
  },
  "required": ["summary"]
}
```

## Advanced Usage

### Resume a Session

```bash
# Resume last session
codex exec resume --last

# Resume specific session
codex exec resume <session-id>
```

### Set Approval Policy

```bash
# Deny all tool executions (read-only)
codex exec --approval-policy=deny-all "analyze this code"

# Allow reads only
codex exec --approval-policy=read-only "explain this codebase"
```

### Set Timeout

Codex CLI doesn't have a built-in timeout. Use shell timeout:

```bash
timeout 600 codex exec "your task"
```

## Using the Scripts

The agent-ergonomics kit provides wrapper scripts for common operations.

### Run a Task

```bash
./scripts/codex/run.sh "your task"

# Force CLI mode
CODEX_MODE=cli ./scripts/codex/run.sh "your task"

# Force cloud mode (requires env ID)
CODEX_MODE=cloud ./scripts/codex/run.sh --env ENV_ID "your task"
```

### Queue Tasks

```bash
./scripts/codex/enqueue.sh "task 1"
./scripts/codex/enqueue.sh "task 2"

# Start background worker
./scripts/codex/worker.sh --daemon &
```

### Check Status

```bash
./scripts/codex/status.sh        # Latest job
./scripts/codex/status.sh --all  # All jobs
```

### Monitor with Watchdog

```bash
./scripts/codex/watchdog.sh          # Check once
./scripts/codex/watchdog.sh --daemon # Continuous monitoring
```

### Diagnose Issues

```bash
./scripts/codex/diagnose.sh
```

## Configuration

### Recommended Config

Create `~/.codex/config.toml`:

```toml
# Increase tool output limit to avoid silent truncation
[tools]
max_output_chars = 50000

# Enable useful capabilities
[capabilities]
web_search = true
```

The kit provides a helper:

```bash
./scripts/codex/print_recommended_config.sh
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CODEX_API_KEY` | API authentication (preferred for automation) |
| `OPENAI_API_KEY` | OAuth login (used with `codex login --api-key`) |
| `CODEX_MODEL` | Model to use |
| `CODEX_MODE` | Default mode (mcp/cli/auto) |
| `CODEX_CLOUD_ENV_ID` | Default Codex Cloud environment ID |

## Troubleshooting

### "No output for hours"

Common causes:
1. Missing `--json` flag (no event stream)
2. Tool output limit too low (silent truncation)
3. Network/auth issues with no visible error

Solution:
- Always use `--json` for automation
- Check the kit's diagnose script
- Review logs in `.agent/runs/`

### Authentication Errors

```bash
# Check if key is set
echo $OPENAI_API_KEY | head -c 8

# Test auth
codex whoami
```

### Timeout Issues

- Use shorter tasks
- Split large tasks
- Check network connectivity

### No Events Received

1. Verify Codex is outputting to stdout
2. Check for buffering issues
3. Try direct execution without wrapper scripts

## Best Practices

1. **Always use `--json`** for automated/background runs
2. **Set reasonable timeouts** using shell `timeout` command
3. **Use output schemas** for structured, parseable results
4. **Monitor with watchdog** for long-running tasks
5. **Queue tasks** instead of waiting synchronously
6. **Split large tasks** into smaller, verifiable pieces

## See Also

- [codex_config_recommended.md](codex_config_recommended.md) - Detailed config recommendations
- [codex_postmortem.md](codex_postmortem.md) - Troubleshooting template
- [.agent/codex.md](../.agent/codex.md) - Sub-agent contract
