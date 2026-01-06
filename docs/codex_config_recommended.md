# Recommended Codex Configuration

This document provides recommended configuration settings to avoid common issues with Codex, especially silent failures and output truncation.

## Configuration File

Codex configuration lives at `~/.codex/config.toml` (or `~/.codex/config.json`).

## Recommended Settings

### config.toml

```toml
# =============================================================================
# Recommended Codex Configuration
#
# These settings help avoid:
# - Silent output truncation
# - Hidden failures
# - Buffering issues
# =============================================================================

# Model settings
[model]
# Use the latest model for best results
name = "gpt-4"

# Tool output limits
# CRITICAL: Increase these to avoid silent truncation
[tools]
# Maximum characters per tool output (default is often too low)
max_output_chars = 50000

# Maximum file read size
max_file_read_chars = 100000

# Approval policies
[approval]
# Options: "always", "never", "suggest"
# For automation, "never" is safest but limits capabilities
default_policy = "suggest"

# Capabilities
[capabilities]
# Enable web search for research tasks
web_search = true

# Enable file operations
file_operations = true

# Network settings
[network]
# Timeout for API requests (seconds)
request_timeout = 120

# Retry configuration
max_retries = 3
retry_delay = 5

# Logging
[logging]
# Enable verbose logging for debugging
level = "info"

# Write logs to file
file = "~/.codex/codex.log"

# Output settings
[output]
# Disable line buffering for real-time output
line_buffered = false

# Stream mode for continuous output
stream = true
```

### config.json (alternative format)

```json
{
  "model": {
    "name": "gpt-4"
  },
  "tools": {
    "max_output_chars": 50000,
    "max_file_read_chars": 100000
  },
  "approval": {
    "default_policy": "suggest"
  },
  "capabilities": {
    "web_search": true,
    "file_operations": true
  },
  "network": {
    "request_timeout": 120,
    "max_retries": 3,
    "retry_delay": 5
  },
  "logging": {
    "level": "info",
    "file": "~/.codex/codex.log"
  },
  "output": {
    "line_buffered": false,
    "stream": true
  }
}
```

## Key Settings Explained

### max_output_chars

**Problem**: Default limits can silently truncate tool output, causing Codex to miss important information.

**Solution**: Increase to 50000+ characters.

### stream

**Problem**: Without streaming, you don't see progress until the task completes (or fails silently).

**Solution**: Enable streaming for real-time output.

### request_timeout

**Problem**: Default timeout may be too short for complex tasks.

**Solution**: Increase to 120+ seconds.

### logging

**Problem**: Without logs, debugging failures is difficult.

**Solution**: Enable logging to file.

## Installation Script

The kit provides a helper to print recommended config:

```bash
./scripts/codex/print_recommended_config.sh
```

You can pipe to file:

```bash
mkdir -p ~/.codex
./scripts/codex/print_recommended_config.sh > ~/.codex/config.toml
```

**Note**: The kit intentionally does NOT write to your home directory automatically. This is a safety measure - review the config before applying.

## Verifying Configuration

After setting config:

```bash
# Check Codex recognizes config
codex config show

# Run diagnostic
./scripts/codex/diagnose.sh
```

## Environment Variables

Some settings can also be set via environment:

```bash
# In ~/.zshrc or ~/.bashrc
export OPENAI_API_KEY="your-key"
export CODEX_LOG_LEVEL="info"
```

## Troubleshooting

### Config not being read

1. Check file location: `~/.codex/config.toml`
2. Check file syntax (TOML is whitespace-sensitive)
3. Verify with `codex config show`

### Settings not taking effect

Some settings require restart of any running Codex processes:

```bash
# Kill any background Codex
pkill -f codex

# Re-run
./scripts/codex/run.sh "test"
```

### Permission errors

```bash
chmod 600 ~/.codex/config.toml
```

## See Also

- [codex_cli.md](codex_cli.md) - CLI usage guide
- [codex_postmortem.md](codex_postmortem.md) - Troubleshooting template
