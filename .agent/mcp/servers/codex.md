# Codex MCP Server

MCP server for invoking OpenAI Codex as a sub-agent.

## Purpose

Allows AI agents (like Claude) to delegate tasks to Codex via MCP protocol, enabling:
- Background task execution
- Parallel workloads
- Specialized code generation

## Installation

### Prerequisites

- Node.js 18+
- OpenAI API key with Codex access

### Install Codex CLI

```bash
npm install -g @openai/codex
# or
brew install codex
```

### Configure MCP

Add to your MCP configuration (typically `~/.config/claude/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "codex": {
      "command": "npx",
      "args": ["-y", "codex", "mcp-server"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `CODEX_MODEL` | No | Model to use (default: codex-latest) |
| `CODEX_TIMEOUT` | No | Request timeout in seconds |

## Usage

### Via MCP

Once configured, Codex is available as an MCP tool:

```
Use the codex tool to: "implement a function that calculates fibonacci numbers"
```

### Direct CLI (fallback)

```bash
codex exec "implement fibonacci function"
```

## Log Locations

- MCP server logs: Check your MCP client's log location
- Codex execution logs: `.agent/runs/<job_id>/`

## Troubleshooting

### Server not starting

1. Check OPENAI_API_KEY is set
2. Verify codex is installed: `codex --version`
3. Check MCP client logs for errors

### Timeout issues

Increase timeout in config:

```json
{
  "env": {
    "CODEX_TIMEOUT": "300"
  }
}
```

### Auth failures

- Verify API key is valid
- Check key has Codex access
- Try `codex whoami` to test auth
