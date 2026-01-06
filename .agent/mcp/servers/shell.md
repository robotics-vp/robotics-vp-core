# Shell MCP Server

MCP server for controlled shell command execution.

## Purpose

Provides controlled shell access for AI agents, enabling:
- Running build commands
- Executing tests
- System operations (with restrictions)

## Installation

### Using npx

```bash
npx -y @anthropic/mcp-server-shell
```

### Configure MCP

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "shell": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-shell"],
      "env": {
        "SHELL_ALLOWED_COMMANDS": "npm,node,git,python,pytest"
      }
    }
  }
}
```

## Configuration

### Allowed commands

Restrict which commands can be run:

```json
{
  "env": {
    "SHELL_ALLOWED_COMMANDS": "npm,node,git,python,make"
  }
}
```

### Working directory

Set the default working directory:

```json
{
  "env": {
    "SHELL_CWD": "/path/to/project"
  }
}
```

### Timeout

Set command timeout (seconds):

```json
{
  "env": {
    "SHELL_TIMEOUT": "300"
  }
}
```

## Security Considerations

- **Allowlist commands** - Only permit necessary commands
- **Avoid sudo** - Don't allow privileged execution
- **Restrict paths** - Use SHELL_CWD to limit scope
- **Review commands** - Audit what's being executed

## Available Tools

| Tool | Description |
|------|-------------|
| `run_command` | Execute a shell command |

## Usage Examples

```
# Run tests
Use the shell tool to run: npm test

# Build project
Use the shell tool to run: npm run build

# Git status
Use the shell tool to run: git status
```

## Troubleshooting

### Command not allowed

- Add command to SHELL_ALLOWED_COMMANDS
- Restart MCP server after config changes

### Timeout

- Increase SHELL_TIMEOUT for long-running commands
- Consider breaking into smaller commands

### Permission denied

- Check file system permissions
- Verify working directory exists and is accessible
