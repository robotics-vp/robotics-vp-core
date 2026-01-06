# MCP (Model Context Protocol) Configuration

## Overview

MCP allows AI agents to connect to external tools and data sources. This repo follows the mcp-local-spec approach for organizing MCP server definitions.

## MCP Local Spec

MCP server definitions are stored in:

- **Repo-local**: `.agent/mcp/servers/*.md` - Servers specific to this project
- **User-global**: `~/.mcp/*.md` - User's personal server configurations

### Server definition format

Each `.md` file in the servers directory describes one MCP server:

```markdown
# Server Name

Brief description of what this server does.

## Installation

How to install/configure this server.

## Configuration

Required environment variables or settings.

## Usage

Example prompts or use cases.
```

## Installing Repo MCP Servers

Run the installer to set up MCP servers from this repo:

```bash
./scripts/mcp/install.sh
```

Options:
- `--force` - Overwrite existing configurations
- `--copy` - Copy instead of symlink (default on Windows)
- `--dry-run` - Show what would be done

## Available Servers

<!-- CUSTOMIZE: List actual MCP servers in this repo -->

| Server | Purpose | Location |
|--------|---------|----------|
| codex | Codex sub-agent integration | `.agent/mcp/servers/codex.md` |
| filesystem | Local file access | `.agent/mcp/servers/filesystem.md` |

## Troubleshooting

### Server not connecting

1. Check the server is installed:
   ```bash
   ls ~/.mcp/
   ```

2. Verify environment variables are set:
   ```bash
   # Check without printing values
   env | grep -E "^(OPENAI|ANTHROPIC)_" | cut -d= -f1
   ```

3. Check server logs (location varies by server)

### Permission errors

```bash
# Ensure install script is executable
chmod +x ./scripts/mcp/install.sh

# Check ~/.mcp permissions
ls -la ~/.mcp/
```

### Symlink issues (Windows)

Windows may not support symlinks. Use copy mode:

```bash
./scripts/mcp/install.sh --copy
```

### Server definition not found

1. Check the file exists in `.agent/mcp/servers/`
2. Run `./scripts/mcp/install.sh` to reinstall
3. Check for typos in filename

## Creating New Server Definitions

1. Create a new `.md` file in `.agent/mcp/servers/`:
   ```bash
   touch .agent/mcp/servers/my-server.md
   ```

2. Follow the template format (see existing servers)

3. Run the installer:
   ```bash
   ./scripts/mcp/install.sh
   ```

## Security Notes

- MCP servers may have access to files and network
- Review server capabilities before enabling
- Don't grant unnecessary permissions
- Audit servers periodically
