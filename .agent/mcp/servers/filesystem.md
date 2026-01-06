# Filesystem MCP Server

MCP server for local file system access.

## Purpose

Provides controlled access to the local filesystem for AI agents, enabling:
- Reading files and directories
- Writing files (with restrictions)
- Searching file contents

## Installation

### Using npx (recommended)

```bash
npx -y @anthropic/mcp-server-filesystem
```

### Configure MCP

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/allowed/directory"],
      "env": {}
    }
  }
}
```

## Configuration

### Allowed directories

The server only accesses directories explicitly listed in args:

```json
{
  "args": ["-y", "@anthropic/mcp-server-filesystem",
    "/Users/me/projects",
    "/Users/me/documents"
  ]
}
```

### Security note

- Only grant access to directories that need to be accessed
- Don't grant access to home directory root
- Avoid granting access to system directories

## Available Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read contents of a file |
| `write_file` | Write content to a file |
| `list_directory` | List contents of a directory |
| `search_files` | Search for files by pattern |

## Usage Examples

```
# Read a file
Use the filesystem tool to read package.json

# List directory
Use the filesystem tool to list files in src/

# Search
Use the filesystem tool to find all .ts files
```

## Troubleshooting

### Permission denied

- Verify directory is in allowed list
- Check file system permissions
- Ensure MCP config is correct

### File not found

- Use absolute paths
- Verify file exists
- Check for typos in path
