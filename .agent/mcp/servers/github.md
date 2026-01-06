# GitHub MCP Server

MCP server for GitHub API access.

## Purpose

Provides GitHub API access for AI agents, enabling:
- Repository operations (clone, pull, push)
- Issue and PR management
- Code review workflows

## Installation

### Using npx

```bash
npx -y @anthropic/mcp-server-github
```

### Configure MCP

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub personal access token |

### Token permissions needed

- `repo` - Full repo access (for private repos)
- `public_repo` - Public repo access only
- `workflow` - For GitHub Actions
- `read:org` - For organization repos

## Available Tools

| Tool | Description |
|------|-------------|
| `list_repos` | List repositories |
| `get_repo` | Get repository details |
| `list_issues` | List issues in a repo |
| `create_issue` | Create a new issue |
| `list_prs` | List pull requests |
| `create_pr` | Create a pull request |
| `get_file` | Get file contents from repo |

## Usage Examples

```
# List open issues
Use the github tool to list open issues in owner/repo

# Create PR
Use the github tool to create a PR from feature-branch to main

# Get file
Use the github tool to read README.md from owner/repo
```

## Troubleshooting

### Authentication failed

1. Verify GITHUB_TOKEN is set
2. Check token hasn't expired
3. Verify token has required scopes

### Rate limiting

GitHub API has rate limits:
- Authenticated: 5000 requests/hour
- Check remaining: `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit`

### Permission denied

- Verify token has access to the repository
- For org repos, may need org-level permissions
