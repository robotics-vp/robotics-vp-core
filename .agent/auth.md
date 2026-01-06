# Authentication and Environment Variables

## Expected Environment Variables

<!-- CUSTOMIZE: Replace with actual env vars used by this repo -->

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `NODE_ENV` | Runtime environment | No | `development` |
| `API_KEY` | API authentication | Yes (for some features) | `sk-...` |
| `DATABASE_URL` | Database connection | Yes (for DB features) | `postgres://...` |

## Local Setup

### Option 1: .env file (recommended for development)

```bash
# Copy template
cp .env.example .env

# Edit with your values
$EDITOR .env
```

### Option 2: Shell export

```bash
export API_KEY="your-key-here"
export DATABASE_URL="postgres://localhost:5432/mydb"
```

### Option 3: Inline with command

```bash
API_KEY="your-key" npm run dev
```

## CI/CD Secrets

In CI environments, secrets are provided via:

- **GitHub Actions**: Repository secrets or environment secrets
- **GitLab CI**: CI/CD variables
- **CircleCI**: Project environment variables or contexts

### GitHub Actions example

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      API_KEY: ${{ secrets.API_KEY }}
    steps:
      - run: npm test
```

## Never Commit Tokens

### Pre-commit checks

If this repo uses pre-commit hooks, they may scan for secrets:

```bash
# Example patterns that trigger alerts
api_key = "sk-abc123..."
password = "hunter2"
AWS_SECRET_ACCESS_KEY = "..."
```

### What to do if you accidentally commit a secret

1. **Rotate the secret immediately** - Assume it's compromised
2. **Remove from history** (if not pushed):
   ```bash
   git reset --soft HEAD~1
   # Remove secret from file
   git add .
   git commit
   ```
3. **If already pushed**: Rotate secret, then use `git filter-branch` or BFG Repo Cleaner

## Agent Auth Patterns

### For Codex sub-agent

```bash
# Preferred for automation
export CODEX_API_KEY="your-key"

# For OAuth login (interactive or CI bootstrap)
export OPENAI_API_KEY="your-key"
codex login --api-key "$OPENAI_API_KEY"
```

### For Claude

```bash
# Claude Code uses ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY="your-key"
```

### For MCP servers

MCP servers may require their own credentials. See `.agent/mcp.md` for details.

## Verifying Auth

```bash
# Check if env var is set (don't print value!)
[ -n "$API_KEY" ] && echo "API_KEY is set" || echo "API_KEY is NOT set"

# Test API connection
npm run test:api  # or equivalent
```

## Codex Credential Storage

Codex stores OAuth tokens using `cli_auth_credentials_store = "keyring" | "file" | "auto"`.

- `keyring` or `auto` is recommended on macOS (uses the system keychain).
- `file` stores tokens in `~/.codex/auth.json` (treat this like a password; do not commit or share).
