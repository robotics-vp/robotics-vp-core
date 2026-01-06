# Agent Setup Guide

This guide explains how to apply the agent-ergonomics kit to a repository.

## Prerequisites

- Bash shell (macOS/Linux, or Git Bash on Windows)
- Git (for version control)
- Node.js 18+ (for Codex CLI)
- OpenAI API key with Codex access (for Codex features)

## Installation

### Step 1: Clone the Kit

```bash
git clone https://github.com/YOUR_ORG/agent-ergonomics.git
cd agent-ergonomics
```

### Step 2: Apply to Target Repo

```bash
./scripts/apply.sh /path/to/your/repo
```

This will:
- Create `AGENTS.md` and `CLAUDE.md`
- Create `.agent/` directory with modular docs
- Install scripts to `scripts/`
- Copy documentation to `docs/`

Use `--force` to overwrite existing files.

### Step 3: Customize AGENTS.md

Edit `AGENTS.md` in your target repo to:
- Add your repo's description
- Update build/test/lint commands
- Add any repo-specific guidelines

### Step 4: Bootstrap

```bash
cd /path/to/your/repo
./scripts/agent/bootstrap.sh
```

This verifies and fixes the setup.

### Step 5: Verify

```bash
./scripts/agent/verify.sh
```

Should show all checks passing.

## MCP Server Installation (Optional)

If you use Claude with MCP:

```bash
./scripts/mcp/install.sh
```

This installs MCP server definitions to `~/.mcp/`.

## Codex Setup

### Install Codex CLI

```bash
npm install -g @openai/codex
# or
brew install codex
```

### Configure Authentication

```bash
export OPENAI_API_KEY="your-key"
codex whoami  # Verify auth
```

### Test Integration

```bash
./scripts/codex/diagnose.sh
```

## CI Integration

Add to your CI workflow:

```yaml
- name: Verify agent ergonomics
  run: ./scripts/agent/verify.sh
```

This prevents drift from the expected structure.

## Why CLAUDE.md is a Shim

Claude Code supports `@` imports to include other files. The shim pattern:

1. Keeps `AGENTS.md` tool-agnostic (other agents can read it directly)
2. Allows Claude-specific features without polluting the main doc
3. Makes updates modular (change one file, not everything)

**Do not add prose to CLAUDE.md.** It should only contain imports.

## Updating the Kit

To update an already-applied kit:

```bash
cd agent-ergonomics
git pull

# Re-apply with force
./scripts/apply.sh /path/to/your/repo --force
```

Review changes to `AGENTS.md` carefully - you may have customizations to preserve.

## Troubleshooting

### verify.sh fails

Run `bootstrap.sh` to auto-fix common issues:

```bash
./scripts/agent/bootstrap.sh
```

### Scripts not executable

```bash
chmod +x scripts/**/*.sh
```

### MCP servers not found

Run the MCP installer:

```bash
./scripts/mcp/install.sh
```

### Codex not working

Run diagnostics:

```bash
./scripts/codex/diagnose.sh
```

See [codex_cli.md](codex_cli.md) for detailed troubleshooting.
