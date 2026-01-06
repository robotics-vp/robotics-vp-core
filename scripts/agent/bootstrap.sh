#!/usr/bin/env bash
#
# bootstrap.sh - Bootstrap agent-ergonomics in a repository
#
# Usage:
#   ./scripts/agent/bootstrap.sh [--install-mcp]
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INSTALL_MCP=false

for arg in "$@"; do
    case $arg in
        --install-mcp)
            INSTALL_MCP=true
            ;;
        --help|-h)
            echo "Usage: $0 [--install-mcp]"
            echo ""
            echo "Bootstrap agent-ergonomics in the current repository."
            echo ""
            echo "Options:"
            echo "  --install-mcp    Also run MCP server installation"
            echo "  --help           Show this help message"
            exit 0
            ;;
    esac
done

echo "=== Agent Ergonomics Bootstrap ==="
echo ""
echo "Repository: $REPO_ROOT"
echo ""

# Expected CLAUDE.md shim content
EXPECTED_SHIM='# Claude Code entrypoint
@AGENTS.md
@.agent/skills.md
@.agent/sandbox.md
@.agent/auth.md
@.agent/mcp.md
@.agent/codex.md
@.agent/inference_speed.md'

FIXES=0

# Check/fix CLAUDE.md
echo "Checking CLAUDE.md..."
if [ -f "$REPO_ROOT/CLAUDE.md" ]; then
    ACTUAL=$(cat "$REPO_ROOT/CLAUDE.md")
    if [ "$ACTUAL" = "$EXPECTED_SHIM" ]; then
        echo -e "${GREEN}  CLAUDE.md matches shim template${NC}"
    else
        echo -e "${YELLOW}  CLAUDE.md does not match template, fixing...${NC}"
        echo "$EXPECTED_SHIM" > "$REPO_ROOT/CLAUDE.md"
        ((FIXES++))
    fi
else
    echo -e "${YELLOW}  CLAUDE.md not found, creating...${NC}"
    echo "$EXPECTED_SHIM" > "$REPO_ROOT/CLAUDE.md"
    ((FIXES++))
fi

# Check AGENTS.md
echo "Checking AGENTS.md..."
if [ -f "$REPO_ROOT/AGENTS.md" ]; then
    echo -e "${GREEN}  AGENTS.md exists${NC}"
else
    echo -e "${RED}  AGENTS.md not found${NC}"
    echo "  Run './scripts/apply.sh' from agent-ergonomics kit to create"
fi

# Check .agent/ directory
echo "Checking .agent/ directory..."
AGENT_FILES=(
    "skills.md"
    "sandbox.md"
    "auth.md"
    "mcp.md"
    "codex.md"
    "inference_speed.md"
)

MISSING=()
for file in "${AGENT_FILES[@]}"; do
    if [ ! -f "$REPO_ROOT/.agent/$file" ]; then
        MISSING+=("$file")
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    echo -e "${GREEN}  All .agent/ files present${NC}"
else
    echo -e "${RED}  Missing .agent/ files: ${MISSING[*]}${NC}"
fi

# Ensure directories exist
echo "Ensuring directories..."
mkdir -p "$REPO_ROOT/.agent/queue"
mkdir -p "$REPO_ROOT/.agent/runs"
mkdir -p "$REPO_ROOT/.agent/mcp/servers"
echo -e "${GREEN}  Directories ready${NC}"

# Make scripts executable
echo "Setting script permissions..."
find "$REPO_ROOT/scripts" -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
echo -e "${GREEN}  Scripts are executable${NC}"

# Optional: Install MCP
if [ "$INSTALL_MCP" = true ]; then
    echo ""
    echo "=== Installing MCP Servers ==="
    if [ -f "$REPO_ROOT/scripts/mcp/install.sh" ]; then
        bash "$REPO_ROOT/scripts/mcp/install.sh"
    else
        echo -e "${YELLOW}  MCP install script not found${NC}"
    fi
fi

echo ""
echo "=== Bootstrap Complete ==="
echo ""

if [ $FIXES -gt 0 ]; then
    echo -e "${YELLOW}Made $FIXES fix(es)${NC}"
fi

echo ""
echo "Next steps:"
echo "  1. Review and customize AGENTS.md with your repo's commands"
echo "  2. Run: ./scripts/agent/verify.sh (to verify setup)"
echo "  3. Run: ./scripts/mcp/install.sh (optional)"
echo "  4. Run: ./scripts/codex/diagnose.sh (to test Codex integration)"
echo ""
echo -e "${GREEN}Ready for agents!${NC}"
