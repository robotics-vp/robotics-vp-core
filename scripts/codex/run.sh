#!/usr/bin/env bash
#
# run.sh - Router entrypoint for Codex execution
#
# Automatically selects MCP or CLI path based on availability.
#
# Usage:
#   ./scripts/codex/run.sh "task description"
#   CODEX_MODE=cli ./scripts/codex/run.sh "task description"
#   CODEX_MODE=mcp ./scripts/codex/run.sh "task description"
#
# Options:
#   --mode mcp|cli      Force specific mode
#   --schema FILE       Output schema file (for structured output)
#   --timeout SECONDS   Timeout in seconds (default: 600)
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
MODE="${CODEX_MODE:-auto}"
SCHEMA=""
TIMEOUT=600
TASK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --schema)
            SCHEMA="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options] \"task description\""
            echo ""
            echo "Run a Codex task via MCP or CLI."
            echo ""
            echo "Options:"
            echo "  --mode mcp|cli    Force specific execution mode"
            echo "  --schema FILE     Output schema file for structured output"
            echo "  --timeout SECS    Timeout in seconds (default: 600)"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment:"
            echo "  CODEX_MODE        Set default mode (mcp|cli|auto)"
            echo "  OPENAI_API_KEY    Required for Codex authentication"
            exit 0
            ;;
        *)
            if [ -z "$TASK" ]; then
                TASK="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$TASK" ]; then
    echo -e "${RED}Error: Task description required${NC}"
    echo "Usage: $0 \"task description\""
    exit 1
fi

# Function to check if MCP is available
check_mcp() {
    # Check if MCP config exists and codex server is configured
    if [ -f ~/.mcp.json ] || [ -f ~/.config/claude/mcp.json ]; then
        # Basic check - could be enhanced to actually test MCP connection
        return 0
    fi
    return 1
}

# Function to check if CLI is available
check_cli() {
    command -v codex &> /dev/null
}

# Determine mode
if [ "$MODE" = "auto" ]; then
    if check_mcp; then
        MODE="mcp"
        echo -e "${GREEN}Auto-selected: MCP mode${NC}"
    elif check_cli; then
        MODE="cli"
        echo -e "${GREEN}Auto-selected: CLI mode${NC}"
    else
        echo -e "${RED}Error: Neither MCP nor CLI available${NC}"
        echo ""
        echo "Install Codex CLI: npm install -g @openai/codex"
        echo "Or configure MCP: see .agent/mcp.md"
        exit 1
    fi
fi

# Build args array for subscript
EXTRA_ARGS=()
if [ -n "$SCHEMA" ]; then
    EXTRA_ARGS+=("--schema" "$SCHEMA")
fi
EXTRA_ARGS+=("--timeout" "$TIMEOUT")

# Execute via chosen mode
case $MODE in
    mcp)
        exec "$SCRIPT_DIR/run_mcp.sh" "${EXTRA_ARGS[@]}" "$TASK"
        ;;
    cli)
        exec "$SCRIPT_DIR/run_cli.sh" "${EXTRA_ARGS[@]}" "$TASK"
        ;;
    *)
        echo -e "${RED}Error: Invalid mode: $MODE${NC}"
        echo "Valid modes: mcp, cli, auto"
        exit 1
        ;;
esac
