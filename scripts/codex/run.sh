#!/usr/bin/env bash
#
# run.sh - Router entrypoint for Codex execution
#
# Automatically selects execution path based on availability:
# - cli: Local CLI execution (default)
# - mcp: MCP server execution
# - cloud: Cloud execution (true parallelism)
#
# Usage:
#   ./scripts/codex/run.sh "task description"
#   CODEX_MODE=cli ./scripts/codex/run.sh "task description"
#   CODEX_MODE=cloud ./scripts/codex/run.sh --env ENV_ID "task description"
#
# Options:
#   --mode cli|mcp|cloud   Force specific mode
#   --env ENV_ID           Cloud environment ID (for cloud mode)
#   --wait                 Wait for cloud completion (cloud mode only)
#   --apply                Wait and apply cloud diff (cloud mode only)
#   --schema FILE          Output schema file (for structured output)
#   --timeout SECONDS      Timeout in seconds (default: 600)
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
ENV_ID="${CODEX_CLOUD_ENV_ID:-${CODEX_CLOUD_ENV:-}}"
TASK=""
CLOUD_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --env)
            ENV_ID="$2"
            shift 2
            ;;
        --wait)
            CLOUD_ARGS+=("--wait")
            shift
            ;;
        --apply)
            CLOUD_ARGS+=("--apply")
            shift
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
            echo "Run a Codex task via CLI, MCP, or Cloud."
            echo ""
            echo "Options:"
            echo "  --mode MODE       Execution mode: cli, mcp, cloud, auto (default: auto)"
            echo "  --env ENV_ID      Cloud environment ID (required for cloud mode)"
            echo "  --wait            Wait for cloud completion (cloud mode only)"
            echo "  --apply           Wait and apply cloud diff (cloud mode only)"
            echo "  --schema FILE     Output schema file for structured output"
            echo "  --timeout SECS    Timeout in seconds (default: 600)"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment:"
            echo "  CODEX_MODE        Default mode (cli|mcp|cloud|auto)"
            echo "  CODEX_CLOUD_ENV_ID Default cloud environment ID (preferred)"
            echo "  CODEX_CLOUD_ENV   Default cloud environment ID (legacy)"
            echo "  CODEX_API_KEY     API key for automation (preferred)"
            echo "  OPENAI_API_KEY    API key (legacy)"
            echo ""
            echo "Modes:"
            echo "  cli   - Local CLI execution (single task, blocking)"
            echo "  mcp   - MCP server execution (for Claude integration)"
            echo "  cloud - Cloud execution (true parallelism, non-blocking)"
            echo "  auto  - Auto-select: cli > mcp (cloud requires explicit --mode)"
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

if [ "$MODE" = "auto" ] && [ ${#CLOUD_ARGS[@]} -gt 0 ]; then
    MODE="cloud"
fi

# Function to check if MCP is available
check_mcp() {
    if [ -f ~/.mcp.json ] || [ -f ~/.config/claude/mcp.json ]; then
        return 0
    fi
    return 1
}

# Function to check if CLI is available
check_cli() {
    command -v codex &> /dev/null
}

# Check for [cloud] tag in task
if [[ "$TASK" == *"[cloud]"* ]] && [ "$MODE" = "auto" ]; then
    MODE="cloud"
    TASK="${TASK//\[cloud\]/}"  # Remove tag
    TASK="${TASK## }"  # Trim leading space
fi

# Determine mode
if [ "$MODE" = "auto" ]; then
    # Auto-select: prefer CLI for local work
    if check_cli; then
        MODE="cli"
        echo -e "${GREEN}Auto-selected: CLI mode${NC}"
    elif check_mcp; then
        MODE="mcp"
        echo -e "${GREEN}Auto-selected: MCP mode${NC}"
    else
        echo -e "${RED}Error: Neither CLI nor MCP available${NC}"
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
        if [ ${#CLOUD_ARGS[@]} -gt 0 ]; then
            echo -e "${RED}Error: --wait/--apply only supported for cloud mode${NC}"
            exit 1
        fi
        exec "$SCRIPT_DIR/run_cli.sh" "${EXTRA_ARGS[@]}" "$TASK"
        ;;
    cloud)
        if [ -z "$ENV_ID" ]; then
            echo -e "${RED}Error: Cloud mode requires --env ENV_ID${NC}"
            echo ""
            echo "To get an environment ID:"
            echo "  1. Run 'codex' interactively"
            echo "  2. Press Ctrl+O to open cloud picker"
            echo "  3. Copy the environment ID"
            echo ""
            echo "Or set CODEX_CLOUD_ENV_ID (or CODEX_CLOUD_ENV) environment variable"
            exit 1
        fi
        if [ ${#CLOUD_ARGS[@]} -gt 0 ]; then
            exec "$SCRIPT_DIR/run_cloud.sh" --env "$ENV_ID" "${CLOUD_ARGS[@]}" "${EXTRA_ARGS[@]}" "$TASK"
        else
            exec "$SCRIPT_DIR/run_cloud.sh" --env "$ENV_ID" "${EXTRA_ARGS[@]}" "$TASK"
        fi
        ;;
    *)
        echo -e "${RED}Error: Invalid mode: $MODE${NC}"
        echo "Valid modes: cli, mcp, cloud, auto"
        exit 1
        ;;
esac
