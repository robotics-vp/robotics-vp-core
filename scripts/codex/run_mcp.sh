#!/usr/bin/env bash
#
# run_mcp.sh - Run Codex task via MCP
#
# This is a stub/wrapper for MCP-based Codex invocation.
# MCP execution is typically handled by the MCP client (Claude, etc.),
# so this script serves as documentation and fallback.
#
# Usage:
#   ./scripts/codex/run_mcp.sh "task description"
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
SCHEMA=""
TIMEOUT=600
TASK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Run a Codex task via MCP."
            echo ""
            echo "Note: MCP execution is typically handled by the MCP client."
            echo "This script checks configuration and provides guidance."
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

echo -e "${CYAN}=== Codex MCP Execution ===${NC}"
echo ""

# Check for MCP configuration
MCP_CONFIG=""
if [ -f ~/.mcp.json ]; then
    MCP_CONFIG=~/.mcp.json
elif [ -f ~/.config/claude/mcp.json ]; then
    MCP_CONFIG=~/.config/claude/mcp.json
fi

if [ -z "$MCP_CONFIG" ]; then
    echo -e "${RED}Error: MCP configuration not found${NC}"
    echo ""
    echo "MCP configuration should be at:"
    echo "  ~/.mcp.json"
    echo "  ~/.config/claude/mcp.json"
    echo ""
    echo "Example Codex MCP server configuration:"
    echo ""
    cat << 'EOF'
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
EOF
    echo ""
    echo "See: .agent/mcp/servers/codex.md"
    echo ""
    echo "Alternatively, use CLI mode:"
    echo "  CODEX_MODE=cli ./scripts/codex/run.sh \"$TASK\""
    exit 1
fi

echo "MCP config found: $MCP_CONFIG"
echo ""

# Check if codex is configured in MCP
if grep -q "codex" "$MCP_CONFIG" 2>/dev/null; then
    echo -e "${GREEN}Codex server is configured in MCP${NC}"
else
    echo -e "${YELLOW}Warning: Codex server may not be configured in MCP${NC}"
    echo "Check $MCP_CONFIG for codex configuration"
fi

echo ""
echo -e "${YELLOW}MCP execution note:${NC}"
echo ""
echo "MCP execution is typically handled by the MCP client (e.g., Claude)."
echo "The client will invoke the Codex MCP server directly."
echo ""
echo "To run this task:"
echo ""
echo "1. Ensure your MCP client is running with Codex configured"
echo "2. Ask the client to use the codex tool with:"
echo "   \"$TASK\""
echo ""
echo "Or use CLI mode instead:"
echo "  ./scripts/codex/run_cli.sh \"$TASK\""
echo ""

# If task was provided, at least log it
if [ -n "$TASK" ]; then
    JOB_ID="$(date +%Y%m%d-%H%M%S)-mcp-$(openssl rand -hex 4 2>/dev/null || echo $$)"
    RUN_DIR="$REPO_ROOT/.agent/runs/$JOB_ID"
    mkdir -p "$RUN_DIR"

    cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "mcp",
  "task": "$TASK",
  "schema": "$SCHEMA",
  "timeout": $TIMEOUT,
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "pending_mcp_execution",
  "note": "MCP execution must be triggered by MCP client"
}
EOF

    echo "Created job stub: $JOB_ID"
    echo "Run directory: $RUN_DIR"
    echo ""
    echo "Once MCP execution completes, update $RUN_DIR/meta.json"
fi
