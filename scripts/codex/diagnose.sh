#!/usr/bin/env bash
#
# diagnose.sh - Diagnose Codex configuration and connectivity
#
# Runs quick tests in both CLI and MCP modes to identify issues.
#
# Usage:
#   ./scripts/codex/diagnose.sh
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

TIMEOUT_SECS=120

echo -e "${CYAN}=== Codex Diagnostic ===${NC}"
echo "Timeout: ${TIMEOUT_SECS}s per test"
echo ""

ISSUES=()
HAS_AUTH=false

# Test 1: Environment / Auth
echo "=== Authentication ==="

# Check OPENAI_API_KEY
if [ -n "${OPENAI_API_KEY:-}" ]; then
    KEY_LEN=${#OPENAI_API_KEY}
    if [ $KEY_LEN -gt 8 ]; then
        echo -e "${GREEN}OPENAI_API_KEY:${NC} ${OPENAI_API_KEY:0:4}...${OPENAI_API_KEY: -4} (${KEY_LEN} chars)"
        HAS_AUTH=true
    else
        echo -e "${YELLOW}OPENAI_API_KEY:${NC} Set but very short"
        ISSUES+=("API key appears too short")
    fi
else
    echo -e "${YELLOW}OPENAI_API_KEY:${NC} Not set (checking OAuth...)"
fi

# Check OAuth tokens in ~/.codex/auth.json
if [ -f ~/.codex/auth.json ]; then
    if grep -q '"access_token"' ~/.codex/auth.json 2>/dev/null; then
        echo -e "${GREEN}OAuth tokens:${NC} Found in ~/.codex/auth.json"
        HAS_AUTH=true
    else
        echo -e "${YELLOW}OAuth tokens:${NC} auth.json exists but no access_token"
    fi
else
    echo -e "${YELLOW}OAuth tokens:${NC} No ~/.codex/auth.json"
fi

if [ "$HAS_AUTH" = false ]; then
    ISSUES+=("No authentication found (need OPENAI_API_KEY or OAuth login)")
fi

echo ""

# Test 2: CLI availability
echo "=== CLI Check ==="

if command -v codex &> /dev/null; then
    CODEX_PATH=$(which codex)
    echo -e "${GREEN}Codex CLI found:${NC} $CODEX_PATH"

    # Get version
    CODEX_VERSION=$(codex --version 2>/dev/null || echo "unknown")
    echo "Version: $CODEX_VERSION"

    # Check config
    if [ -f ~/.codex/config.toml ]; then
        echo -e "${GREEN}Config:${NC} ~/.codex/config.toml exists"
    fi
else
    echo -e "${RED}Codex CLI not found${NC}"
    echo ""
    echo "Install with:"
    echo "  npm install -g @openai/codex"
    echo "  # or"
    echo "  brew install codex"
    ISSUES+=("Codex CLI not installed")
fi

echo ""

# Test 3: MCP configuration
echo "=== MCP Check ==="

MCP_CONFIG=""
if [ -f ~/.mcp.json ]; then
    MCP_CONFIG=~/.mcp.json
elif [ -f ~/.config/claude/mcp.json ]; then
    MCP_CONFIG=~/.config/claude/mcp.json
fi

if [ -n "$MCP_CONFIG" ]; then
    echo -e "${GREEN}MCP config found:${NC} $MCP_CONFIG"

    # Check for codex in config
    if grep -q "codex" "$MCP_CONFIG" 2>/dev/null; then
        echo -e "${GREEN}Codex server configured${NC}"
    else
        echo -e "${YELLOW}Codex server not found in MCP config${NC}"
        # Not a blocking issue if CLI works
    fi
else
    echo -e "${YELLOW}No MCP configuration found${NC}"
    echo "Expected at: ~/.mcp.json or ~/.config/claude/mcp.json"
fi

echo ""

# Test 4: Quick CLI test
echo "=== CLI Execution Test ==="

if command -v codex &> /dev/null && [ "$HAS_AUTH" = true ]; then
    echo "Running quick test..."

    TEST_DIR="$REPO_ROOT/.agent/runs/diagnostic-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$TEST_DIR"

    START_TIME=$(date +%s)
    set +e

    # Run minimal test with JSON output (background with manual timeout)
    echo "say hello" | codex exec --json - > "$TEST_DIR/events.jsonl" 2>&1 &
    TEST_PID=$!

    # Monitor for completion or timeout
    ELAPSED=0
    FIRST_EVENT=false
    while kill -0 $TEST_PID 2>/dev/null && [ $ELAPSED -lt $TIMEOUT_SECS ]; do
        if [ -s "$TEST_DIR/events.jsonl" ] && [ "$FIRST_EVENT" = false ]; then
            FIRST_EVENT=true
            echo -e "${GREEN}First event received after ${ELAPSED}s${NC}"
        fi
        sleep 1
        ((ELAPSED++))
    done

    # Check if still running (timeout)
    if kill -0 $TEST_PID 2>/dev/null; then
        echo -e "${YELLOW}Timeout after ${TIMEOUT_SECS}s, killing...${NC}"
        kill $TEST_PID 2>/dev/null || true
        sleep 1
        kill -9 $TEST_PID 2>/dev/null || true
        ISSUES+=("CLI test timed out")
    fi

    wait $TEST_PID 2>/dev/null
    EXIT_CODE=$?
    set -e

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "Duration: ${DURATION}s"

    # Check for successful completion
    if [ -s "$TEST_DIR/events.jsonl" ]; then
        if grep -q '"turn.completed"' "$TEST_DIR/events.jsonl" 2>/dev/null; then
            echo -e "${GREEN}CLI test: PASSED${NC}"

            # Show events summary
            echo ""
            echo "Events received:"
            grep -o '"type":"[^"]*"' "$TEST_DIR/events.jsonl" | sort | uniq -c | head -10
        else
            echo -e "${YELLOW}CLI test: Partial (no turn.completed)${NC}"
            echo ""
            echo "Last events:"
            tail -5 "$TEST_DIR/events.jsonl"
        fi
    else
        echo -e "${RED}CLI test: No output${NC}"
        ISSUES+=("CLI produced no output")
    fi

    echo ""
    echo "Test logs: $TEST_DIR/"
else
    if [ "$HAS_AUTH" = false ]; then
        echo -e "${YELLOW}Skipping (no authentication)${NC}"
    else
        echo -e "${YELLOW}Skipping (CLI not available)${NC}"
    fi
fi

echo ""

# Test 5: Directory structure
echo "=== Directory Structure ==="

if [ -d "$REPO_ROOT/.agent" ]; then
    echo -e "${GREEN}.agent/ directory exists${NC}"
else
    echo -e "${YELLOW}.agent/ directory missing${NC}"
    ISSUES+=(".agent/ directory missing")
fi

if [ -d "$REPO_ROOT/.agent/runs" ]; then
    echo -e "${GREEN}.agent/runs/ directory exists${NC}"
    RUN_COUNT=$(ls -1 "$REPO_ROOT/.agent/runs" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Previous runs: $RUN_COUNT"
else
    echo -e "${YELLOW}.agent/runs/ directory missing${NC}"
fi

if [ -d "$REPO_ROOT/.agent/queue" ]; then
    echo -e "${GREEN}.agent/queue/ directory exists${NC}"
    if [ -f "$REPO_ROOT/.agent/queue/codex.jsonl" ]; then
        QUEUE_SIZE=$(wc -l < "$REPO_ROOT/.agent/queue/codex.jsonl" | tr -d ' ')
        echo "  Queue entries: $QUEUE_SIZE"
    fi
else
    echo -e "${YELLOW}.agent/queue/ directory missing${NC}"
fi

echo ""

# Summary
echo "=== Summary ==="

if [ ${#ISSUES[@]} -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "Codex is ready to use."
    echo ""
    echo "Quick start:"
    echo "  ./scripts/codex/run.sh \"your task here\""
    echo "  ./scripts/codex/enqueue.sh \"queue a task\""
else
    echo -e "${RED}Found ${#ISSUES[@]} issue(s):${NC}"
    echo ""
    for issue in "${ISSUES[@]}"; do
        echo -e "  ${RED}-${NC} $issue"
    done
    echo ""
    echo "Recommendations:"

    for issue in "${ISSUES[@]}"; do
        case "$issue" in
            *"API key"*|*"authentication"*)
                echo "  - Run 'codex' interactively to login via browser"
                echo "  - Or set OPENAI_API_KEY: export OPENAI_API_KEY='your-key'"
                ;;
            *"not installed"*)
                echo "  - Install Codex: npm install -g @openai/codex"
                ;;
            *"timeout"*|*"timed out"*)
                echo "  - Check network connectivity to OpenAI API"
                echo "  - Try a simpler prompt"
                ;;
            *"No output"*|*"no output"*)
                echo "  - Check ~/.codex/config.toml for errors"
                echo "  - Try: echo 'hello' | codex exec --json -"
                ;;
        esac
    done
fi
