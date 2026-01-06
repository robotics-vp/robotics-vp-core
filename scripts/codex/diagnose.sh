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

# Test 1: Environment
echo "=== Environment ==="

# Check OPENAI_API_KEY
if [ -n "${OPENAI_API_KEY:-}" ]; then
    # Show first/last few chars
    KEY_LEN=${#OPENAI_API_KEY}
    if [ $KEY_LEN -gt 8 ]; then
        echo -e "${GREEN}OPENAI_API_KEY:${NC} ${OPENAI_API_KEY:0:4}...${OPENAI_API_KEY: -4} (${KEY_LEN} chars)"
    else
        echo -e "${YELLOW}OPENAI_API_KEY:${NC} Set but very short"
        ISSUES+=("API key appears too short")
    fi
else
    echo -e "${RED}OPENAI_API_KEY:${NC} NOT SET"
    ISSUES+=("OPENAI_API_KEY not set")
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

    # Test authentication
    echo ""
    echo "Testing authentication..."
    set +e
    AUTH_RESULT=$(timeout 30 codex whoami 2>&1)
    AUTH_EXIT=$?
    set -e

    if [ $AUTH_EXIT -eq 0 ]; then
        echo -e "${GREEN}Authentication: OK${NC}"
        echo "$AUTH_RESULT"
    elif [ $AUTH_EXIT -eq 124 ]; then
        echo -e "${RED}Authentication: TIMEOUT${NC}"
        ISSUES+=("CLI auth timed out")
    else
        echo -e "${RED}Authentication: FAILED${NC}"
        echo "$AUTH_RESULT"
        ISSUES+=("CLI auth failed")
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
        ISSUES+=("Codex not in MCP config")
    fi
else
    echo -e "${YELLOW}No MCP configuration found${NC}"
    echo "Expected at: ~/.mcp.json or ~/.config/claude/mcp.json"
fi

echo ""

# Test 4: Quick CLI test
echo "=== CLI Execution Test ==="

if command -v codex &> /dev/null && [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "Running quick test (timeout: ${TIMEOUT_SECS}s)..."

    TEST_DIR="$REPO_ROOT/.agent/runs/diagnostic-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$TEST_DIR"

    START_TIME=$(date +%s)
    set +e

    # Run minimal test with JSON output
    timeout "$TIMEOUT_SECS" codex exec --json -o "$TEST_DIR/output.txt" \
        "Say 'Hello from Codex diagnostic test' and list the current directory's top-level files" \
        > "$TEST_DIR/events.jsonl" 2>&1 &
    TEST_PID=$!

    # Monitor for first event
    FIRST_EVENT=false
    for i in $(seq 1 60); do
        if [ -s "$TEST_DIR/events.jsonl" ]; then
            FIRST_EVENT=true
            echo -e "${GREEN}First event received after ${i}s${NC}"
            break
        fi
        sleep 1
    done

    if [ "$FIRST_EVENT" = false ]; then
        echo -e "${RED}No events received within 60s${NC}"
        ISSUES+=("No CLI events within 60s")
        kill $TEST_PID 2>/dev/null || true
    fi

    # Wait for completion
    wait $TEST_PID
    EXIT_CODE=$?
    set -e

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "Duration: ${DURATION}s"
    echo "Exit code: $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}CLI test: PASSED${NC}"

        # Check output
        if [ -f "$TEST_DIR/output.txt" ]; then
            echo ""
            echo "=== Output ==="
            cat "$TEST_DIR/output.txt"
        fi
    elif [ $EXIT_CODE -eq 124 ]; then
        echo -e "${RED}CLI test: TIMEOUT${NC}"
        ISSUES+=("CLI test timed out")
    else
        echo -e "${RED}CLI test: FAILED${NC}"
        ISSUES+=("CLI test failed (exit code: $EXIT_CODE)")

        # Show last events
        if [ -f "$TEST_DIR/events.jsonl" ]; then
            echo ""
            echo "=== Last events ==="
            tail -10 "$TEST_DIR/events.jsonl"
        fi
    fi

    echo ""
    echo "Test logs: $TEST_DIR/"
else
    echo -e "${YELLOW}Skipping (CLI not available or no API key)${NC}"
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
            *"API key"*)
                echo "  - Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key'"
                ;;
            *"not installed"*)
                echo "  - Install Codex: npm install -g @openai/codex"
                ;;
            *"MCP"*)
                echo "  - Configure MCP: see .agent/mcp/servers/codex.md"
                ;;
            *"timeout"*|*"timed out"*)
                echo "  - Check network connectivity to OpenAI API"
                echo "  - Verify API key has Codex access"
                ;;
            *"No CLI events"*)
                echo "  - Possible output buffering issue"
                echo "  - Try running directly: codex exec --json 'hello'"
                ;;
        esac
    done
fi
