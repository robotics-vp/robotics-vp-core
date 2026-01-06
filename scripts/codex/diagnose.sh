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

JOB_ID="diagnostic-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$REPO_ROOT/.agent/runs/$JOB_ID"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/meta.json" << EOF_META
{
  "job_id": "$JOB_ID",
  "mode": "diagnostic",
  "timeout_seconds": $TIMEOUT_SECS,
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF_META

echo -e "${CYAN}=== Codex Diagnostic ===${NC}"
echo "Timeout: ${TIMEOUT_SECS}s per test"
echo "Run directory: $RUN_DIR"
echo ""

ISSUES=()
HAS_AUTH=false
AUTH_METHOD=""
LAST_TEST_DIR=""
LAST_TEST_ISSUE=""

CONFIG_FILE="$HOME/.codex/config.toml"
CRED_STORE=""
REASONING_EFFORT=""
XHIGH_CONFIGURED=false

# TOML reader (best-effort, single-line key/value)
toml_get() {
    local key="$1"
    local file="$2"
    local line=""

    if [ ! -f "$file" ]; then
        return 0
    fi

    line=$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" | tail -n1 || true)
    if [ -z "$line" ]; then
        return 0
    fi

    line=${line#*=}
    line=$(printf '%s' "$line" | sed -E 's/^[[:space:]]*//; s/[[:space:]]*(#.*)?$//')
    line=$(printf '%s' "$line" | sed -E "s/^['\"]//; s/['\"]$//")

    printf '%s' "$line"
}

if [ -f "$CONFIG_FILE" ]; then
    CRED_STORE="$(toml_get "cli_auth_credentials_store" "$CONFIG_FILE")"
    REASONING_EFFORT="$(toml_get "model_reasoning_effort" "$CONFIG_FILE")"
fi

if [ -n "$REASONING_EFFORT" ]; then
    echo -e "${GREEN}Config reasoning effort:${NC} $REASONING_EFFORT"
    if [ "$(printf '%s' "$REASONING_EFFORT" | tr '[:upper:]' '[:lower:]')" = "xhigh" ]; then
        XHIGH_CONFIGURED=true
        echo -e "${YELLOW}Warning:${NC} model_reasoning_effort is xhigh (allowed, but model-dependent)."
    fi
    echo ""
fi

run_cli_test() {
    local label="$1"
    local effort_override="$2"
    local env_name="$3"
    local env_value="$4"

    local test_dir="$RUN_DIR/cli-test-$label"
    local events_file="$test_dir/events.jsonl"
    local cmd_desc="codex exec --json"
    local -a cmd=("codex" "exec" "--json")

    LAST_TEST_DIR="$test_dir"
    LAST_TEST_ISSUE=""
    mkdir -p "$test_dir"

    if [ -n "$effort_override" ]; then
        cmd_desc="$cmd_desc -c model_reasoning_effort=$effort_override"
        cmd+=("-c" "model_reasoning_effort=$effort_override")
    fi
    cmd_desc="$cmd_desc -"
    cmd+=("-")

    echo "Running quick test ($label)..."
    echo "Command: $cmd_desc"
    echo "Output: $events_file"

    START_TIME=$(date +%s)
    set +e

    if [ -n "$env_name" ]; then
        echo "say hello" | env "$env_name=$env_value" "${cmd[@]}" > "$events_file" 2>&1 &
    else
        echo "say hello" | "${cmd[@]}" > "$events_file" 2>&1 &
    fi
    TEST_PID=$!

    ELAPSED=0
    FIRST_EVENT=false
    while kill -0 $TEST_PID 2>/dev/null && [ $ELAPSED -lt $TIMEOUT_SECS ]; do
        if [ -s "$events_file" ] && [ "$FIRST_EVENT" = false ]; then
            FIRST_EVENT=true
            echo -e "${GREEN}First event received after ${ELAPSED}s${NC}"
        fi
        if [ $ELAPSED -gt 0 ] && [ $((ELAPSED % 10)) -eq 0 ]; then
            echo "  ... ${ELAPSED}s elapsed"
        fi
        sleep 1
        ((ELAPSED++))
    done

    if kill -0 $TEST_PID 2>/dev/null; then
        echo -e "${YELLOW}Timeout after ${TIMEOUT_SECS}s, killing...${NC}"
        kill $TEST_PID 2>/dev/null || true
        sleep 1
        kill -9 $TEST_PID 2>/dev/null || true
        LAST_TEST_ISSUE="CLI test timed out"
    fi

    wait $TEST_PID 2>/dev/null
    set -e

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "Duration: ${DURATION}s"

    if [ -s "$events_file" ]; then
        if grep -q '"turn.completed"' "$events_file" 2>/dev/null; then
            echo -e "${GREEN}CLI test: PASSED${NC}"
            echo ""
            echo "Events received:"
            grep -o '"type":"[^"]*"' "$events_file" | sort | uniq -c | head -10
            return 0
        fi

        echo -e "${YELLOW}CLI test: Partial (no turn.completed)${NC}"
        echo ""
        echo "Last events:"
        tail -5 "$events_file"
        if [ -z "$LAST_TEST_ISSUE" ]; then
            LAST_TEST_ISSUE="CLI test incomplete (no turn.completed)"
        fi
        return 1
    fi

    echo -e "${RED}CLI test: No output${NC}"
    if [ -z "$LAST_TEST_ISSUE" ]; then
        LAST_TEST_ISSUE="CLI produced no output"
    fi
    return 1
}

echo "=== Authentication ==="

if [ -n "$CRED_STORE" ]; then
    echo -e "${GREEN}Credential store:${NC} $CRED_STORE"
    if [ "$CRED_STORE" = "keyring" ] || [ "$CRED_STORE" = "auto" ]; then
        echo "OAuth tokens may be stored in the system keychain."
    fi
fi

# OAuth tokens in ~/.codex/auth.json (file store)
if [ -f ~/.codex/auth.json ]; then
    if grep -q '"access_token"' ~/.codex/auth.json 2>/dev/null; then
        echo -e "${GREEN}OAuth tokens:${NC} Found in ~/.codex/auth.json"
    else
        echo -e "${YELLOW}OAuth tokens:${NC} auth.json exists but no access_token"
    fi
else
    echo -e "${YELLOW}OAuth tokens:${NC} No ~/.codex/auth.json"
fi

# CODEX_API_KEY
if [ -n "${CODEX_API_KEY:-}" ]; then
    KEY_LEN=${#CODEX_API_KEY}
    if [ $KEY_LEN -gt 8 ]; then
        echo -e "${GREEN}CODEX_API_KEY:${NC} ${CODEX_API_KEY:0:4}...${CODEX_API_KEY: -4} (${KEY_LEN} chars)"
    else
        echo -e "${YELLOW}CODEX_API_KEY:${NC} Set but very short"
        ISSUES+=("CODEX_API_KEY appears too short")
    fi
else
    echo -e "${YELLOW}CODEX_API_KEY:${NC} Not set"
fi

# OPENAI_API_KEY
if [ -n "${OPENAI_API_KEY:-}" ]; then
    KEY_LEN=${#OPENAI_API_KEY}
    if [ $KEY_LEN -gt 8 ]; then
        echo -e "${GREEN}OPENAI_API_KEY:${NC} ${OPENAI_API_KEY:0:4}...${OPENAI_API_KEY: -4} (${KEY_LEN} chars)"
    else
        echo -e "${YELLOW}OPENAI_API_KEY:${NC} Set but very short"
        ISSUES+=("OPENAI_API_KEY appears too short")
    fi
else
    echo -e "${YELLOW}OPENAI_API_KEY:${NC} Not set"
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

# Test 4: Quick CLI test with auth precedence
echo "=== CLI Execution Test ==="

if command -v codex &> /dev/null; then
    FIRST_ATTEMPT=true
    TEST_SUCCESS=false

    # OAuth (default) attempt
    echo "Testing OAuth credentials (default codex auth)..."
    if run_cli_test "oauth" "" "" ""; then
        HAS_AUTH=true
        AUTH_METHOD="oauth"
        TEST_SUCCESS=true
    else
        if [ "$FIRST_ATTEMPT" = true ] && [ "$XHIGH_CONFIGURED" = true ]; then
            echo -e "${YELLOW}Retrying with model_reasoning_effort=high due to xhigh config...${NC}"
            if run_cli_test "oauth-fallback" "high" "" ""; then
                HAS_AUTH=true
                AUTH_METHOD="oauth"
                TEST_SUCCESS=true
            fi
        fi
    fi
    FIRST_ATTEMPT=false

    # CODEX_API_KEY attempt
    if [ "$TEST_SUCCESS" = false ] && [ -n "${CODEX_API_KEY:-}" ]; then
        echo "Testing CODEX_API_KEY..."
        if run_cli_test "codex-api-key" "" "CODEX_API_KEY" "$CODEX_API_KEY"; then
            HAS_AUTH=true
            AUTH_METHOD="codex_api_key"
            TEST_SUCCESS=true
        fi
    fi

    # OPENAI_API_KEY attempt via login
    if [ "$TEST_SUCCESS" = false ] && [ -n "${OPENAI_API_KEY:-}" ]; then
        echo "Attempting codex login with OPENAI_API_KEY..."
        LOGIN_LOG="$RUN_DIR/openai_login.log"
        set +e
        codex login --api-key "$OPENAI_API_KEY" > "$LOGIN_LOG" 2>&1
        LOGIN_EXIT=$?
        set -e
        if [ $LOGIN_EXIT -ne 0 ]; then
            echo -e "${YELLOW}codex login --api-key failed (see $LOGIN_LOG)${NC}"
            echo "You may need to run 'codex login' interactively."
        else
            echo -e "${GREEN}codex login succeeded${NC}"
        fi

        echo "Testing after OPENAI_API_KEY login..."
        if run_cli_test "openai-api-key" "" "" ""; then
            HAS_AUTH=true
            AUTH_METHOD="openai_api_key"
            TEST_SUCCESS=true
        fi
    fi

    if [ "$TEST_SUCCESS" = true ]; then
        echo ""
        echo -e "${GREEN}Auth method in use:${NC} $AUTH_METHOD"
    else
        echo -e "${YELLOW}CLI test failed for all auth methods${NC}"
        ISSUES+=("No usable authentication found (OAuth, CODEX_API_KEY, or OPENAI_API_KEY)")
        if [ -n "$LAST_TEST_ISSUE" ]; then
            ISSUES+=("$LAST_TEST_ISSUE")
        fi
    fi

    echo ""
    if [ -n "$LAST_TEST_DIR" ]; then
        echo "Test logs: $LAST_TEST_DIR/"
    fi
else
    echo -e "${YELLOW}Skipping (CLI not available)${NC}"
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
            *"No usable authentication"*|*"authentication"*)
                echo "  - Run 'codex login' interactively to authenticate"
                echo "  - Or set CODEX_API_KEY for automation: export CODEX_API_KEY='your-key'"
                echo "  - If you only have OPENAI_API_KEY: codex login --api-key \"\\$OPENAI_API_KEY\""
                ;;
            *"API key"*|*"CODEX_API_KEY"*|*"OPENAI_API_KEY"*)
                echo "  - Check that your API key is correct and not truncated"
                ;;
            *"not installed"*)
                echo "  - Install Codex: npm install -g @openai/codex"
                ;;
            *"timeout"*|*"timed out"*)
                echo "  - Check network connectivity to OpenAI API"
                echo "  - Try a simpler prompt"
                ;;
            *"No output"*|*"no output"*|*"incomplete"*)
                echo "  - Check ~/.codex/config.toml for errors"
                echo "  - Try: echo 'hello' | codex exec --json -"
                ;;
        esac
    done
fi
