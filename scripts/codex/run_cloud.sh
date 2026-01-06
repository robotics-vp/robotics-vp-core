#!/usr/bin/env bash
#
# run_cloud.sh - Run Codex task via cloud execution
#
# Cloud execution runs tasks on OpenAI's infrastructure, enabling:
# - True parallel execution (multiple tasks simultaneously)
# - Persistent environments
# - Longer-running tasks without local resource constraints
#
# Prerequisites:
# - ChatGPT login (OAuth tokens in ~/.codex/auth.json)
# - Environment ID from `codex cloud picker` (Ctrl+O in interactive mode)
#
# Usage:
#   ./scripts/codex/run_cloud.sh --env ENV_ID "task description"
#   ./scripts/codex/run_cloud.sh --env ENV_ID --wait "task description"
#   ./scripts/codex/run_cloud.sh --env ENV_ID --apply "task description"
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
ENV_ID="${CODEX_CLOUD_ENV_ID:-${CODEX_CLOUD_ENV:-}}"
WAIT_FOR_COMPLETION=false
APPLY_PATCH=false
POLL_INTERVAL=30
TIMEOUT=3600  # 1 hour
ATTEMPTS=""
TASK=""
TASK_URL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_ID="$2"
            shift 2
            ;;
        --wait)
            WAIT_FOR_COMPLETION=true
            shift
            ;;
        --apply)
            APPLY_PATCH=true
            WAIT_FOR_COMPLETION=true
            shift
            ;;
        --poll)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --attempts)
            ATTEMPTS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --env ENV_ID [options] \"task description\""
            echo ""
            echo "Run a Codex task via cloud execution."
            echo ""
            echo "Options:"
            echo "  --env ENV_ID      Cloud environment ID (required, or set CODEX_CLOUD_ENV_ID)"
            echo "  --wait            Wait for task completion (non-interactive requires --apply)"
            echo "  --apply           Wait and apply patch (implies --wait)"
            echo "  --poll SECS       Poll interval when waiting (default: 30)"
            echo "  --attempts N      Number of assistant attempts (best-of-N)"
            echo "  --timeout SECS    Max wait time (default: 3600)"
            echo "  --help            Show this help"
            echo ""
            echo "To get an environment ID:"
            echo "  1. Run 'codex' interactively"
            echo "  2. Press Ctrl+O to open cloud picker"
            echo "  3. Copy the environment ID"
            echo ""
            echo "Environment:"
            echo "  CODEX_CLOUD_ENV_ID Default environment ID (preferred)"
            echo "  CODEX_CLOUD_ENV   Default environment ID (legacy)"
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

# Validate inputs
if [ -z "$ENV_ID" ]; then
    echo -e "${RED}Error: Environment ID required${NC}"
    echo ""
    echo "Provide via --env or set CODEX_CLOUD_ENV_ID (or CODEX_CLOUD_ENV)"
    echo ""
    echo "To get an environment ID:"
    echo "  1. Run 'codex' interactively"
    echo "  2. Press Ctrl+O to open cloud picker"
    echo "  3. Copy the environment ID"
    exit 1
fi

if [ -z "$TASK" ]; then
    echo -e "${RED}Error: Task description required${NC}"
    echo "Usage: $0 --env ENV_ID \"task description\""
    exit 1
fi

if [ -n "$ATTEMPTS" ]; then
    if ! [[ "$ATTEMPTS" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: --attempts must be an integer${NC}"
        exit 1
    fi
    if [ "$ATTEMPTS" -lt 1 ] || [ "$ATTEMPTS" -gt 4 ]; then
        echo -e "${RED}Error: --attempts must be between 1 and 4${NC}"
        exit 1
    fi
fi

# Check for codex CLI
if ! command -v codex &> /dev/null; then
    echo -e "${RED}Error: Codex CLI not found${NC}"
    exit 1
fi

# Generate job ID
JOB_ID="cloud-$(date +%Y%m%d-%H%M%S)-$(openssl rand -hex 4 2>/dev/null || echo $$)"
RUN_DIR="$REPO_ROOT/.agent/runs/$JOB_ID"
mkdir -p "$RUN_DIR"

echo -e "${CYAN}=== Codex Cloud Execution ===${NC}"
echo "Job ID: $JOB_ID"
echo "Environment: $ENV_ID"
echo "Task: $TASK"
echo ""

# Write metadata
cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cloud",
  "env_id": "$ENV_ID",
  "task": "$TASK",
  "timeout": $TIMEOUT,
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "submitted"
}
EOF

# Submit to cloud
echo -e "${GREEN}Submitting to cloud...${NC}"
set +e

# Use codex cloud exec
# Note: This captures the task ID for tracking
EXEC_ARGS=(codex cloud exec --env "$ENV_ID")
if [ -n "$ATTEMPTS" ]; then
    EXEC_ARGS+=(--attempts "$ATTEMPTS")
fi
SUBMIT_OUTPUT=$("${EXEC_ARGS[@]}" "$TASK" 2>&1)
SUBMIT_EXIT=$?

echo "$SUBMIT_OUTPUT" > "$RUN_DIR/submit_output.json"

set -e

if [ $SUBMIT_EXIT -ne 0 ]; then
    echo -e "${RED}Submission failed${NC}"
    echo "$SUBMIT_OUTPUT"

    # Update metadata
    cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cloud",
  "env_id": "$ENV_ID",
  "task": "$TASK",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "failed",
  "error": "Submission failed with exit code $SUBMIT_EXIT"
}
EOF
    exit 1
fi

echo -e "${GREEN}Task submitted${NC}"

TASK_URL=$(echo "$SUBMIT_OUTPUT" | grep -oE 'https?://[^ ]*task_[A-Za-z0-9_-]+' | head -1)

# Try to extract task ID from output
TASK_ID=$(echo "$SUBMIT_OUTPUT" | sed -nE 's/.*"task_id"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' | head -1)
if [ -z "$TASK_ID" ]; then
    TASK_ID=$(echo "$SUBMIT_OUTPUT" | sed -nE 's/.*task_id[[:space:]]*[:=][[:space:]]*([A-Za-z0-9_-]+).*/\1/p' | head -1)
fi
if [ -z "$TASK_ID" ]; then
    TASK_ID=$(echo "$SUBMIT_OUTPUT" | sed -nE 's/.*Task ID[[:space:]]*[:=][[:space:]]*([A-Za-z0-9_-]+).*/\1/p' | head -1)
fi
if [ -z "$TASK_ID" ]; then
    TASK_ID=$(echo "$SUBMIT_OUTPUT" | grep -oE 'task_[A-Za-z0-9_-]+' | head -1)
fi

if [ -n "$TASK_ID" ]; then
    echo "Task ID: $TASK_ID"
    if [ -n "$TASK_URL" ]; then
        echo "Task URL: $TASK_URL"
    fi

    # Update metadata with task ID
    cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cloud",
  "env_id": "$ENV_ID",
  "task": "$TASK",
  "task_id": "$TASK_ID",
  "task_url": "$TASK_URL",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "running"
}
EOF
fi

echo "Run directory: $RUN_DIR"

# If not waiting, exit here
if [ "$WAIT_FOR_COMPLETION" = false ]; then
    echo ""
    echo -e "${GREEN}Task submitted to cloud${NC}"
    echo ""
    echo "To check status:"
    echo "  ./scripts/codex/status.sh $JOB_ID"
    echo ""
    echo "To wait for completion:"
    echo "  $0 --env $ENV_ID --wait \"$TASK\""
    echo ""
    echo "To wait and apply patch:"
    echo "  $0 --env $ENV_ID --apply \"$TASK\""
    exit 0
fi

if [ -z "$TASK_ID" ]; then
    echo -e "${YELLOW}Warning: Task ID not found in submission output${NC}"
    if [ -n "$TASK_URL" ]; then
        echo "Task URL: $TASK_URL"
    fi
    echo "Cannot wait/apply without a task ID."
    exit 1
fi

if [ "$APPLY_PATCH" = false ]; then
    echo -e "${YELLOW}Waiting without --apply is not supported in non-interactive mode${NC}"
    echo "Re-run with --apply to apply the cloud diff when ready."
    exit 1
fi

# Wait for completion and apply patch
echo ""
echo "Waiting for completion (poll: ${POLL_INTERVAL}s, timeout: ${TIMEOUT}s)..."

ELAPSED=0
PATCH_APPLIED=false
SLEEP_SECS=$POLL_INTERVAL
MAX_SLEEP=120
while [ $ELAPSED -lt $TIMEOUT ]; do
    set +e
    APPLY_OUTPUT=$(codex apply "$TASK_ID" 2>&1)
    APPLY_EXIT=$?
    set -e
    echo "$APPLY_OUTPUT" > "$RUN_DIR/apply_output.txt"

    if [ $APPLY_EXIT -eq 0 ]; then
        PATCH_APPLIED=true
        echo -e "${GREEN}Patch applied${NC}"
        cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cloud",
  "env_id": "$ENV_ID",
  "task": "$TASK",
  "task_id": "$TASK_ID",
  "task_url": "$TASK_URL",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "completed",
  "patch_applied": $PATCH_APPLIED
}
EOF
        exit 0
    fi

    if echo "$APPLY_OUTPUT" | grep -qiE "still running|not ready|pending|in progress|no diff turn found"; then
        if [ $((ELAPSED + SLEEP_SECS)) -gt $TIMEOUT ]; then
            SLEEP_SECS=$((TIMEOUT - ELAPSED))
        fi
        if [ $SLEEP_SECS -le 0 ]; then
            break
        fi
        echo "  Elapsed: ${ELAPSED}s (not ready; next check in ${SLEEP_SECS}s)..."
        sleep "$SLEEP_SECS"
        ((ELAPSED += SLEEP_SECS))
        if [ $SLEEP_SECS -lt $MAX_SLEEP ]; then
            SLEEP_SECS=$((SLEEP_SECS * 2))
            if [ $SLEEP_SECS -gt $MAX_SLEEP ]; then
                SLEEP_SECS=$MAX_SLEEP
            fi
        fi
        continue
    fi

    echo -e "${RED}Patch apply failed${NC}"
    echo "$APPLY_OUTPUT"
    if [ -n "$TASK_URL" ]; then
        echo "Task URL: $TASK_URL"
    fi
    echo "Next steps:"
    echo "  - Re-run: codex apply $TASK_ID"
    echo "  - Review the task diff in the cloud UI"
    cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cloud",
  "env_id": "$ENV_ID",
  "task": "$TASK",
  "task_id": "$TASK_ID",
  "task_url": "$TASK_URL",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "failed",
  "patch_applied": $PATCH_APPLIED
}
EOF
    exit 1
done

echo -e "${YELLOW}Timeout waiting for completion${NC}"
echo "Task may still be running in cloud"
if [ -n "$TASK_URL" ]; then
    echo "Task URL: $TASK_URL"
fi
echo "Try: codex apply $TASK_ID"
exit 1
