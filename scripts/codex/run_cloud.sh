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
TASK=""

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
            echo "  --wait            Wait for task completion"
            echo "  --apply           Wait and apply patch (implies --wait)"
            echo "  --poll SECS       Poll interval when waiting (default: 30)"
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
SUBMIT_OUTPUT=$(codex cloud exec --env "$ENV_ID" --json "$TASK" 2>&1)
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

# Try to extract task ID from output
TASK_ID=$(echo "$SUBMIT_OUTPUT" | grep -o '"task_id":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")

if [ -n "$TASK_ID" ]; then
    echo "Task ID: $TASK_ID"

    # Update metadata with task ID
    cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cloud",
  "env_id": "$ENV_ID",
  "task": "$TASK",
  "task_id": "$TASK_ID",
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

# Wait for completion
echo ""
echo "Waiting for completion (poll: ${POLL_INTERVAL}s, timeout: ${TIMEOUT}s)..."

ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    # Poll for status
    # Note: Actual implementation depends on Codex cloud API
    # This is a placeholder that checks for patch files

    if [ -f "$RUN_DIR/patch.diff" ]; then
        echo -e "${GREEN}Task completed!${NC}"

        PATCH_APPLIED=false
        if [ "$APPLY_PATCH" = true ]; then
            # Apply patch only when explicitly requested
            echo ""
            echo "Applying patch..."
            cd "$REPO_ROOT"
            if git apply "$RUN_DIR/patch.diff"; then
                PATCH_APPLIED=true
                echo -e "${GREEN}Patch applied${NC}"
            else
                echo -e "${YELLOW}Patch apply failed (may need manual review)${NC}"
            fi
        else
            echo ""
            echo "Patch available at: $RUN_DIR/patch.diff"
            echo "Re-run with --apply to apply it automatically."
        fi

        # Update metadata
        cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cloud",
  "env_id": "$ENV_ID",
  "task": "$TASK",
  "task_id": "$TASK_ID",
  "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "completed",
  "patch_applied": $PATCH_APPLIED
}
EOF
        exit 0
    fi

    echo "  Elapsed: ${ELAPSED}s..."
    sleep "$POLL_INTERVAL"
    ((ELAPSED += POLL_INTERVAL))
done

echo -e "${YELLOW}Timeout waiting for completion${NC}"
echo "Task may still be running in cloud"
exit 1
