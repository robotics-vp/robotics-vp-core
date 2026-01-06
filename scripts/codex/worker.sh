#!/usr/bin/env bash
#
# worker.sh - Process Codex queue sequentially
#
# Usage:
#   ./scripts/codex/worker.sh           # Run until queue empty
#   ./scripts/codex/worker.sh --daemon  # Run continuously
#   nohup ./scripts/codex/worker.sh --daemon &  # Background daemon
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
QUEUE_DIR="$REPO_ROOT/.agent/queue"
QUEUE_FILE="$QUEUE_DIR/codex.jsonl"
LOCK_FILE="$QUEUE_DIR/worker.lock"

DAEMON_MODE=false
POLL_INTERVAL=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon|-d)
            DAEMON_MODE=true
            shift
            ;;
        --poll)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Process Codex queue sequentially."
            echo ""
            echo "Options:"
            echo "  --daemon, -d     Run continuously (poll for new tasks)"
            echo "  --poll SECS      Poll interval in daemon mode (default: 10)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Check for existing worker
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo -e "${YELLOW}Worker already running (PID: $LOCK_PID)${NC}"
        echo "To force restart, remove $LOCK_FILE"
        exit 1
    else
        echo "Stale lock file found, removing"
        rm "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"
trap "rm -f '$LOCK_FILE'" EXIT

echo -e "${CYAN}=== Codex Queue Worker ===${NC}"
echo "Queue file: $QUEUE_FILE"
echo ""

# Function to get next pending task
get_next_task() {
    if [ ! -f "$QUEUE_FILE" ]; then
        return 1
    fi

    # Get first pending task (prioritized: high > normal > low)
    for priority in high normal low; do
        local task=$(grep "\"status\":\"pending\"" "$QUEUE_FILE" 2>/dev/null | grep "\"priority\":\"$priority\"" | head -1)
        if [ -n "$task" ]; then
            echo "$task"
            return 0
        fi
    done

    return 1
}

# Function to mark task as in_progress
mark_in_progress() {
    local task_id="$1"
    if [ -f "$QUEUE_FILE" ]; then
        sed -i.bak "s/\"id\":\"$task_id\",\\(.*\\)\"status\":\"pending\"/\"id\":\"$task_id\",\\1\"status\":\"in_progress\"/" "$QUEUE_FILE" 2>/dev/null || \
        sed -i '' "s/\"id\":\"$task_id\",\\(.*\\)\"status\":\"pending\"/\"id\":\"$task_id\",\\1\"status\":\"in_progress\"/" "$QUEUE_FILE"
    fi
}

# Function to mark task as completed
mark_completed() {
    local task_id="$1"
    local result="$2"
    if [ -f "$QUEUE_FILE" ]; then
        sed -i.bak "s/\"id\":\"$task_id\",\\(.*\\)\"status\":\"in_progress\"/\"id\":\"$task_id\",\\1\"status\":\"$result\"/" "$QUEUE_FILE" 2>/dev/null || \
        sed -i '' "s/\"id\":\"$task_id\",\\(.*\\)\"status\":\"in_progress\"/\"id\":\"$task_id\",\\1\"status\":\"$result\"/" "$QUEUE_FILE"
    fi
}

# Main loop
while true; do
    task_json=$(get_next_task) || {
        if [ "$DAEMON_MODE" = true ]; then
            sleep "$POLL_INTERVAL"
            continue
        else
            echo -e "${GREEN}Queue empty, exiting${NC}"
            break
        fi
    }

    # Parse task
    task_id=$(echo "$task_json" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    task=$(echo "$task_json" | grep -o '"task":"[^"]*"' | cut -d'"' -f4)
    mode=$(echo "$task_json" | grep -o '"mode":"[^"]*"' | cut -d'"' -f4)
    schema=$(echo "$task_json" | grep -o '"schema":"[^"]*"' | cut -d'"' -f4)

    echo "--- $(date) ---"
    echo "Processing: $task_id"
    echo "Task: $task"
    echo ""

    # Mark as in progress
    mark_in_progress "$task_id"

    # Build command
    CMD_ARGS=("--mode" "$mode")
    if [ -n "$schema" ]; then
        CMD_ARGS+=("--schema" "$schema")
    fi
    CMD_ARGS+=("$task")

    # Execute
    set +e
    "$SCRIPT_DIR/run.sh" "${CMD_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    # Mark result
    if [ $EXIT_CODE -eq 0 ]; then
        mark_completed "$task_id" "completed"
        echo -e "${GREEN}Task $task_id completed${NC}"
    else
        mark_completed "$task_id" "failed"
        echo -e "${RED}Task $task_id failed (exit code: $EXIT_CODE)${NC}"
    fi

    echo ""
done
