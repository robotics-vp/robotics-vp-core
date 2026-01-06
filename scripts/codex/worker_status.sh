#!/usr/bin/env bash
#
# worker_status.sh - Show queue and worker status
#
# Usage:
#   ./scripts/codex/worker_status.sh
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

echo -e "${CYAN}=== Codex Queue Status ===${NC}"
echo ""

# Check worker
echo "=== Worker ==="
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo -e "${GREEN}Worker running (PID: $LOCK_PID)${NC}"
    else
        echo -e "${YELLOW}Stale lock file (PID: $LOCK_PID not running)${NC}"
    fi
else
    echo -e "${YELLOW}No worker running${NC}"
    echo "Start with: ./scripts/codex/worker.sh"
fi
echo ""

# Check queue
echo "=== Queue ==="
if [ ! -f "$QUEUE_FILE" ]; then
    echo "Queue file does not exist (no tasks queued yet)"
    exit 0
fi

# Count by status
pending=$(grep -c '"status":"pending"' "$QUEUE_FILE" 2>/dev/null || echo 0)
in_progress=$(grep -c '"status":"in_progress"' "$QUEUE_FILE" 2>/dev/null || echo 0)
completed=$(grep -c '"status":"completed"' "$QUEUE_FILE" 2>/dev/null || echo 0)
failed=$(grep -c '"status":"failed"' "$QUEUE_FILE" 2>/dev/null || echo 0)

echo "Pending:     $pending"
echo "In Progress: $in_progress"
echo "Completed:   $completed"
echo "Failed:      $failed"
echo ""

# Show pending tasks
if [ "$pending" -gt 0 ]; then
    echo "=== Pending Tasks ==="
    grep '"status":"pending"' "$QUEUE_FILE" | while read -r line; do
        task_id=$(echo "$line" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
        task=$(echo "$line" | grep -o '"task":"[^"]*"' | cut -d'"' -f4)
        priority=$(echo "$line" | grep -o '"priority":"[^"]*"' | cut -d'"' -f4)
        echo "  [$priority] $task_id: $task"
    done
    echo ""
fi

# Show in-progress
if [ "$in_progress" -gt 0 ]; then
    echo "=== In Progress ==="
    grep '"status":"in_progress"' "$QUEUE_FILE" | while read -r line; do
        task_id=$(echo "$line" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
        task=$(echo "$line" | grep -o '"task":"[^"]*"' | cut -d'"' -f4)
        echo "  $task_id: $task"
    done
    echo ""
fi

# Show recent failures
if [ "$failed" -gt 0 ]; then
    echo "=== Recent Failures ==="
    grep '"status":"failed"' "$QUEUE_FILE" | tail -5 | while read -r line; do
        task_id=$(echo "$line" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
        task=$(echo "$line" | grep -o '"task":"[^"]*"' | cut -d'"' -f4)
        echo -e "  ${RED}$task_id: $task${NC}"
    done
    echo ""
fi
