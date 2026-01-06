#!/usr/bin/env bash
#
# enqueue.sh - Add a task to the Codex queue
#
# Usage:
#   ./scripts/codex/enqueue.sh "task description"
#   ./scripts/codex/enqueue.sh --mode cli "task description"
#   ./scripts/codex/enqueue.sh --priority high "urgent task"
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUEUE_DIR="$REPO_ROOT/.agent/queue"
QUEUE_FILE="$QUEUE_DIR/codex.jsonl"

# Defaults
MODE="auto"
PRIORITY="normal"
SCHEMA=""
TASK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --priority)
            PRIORITY="$2"
            shift 2
            ;;
        --schema)
            SCHEMA="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options] \"task description\""
            echo ""
            echo "Add a task to the Codex queue."
            echo ""
            echo "Options:"
            echo "  --mode mcp|cli|auto  Execution mode (default: auto)"
            echo "  --priority LEVEL     Priority: high, normal, low (default: normal)"
            echo "  --schema FILE        Output schema file"
            echo "  --help               Show this help"
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
    echo "Error: Task description required"
    echo "Usage: $0 \"task description\""
    exit 1
fi

# Ensure queue directory exists
mkdir -p "$QUEUE_DIR"

# Generate task ID
TASK_ID="$(date +%Y%m%d-%H%M%S)-$(openssl rand -hex 4 2>/dev/null || echo $$)"

# Create queue entry
ENTRY=$(cat << EOF
{"id":"$TASK_ID","task":"$TASK","mode":"$MODE","priority":"$PRIORITY","schema":"$SCHEMA","created_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","status":"pending"}
EOF
)

# Append to queue
echo "$ENTRY" >> "$QUEUE_FILE"

echo -e "${GREEN}Task queued${NC}"
echo "ID: $TASK_ID"
echo "Task: $TASK"
echo "Mode: $MODE"
echo "Priority: $PRIORITY"
echo ""
echo "Start worker: ./scripts/codex/worker.sh"
echo "Check status: ./scripts/codex/worker_status.sh"
