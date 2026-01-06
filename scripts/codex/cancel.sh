#!/usr/bin/env bash
#
# cancel.sh - Cancel a running Codex job
#
# Usage:
#   ./scripts/codex/cancel.sh           # Cancel latest job
#   ./scripts/codex/cancel.sh <job_id>  # Cancel specific job
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNS_DIR="$REPO_ROOT/.agent/runs"

JOB_ID=""
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [job_id] [--force]"
            echo ""
            echo "Cancel a running Codex job."
            echo ""
            echo "Options:"
            echo "  --force, -f   Force kill (SIGKILL instead of SIGTERM)"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            JOB_ID="$1"
            shift
            ;;
    esac
done

# Get latest job if none specified
if [ -z "$JOB_ID" ]; then
    JOB_ID=$(ls -1t "$RUNS_DIR" 2>/dev/null | head -1)
    if [ -z "$JOB_ID" ]; then
        echo -e "${YELLOW}No jobs found${NC}"
        exit 0
    fi
fi

# Find job directory
JOB_DIR="$RUNS_DIR/$JOB_ID"
if [ ! -d "$JOB_DIR" ]; then
    echo -e "${RED}Job not found: $JOB_ID${NC}"
    exit 1
fi

echo "Cancelling job: $JOB_ID"

# Get PID from metadata
if [ ! -f "$JOB_DIR/meta.json" ]; then
    echo -e "${YELLOW}No metadata file found${NC}"
    exit 1
fi

PID=$(grep -o '"pid":[0-9]*' "$JOB_DIR/meta.json" 2>/dev/null | cut -d: -f2 || echo "")

if [ -z "$PID" ] || [ "$PID" = "null" ]; then
    echo -e "${YELLOW}No PID found in metadata${NC}"
    echo "Job may have already completed or was not started via CLI"
    exit 0
fi

# Check if process is running
if ! kill -0 "$PID" 2>/dev/null; then
    echo -e "${YELLOW}Process $PID is not running${NC}"
    echo "Job may have already completed"
    exit 0
fi

# Kill the process
if [ "$FORCE" = true ]; then
    echo "Force killing PID $PID..."
    kill -9 "$PID" 2>/dev/null || true
else
    echo "Terminating PID $PID..."
    kill "$PID" 2>/dev/null || true

    # Wait a moment
    sleep 2

    # Check if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${YELLOW}Process still running, force killing...${NC}"
        kill -9 "$PID" 2>/dev/null || true
    fi
fi

# Verify
sleep 1
if kill -0 "$PID" 2>/dev/null; then
    echo -e "${RED}Failed to kill process $PID${NC}"
    exit 1
else
    echo -e "${GREEN}Job cancelled${NC}"

    # Update metadata
    if [ -f "$JOB_DIR/meta.json" ]; then
        # Add cancelled flag (simple append, not proper JSON update)
        echo "Job cancelled at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOB_DIR/cancelled.txt"
    fi
fi
