#!/usr/bin/env bash
#
# status.sh - Show Codex job status and logs
#
# Usage:
#   ./scripts/codex/status.sh           # Latest job
#   ./scripts/codex/status.sh <job_id>  # Specific job
#   ./scripts/codex/status.sh --all     # List all jobs
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
RUNS_DIR="$REPO_ROOT/.agent/runs"

JOB_ID=""
SHOW_ALL=false
TAIL_LOGS=false
TAIL_LINES=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            SHOW_ALL=true
            shift
            ;;
        --tail|-f)
            TAIL_LOGS=true
            shift
            ;;
        --lines|-n)
            TAIL_LINES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [job_id] [options]"
            echo ""
            echo "Show Codex job status and logs."
            echo ""
            echo "Options:"
            echo "  --all         List all jobs"
            echo "  --tail, -f    Follow logs (if job running)"
            echo "  --lines, -n   Number of log lines to show (default: 50)"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            JOB_ID="$1"
            shift
            ;;
    esac
done

# Ensure runs directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo -e "${YELLOW}No jobs found${NC}"
    echo "Run directory does not exist: $RUNS_DIR"
    exit 0
fi

# List all jobs
if [ "$SHOW_ALL" = true ]; then
    echo -e "${CYAN}=== All Codex Jobs ===${NC}"
    echo ""

    # Find all jobs and show status
    for job_dir in $(ls -1dt "$RUNS_DIR"/*/ 2>/dev/null | head -20); do
        job_name=$(basename "$job_dir")
        meta_file="$job_dir/meta.json"

        if [ -f "$meta_file" ]; then
            status=$(grep -o '"exit_code":[^,}]*' "$meta_file" 2>/dev/null | head -1 | cut -d: -f2 | tr -d ' ' || echo "running")
            task=$(grep -o '"task":"[^"]*"' "$meta_file" 2>/dev/null | head -1 | cut -d'"' -f4 | cut -c1-50 || echo "unknown")

            if [ "$status" = "0" ]; then
                echo -e "${GREEN}[DONE]${NC} $job_name - $task"
            elif [ "$status" = "running" ] || [ "$status" = "null" ]; then
                echo -e "${YELLOW}[RUN]${NC}  $job_name - $task"
            else
                echo -e "${RED}[FAIL]${NC} $job_name - $task"
            fi
        else
            echo -e "${YELLOW}[?]${NC}    $job_name"
        fi
    done

    exit 0
fi

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
    echo ""
    echo "Available jobs:"
    ls -1t "$RUNS_DIR" 2>/dev/null | head -10
    exit 1
fi

echo -e "${CYAN}=== Codex Job Status ===${NC}"
echo "Job ID: $JOB_ID"
echo "Directory: $JOB_DIR"
echo ""

# Show metadata
if [ -f "$JOB_DIR/meta.json" ]; then
    echo "=== Metadata ==="
    cat "$JOB_DIR/meta.json"
    echo ""
fi

# Check for running process
if [ -f "$JOB_DIR/meta.json" ]; then
    PID=$(grep -o '"pid":[0-9]*' "$JOB_DIR/meta.json" 2>/dev/null | cut -d: -f2 || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo -e "${GREEN}Process is running (PID: $PID)${NC}"
        echo ""
    fi
fi

# Show files
echo "=== Files ==="
ls -la "$JOB_DIR/"
echo ""

# Show events/logs
if [ -f "$JOB_DIR/events.jsonl" ]; then
    echo "=== Recent Events ==="
    if [ "$TAIL_LOGS" = true ]; then
        tail -f "$JOB_DIR/events.jsonl"
    else
        tail -n "$TAIL_LINES" "$JOB_DIR/events.jsonl"
    fi
    echo ""
fi

# Show final output
if [ -f "$JOB_DIR/final.md" ]; then
    echo "=== Final Output ==="
    head -100 "$JOB_DIR/final.md"
    if [ "$(wc -l < "$JOB_DIR/final.md")" -gt 100 ]; then
        echo "... (truncated, see $JOB_DIR/final.md)"
    fi
fi
