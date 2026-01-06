#!/usr/bin/env bash
#
# watchdog.sh - Monitor Codex jobs and handle timeouts
#
# Watches running jobs for:
# - Start acknowledgment (60s)
# - Heartbeat/activity (5min)
# - Completion or failure
#
# Usage:
#   ./scripts/codex/watchdog.sh           # Watch latest job
#   ./scripts/codex/watchdog.sh <job_id>  # Watch specific job
#   ./scripts/codex/watchdog.sh --daemon  # Run continuously
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

# Thresholds (in seconds)
ACK_TIMEOUT=60
HEARTBEAT_TIMEOUT=300  # 5 minutes
STALE_TIMEOUT=600      # 10 minutes
MAX_RETRIES=3

JOB_ID=""
DAEMON_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon|-d)
            DAEMON_MODE=true
            shift
            ;;
        --ack-timeout)
            ACK_TIMEOUT="$2"
            shift 2
            ;;
        --heartbeat-timeout)
            HEARTBEAT_TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [job_id] [options]"
            echo ""
            echo "Monitor Codex jobs for activity and handle failures."
            echo ""
            echo "Options:"
            echo "  --daemon, -d        Run continuously watching all jobs"
            echo "  --ack-timeout SEC   Start ack timeout (default: 60)"
            echo "  --heartbeat-timeout SEC  Heartbeat timeout (default: 300)"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            JOB_ID="$1"
            shift
            ;;
    esac
done

# Function to check job health
check_job() {
    local job_id="$1"
    local job_dir="$RUNS_DIR/$job_id"

    if [ ! -d "$job_dir" ]; then
        return 1
    fi

    # Check metadata
    if [ ! -f "$job_dir/meta.json" ]; then
        echo -e "${YELLOW}[$job_id]${NC} No metadata"
        return 0
    fi

    # Check if already completed
    local exit_code=$(grep -o '"exit_code":[0-9]*' "$job_dir/meta.json" 2>/dev/null | cut -d: -f2 || echo "")
    if [ -n "$exit_code" ] && [ "$exit_code" != "null" ]; then
        if [ "$exit_code" = "0" ]; then
            echo -e "${GREEN}[$job_id]${NC} Completed successfully"
        else
            echo -e "${RED}[$job_id]${NC} Failed with exit code $exit_code"
        fi
        return 0
    fi

    # Check PID
    local pid=$(grep -o '"pid":[0-9]*' "$job_dir/meta.json" 2>/dev/null | cut -d: -f2 || echo "")
    if [ -z "$pid" ] || [ "$pid" = "null" ]; then
        echo -e "${YELLOW}[$job_id]${NC} No PID (may be MCP job)"
        return 0
    fi

    # Check if process is running
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}[$job_id]${NC} Process not running (PID: $pid)"
        return 0
    fi

    echo -e "${GREEN}[$job_id]${NC} Running (PID: $pid)"

    # Check for recent activity
    if [ -f "$job_dir/events.jsonl" ]; then
        local last_mod=$(stat -f %m "$job_dir/events.jsonl" 2>/dev/null || stat -c %Y "$job_dir/events.jsonl" 2>/dev/null || echo 0)
        local now=$(date +%s)
        local age=$((now - last_mod))

        if [ $age -gt $STALE_TIMEOUT ]; then
            echo -e "${RED}  No activity for ${age}s (stale threshold: ${STALE_TIMEOUT}s)${NC}"
            echo "  Consider cancelling and retrying with smaller scope"
            return 2  # Stale
        elif [ $age -gt $HEARTBEAT_TIMEOUT ]; then
            echo -e "${YELLOW}  No activity for ${age}s (heartbeat threshold: ${HEARTBEAT_TIMEOUT}s)${NC}"
            return 1  # Warning
        else
            echo "  Last activity: ${age}s ago"
        fi
    else
        # Check metadata file modification instead
        local meta_mod=$(stat -f %m "$job_dir/meta.json" 2>/dev/null || stat -c %Y "$job_dir/meta.json" 2>/dev/null || echo 0)
        local now=$(date +%s)
        local age=$((now - meta_mod))

        if [ $age -gt $ACK_TIMEOUT ]; then
            echo -e "${YELLOW}  No events file after ${age}s${NC}"
        fi
    fi

    return 0
}

# Function to handle stale job
handle_stale_job() {
    local job_id="$1"
    local job_dir="$RUNS_DIR/$job_id"

    echo -e "${YELLOW}Handling stale job: $job_id${NC}"

    # Get task from metadata
    local task=$(grep -o '"task":"[^"]*"' "$job_dir/meta.json" 2>/dev/null | cut -d'"' -f4 || echo "")

    if [ -z "$task" ]; then
        echo "Cannot determine original task"
        return 1
    fi

    # Get retry count
    local retries=0
    if [ -f "$job_dir/retries.txt" ]; then
        retries=$(cat "$job_dir/retries.txt")
    fi

    if [ $retries -ge $MAX_RETRIES ]; then
        echo -e "${RED}Max retries ($MAX_RETRIES) exceeded${NC}"
        echo "Manual intervention required for task: $task"
        return 1
    fi

    # Cancel current job
    "$SCRIPT_DIR/cancel.sh" "$job_id" --force

    # Increment retry counter
    echo $((retries + 1)) > "$job_dir/retries.txt"

    # Suggest splitting
    echo ""
    echo "=== Suggested Action ==="
    echo "Task appears to be too large. Consider splitting:"
    echo ""
    echo "1. By directory:"
    echo "   ./scripts/codex/run.sh \"$task - focus on src/\""
    echo "   ./scripts/codex/run.sh \"$task - focus on tests/\""
    echo ""
    echo "2. By phase:"
    echo "   ./scripts/codex/run.sh \"Write tests for: $task\""
    echo "   ./scripts/codex/run.sh \"Implement: $task\""
    echo ""
    echo "3. Manually queue smaller pieces:"
    echo "   ./scripts/codex/enqueue.sh \"smaller piece 1\""
    echo "   ./scripts/codex/enqueue.sh \"smaller piece 2\""
}

# Main logic
if [ "$DAEMON_MODE" = true ]; then
    echo -e "${CYAN}=== Watchdog Daemon ===${NC}"
    echo "Monitoring all jobs..."
    echo "Press Ctrl+C to stop"
    echo ""

    while true; do
        echo "--- $(date) ---"

        # Check all running jobs
        for job_dir in "$RUNS_DIR"/*/; do
            if [ -d "$job_dir" ]; then
                job_id=$(basename "$job_dir")
                result=0
                check_job "$job_id" || result=$?

                if [ $result -eq 2 ]; then
                    handle_stale_job "$job_id"
                fi
            fi
        done

        echo ""
        sleep 30
    done
else
    # Single job mode
    if [ -z "$JOB_ID" ]; then
        JOB_ID=$(ls -1t "$RUNS_DIR" 2>/dev/null | head -1)
        if [ -z "$JOB_ID" ]; then
            echo -e "${YELLOW}No jobs found${NC}"
            exit 0
        fi
    fi

    echo -e "${CYAN}=== Watchdog Check ===${NC}"
    result=0
    check_job "$JOB_ID" || result=$?

    if [ $result -eq 2 ]; then
        echo ""
        read -p "Job is stale. Handle it? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            handle_stale_job "$JOB_ID"
        fi
    fi
fi
