#!/usr/bin/env bash
#
# run_cli.sh - Run Codex task via CLI
#
# Usage:
#   ./scripts/codex/run_cli.sh "task description"
#   ./scripts/codex/run_cli.sh --schema schema.json "task description"
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
SCHEMA=""
TIMEOUT=600
TASK=""
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --schema)
            SCHEMA="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options] \"task description\""
            echo ""
            echo "Run a Codex task via CLI."
            echo ""
            echo "Options:"
            echo "  --schema FILE     Output schema for structured output"
            echo "  --timeout SECS    Timeout in seconds (default: 600)"
            echo "  --resume ID       Resume session (ID or --last)"
            echo "  --help            Show this help"
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

# Check for codex CLI
if ! command -v codex &> /dev/null; then
    echo -e "${RED}Error: Codex CLI not found${NC}"
    echo ""
    echo "Install with:"
    echo "  npm install -g @openai/codex"
    echo "  # or"
    echo "  brew install codex"
    echo ""
    echo "Then authenticate:"
    echo "  export OPENAI_API_KEY=\"your-key\""
    echo "  codex whoami"
    exit 1
fi

# Check auth
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not set${NC}"
    echo "Codex may not be able to authenticate."
fi

# Generate job ID
JOB_ID="$(date +%Y%m%d-%H%M%S)-$(openssl rand -hex 4 2>/dev/null || echo $$)"
RUN_DIR="$REPO_ROOT/.agent/runs/$JOB_ID"

# Create run directory
mkdir -p "$RUN_DIR"

echo -e "${CYAN}=== Codex CLI Execution ===${NC}"
echo "Job ID: $JOB_ID"
echo "Run directory: $RUN_DIR"
echo "Task: $TASK"
echo ""

# Build codex command
CODEX_ARGS=("exec")

# Add --json for event streaming (critical for observability)
CODEX_ARGS+=("--json")

# Add schema if provided
if [ -n "$SCHEMA" ]; then
    if [ -f "$SCHEMA" ]; then
        CODEX_ARGS+=("--output-schema" "$SCHEMA")
    else
        echo -e "${RED}Error: Schema file not found: $SCHEMA${NC}"
        exit 1
    fi
fi

# Add output file
CODEX_ARGS+=("-o" "$RUN_DIR/final.txt")

# Handle resume
if [ -n "$RESUME" ]; then
    CODEX_ARGS+=("resume")
    if [ "$RESUME" = "--last" ]; then
        CODEX_ARGS+=("--last")
    else
        CODEX_ARGS+=("$RESUME")
    fi
elif [ -n "$TASK" ]; then
    CODEX_ARGS+=("$TASK")
else
    echo -e "${RED}Error: Either task or --resume required${NC}"
    exit 1
fi

# Write metadata
cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cli",
  "task": "$TASK",
  "schema": "$SCHEMA",
  "timeout": $TIMEOUT,
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "pid": null,
  "command": "codex ${CODEX_ARGS[*]}"
}
EOF

echo -e "${GREEN}Starting Codex...${NC}"
echo "Command: codex ${CODEX_ARGS[*]}"
echo ""

# Run codex with timeout, capturing events
START_TIME=$(date +%s)
set +e

# Run with timeout, stream JSON events to file
timeout "$TIMEOUT" codex "${CODEX_ARGS[@]}" 2>&1 | tee "$RUN_DIR/events.jsonl" &
CODEX_PID=$!

# Update metadata with PID
cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cli",
  "task": "$TASK",
  "schema": "$SCHEMA",
  "timeout": $TIMEOUT,
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "pid": $CODEX_PID,
  "command": "codex ${CODEX_ARGS[*]}"
}
EOF

# Wait for completion
wait $CODEX_PID
EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Update metadata with completion
cat > "$RUN_DIR/meta.json" << EOF
{
  "job_id": "$JOB_ID",
  "mode": "cli",
  "task": "$TASK",
  "schema": "$SCHEMA",
  "timeout": $TIMEOUT,
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": $DURATION,
  "exit_code": $EXIT_CODE,
  "pid": $CODEX_PID,
  "command": "codex ${CODEX_ARGS[*]}"
}
EOF

echo ""
echo -e "${CYAN}=== Execution Complete ===${NC}"
echo "Duration: ${DURATION}s"
echo "Exit code: $EXIT_CODE"
echo "Logs: $RUN_DIR/"

# Parse result
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Success!${NC}"

    # Try to create final.md from output
    if [ -f "$RUN_DIR/final.txt" ]; then
        cp "$RUN_DIR/final.txt" "$RUN_DIR/final.md"
    fi

    # Show summary if available
    if [ -f "$RUN_DIR/final.md" ]; then
        echo ""
        echo "=== Output ==="
        head -50 "$RUN_DIR/final.md"
        if [ "$(wc -l < "$RUN_DIR/final.md")" -gt 50 ]; then
            echo "... (truncated, see $RUN_DIR/final.md)"
        fi
    fi
elif [ $EXIT_CODE -eq 124 ]; then
    echo -e "${RED}Timeout after ${TIMEOUT}s${NC}"
    echo ""
    echo "Consider:"
    echo "  1. Increase timeout with --timeout"
    echo "  2. Split task into smaller pieces"
    echo "  3. Check logs in $RUN_DIR/"
    exit 1
else
    echo -e "${RED}Failed with exit code $EXIT_CODE${NC}"
    echo ""
    echo "Check logs in $RUN_DIR/"

    # Show recent events if available
    if [ -f "$RUN_DIR/events.jsonl" ]; then
        echo ""
        echo "=== Last events ==="
        tail -20 "$RUN_DIR/events.jsonl"
    fi
    exit 1
fi
