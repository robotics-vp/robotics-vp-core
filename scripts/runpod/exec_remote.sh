#!/usr/bin/env bash
set -euo pipefail

# exec_remote.sh — Execute a command on a running RunPod pod via SSH.
#
# Usage:
#   ./scripts/runpod/exec_remote.sh --pod <pod_id> -- <command...>
#
# Captures stdout/stderr to .agent/runs/runpod-<run_id>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

POD_ID=""
COMMAND_ARGS=()

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pod)
            POD_ID="$2"; shift 2 ;;
        --)
            shift
            COMMAND_ARGS=("$@")
            break ;;
        -h|--help)
            echo "Usage: $0 --pod <pod_id> -- <command...>"
            exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1. Place command after '--'."
            exit 1 ;;
    esac
done

if [ -z "$POD_ID" ]; then
    echo "ERROR: --pod is required"
    exit 1
fi

if [ ${#COMMAND_ARGS[@]} -eq 0 ]; then
    echo "ERROR: No command specified. Use: $0 --pod <id> -- <command>"
    exit 1
fi

# Generate run ID for this execution
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_ID="runpod-exec-${TIMESTAMP}-$(openssl rand -hex 3)"
RUN_DIR="$REPO_ROOT/.agent/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

REMOTE_CMD="${COMMAND_ARGS[*]}"

echo "Executing on pod $POD_ID:"
echo "  Command: $REMOTE_CMD"
echo "  Run ID:  $RUN_ID"
echo ""

# Record command metadata
cat > "$RUN_DIR/meta.json" <<EOF
{
  "run_id": "$RUN_ID",
  "pod_id": "$POD_ID",
  "command": $(printf '%s' "$REMOTE_CMD" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "running"
}
EOF

# Execute via runpodctl exec (SSH-based)
# runpodctl exec provides SSH access to the pod
EXIT_CODE=0
runpodctl exec "$POD_ID" -- bash -c "$REMOTE_CMD" \
    > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log" \
    || EXIT_CODE=$?

# Update status
if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="completed"
else
    STATUS="failed"
fi

# Update metadata with completion info
python3 -c "
import json, sys
with open('$RUN_DIR/meta.json', 'r') as f:
    meta = json.load(f)
meta['finished_at'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
meta['status'] = '$STATUS'
meta['exit_code'] = $EXIT_CODE
with open('$RUN_DIR/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

echo ""
echo "Status: $STATUS (exit code $EXIT_CODE)"
echo "Stdout: $RUN_DIR/stdout.log"
echo "Stderr: $RUN_DIR/stderr.log"

exit "$EXIT_CODE"
