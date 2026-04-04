#!/usr/bin/env bash
set -euo pipefail

# sync_down.sh — Sync results from a RunPod pod to local.
#
# Usage:
#   ./scripts/runpod/sync_down.sh --pod <pod_id> --remote-path /workspace/results --local-path results/run_registry/<run_id>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

POD_ID=""
REMOTE_PATH=""
LOCAL_PATH=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pod)
            POD_ID="$2"; shift 2 ;;
        --remote-path)
            REMOTE_PATH="$2"; shift 2 ;;
        --local-path)
            LOCAL_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --pod <pod_id> --remote-path <path> --local-path <path>"
            exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1 ;;
    esac
done

if [ -z "$POD_ID" ]; then
    echo "ERROR: --pod is required"
    exit 1
fi

if [ -z "$REMOTE_PATH" ]; then
    echo "ERROR: --remote-path is required"
    exit 1
fi

if [ -z "$LOCAL_PATH" ]; then
    echo "ERROR: --local-path is required"
    exit 1
fi

# Resolve local path relative to repo root if not absolute
if [[ "$LOCAL_PATH" != /* ]]; then
    LOCAL_PATH="$REPO_ROOT/$LOCAL_PATH"
fi

# Create local destination
mkdir -p "$LOCAL_PATH"

echo "Syncing results from pod $POD_ID..."
echo "  Remote: $REMOTE_PATH"
echo "  Local:  $LOCAL_PATH"

# Use runpodctl to receive files
runpodctl receive "$POD_ID":"$REMOTE_PATH" "$LOCAL_PATH" || {
    echo "ERROR: Sync failed. Verify pod $POD_ID is running and the remote path exists."
    exit 1
}

echo ""
echo "Sync complete: results pulled to $LOCAL_PATH"

# Report size
if command -v du &>/dev/null; then
    TOTAL_SIZE=$(du -sh "$LOCAL_PATH" 2>/dev/null | cut -f1)
    echo "Total size: $TOTAL_SIZE"
fi
