#!/usr/bin/env bash
set -euo pipefail

# sync_up.sh — Sync repo to a RunPod pod.
#
# Usage:
#   ./scripts/runpod/sync_up.sh --pod <pod_id> [--exclude .git --exclude results]
#
# Uses rsync via runpodctl SSH to push the repo to /workspace/ on the pod.
# Respects .gitignore patterns by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

POD_ID=""
EXTRA_EXCLUDES=()

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pod)
            POD_ID="$2"; shift 2 ;;
        --exclude)
            EXTRA_EXCLUDES+=("--exclude" "$2"); shift 2 ;;
        -h|--help)
            echo "Usage: $0 --pod <pod_id> [--exclude PATTERN ...]"
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

if ! command -v rsync &>/dev/null; then
    echo "ERROR: rsync is not installed"
    exit 1
fi

# Get pod SSH info via runpodctl
echo "Resolving SSH connection for pod $POD_ID..."
SSH_INFO=$(runpodctl get pod "$POD_ID" 2>/dev/null) || {
    echo "ERROR: Could not retrieve pod info for $POD_ID"
    exit 1
}

# Build rsync command
# Uses .gitignore as the base exclusion filter
RSYNC_CMD=(
    rsync -avz --progress
    --filter=":- .gitignore"
    --exclude ".agent/runs/"
    --exclude "__pycache__/"
    --exclude "*.pyc"
    --exclude ".env"
)

# Add extra excludes
if [ ${#EXTRA_EXCLUDES[@]} -gt 0 ]; then
    RSYNC_CMD+=("${EXTRA_EXCLUDES[@]}")
fi

echo "Syncing repo to pod $POD_ID:/workspace/"
echo "  Source: $REPO_ROOT/"
echo "  Excludes: .gitignore patterns + .agent/runs/ + __pycache__/ + .env"
if [ ${#EXTRA_EXCLUDES[@]} -gt 0 ]; then
    echo "  Extra excludes: ${EXTRA_EXCLUDES[*]}"
fi

# Use runpodctl to transfer files
# runpodctl send/receive handles SSH tunneling
runpodctl send "$REPO_ROOT" "$POD_ID":/workspace/ || {
    echo "ERROR: Sync failed. Verify pod $POD_ID is running and SSH is available."
    exit 1
}

echo ""
echo "Sync complete: repo pushed to pod $POD_ID:/workspace/"
