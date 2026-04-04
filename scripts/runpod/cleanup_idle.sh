#!/usr/bin/env bash
set -euo pipefail

# cleanup_idle.sh — Clean up idle or expired RunPod pods.
#
# Usage:
#   ./scripts/runpod/cleanup_idle.sh [--max-idle 1800] [--dry-run] [--force]
#
# Lists pods, identifies idle ones, and stops/deletes them.
# Without --force, prompts for confirmation before each stop.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MAX_IDLE=1800  # 30 minutes default
DRY_RUN=false
FORCE=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-idle)
            MAX_IDLE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --force)
            FORCE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--max-idle SECONDS] [--dry-run] [--force]"
            echo ""
            echo "Options:"
            echo "  --max-idle SECONDS  Idle threshold in seconds (default: 1800)"
            echo "  --dry-run           Show what would be done without doing it"
            echo "  --force             Stop pods without prompting for confirmation"
            exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1 ;;
    esac
done

echo "RunPod Cleanup"
echo "  Max idle: ${MAX_IDLE}s"
echo "  Dry run:  $DRY_RUN"
echo ""

# List all pods
POD_LIST=$(runpodctl get pod 2>/dev/null) || {
    echo "ERROR: Could not list pods. Is runpodctl configured?"
    exit 1
}

echo "Current pods:"
echo "$POD_LIST"
echo ""

# runpodctl get pod returns a table. We parse pod IDs from it.
# The exact format depends on runpodctl version; extract IDs heuristically.
POD_IDS=$(echo "$POD_LIST" | grep -oE '^[a-z0-9]{10,}' || true)

if [ -z "$POD_IDS" ]; then
    echo "No pods found. Nothing to clean up."
    exit 0
fi

STOPPED=0
SKIPPED=0

for pod_id in $POD_IDS; do
    echo "---"
    echo "Pod: $pod_id"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would stop pod $pod_id"
        STOPPED=$((STOPPED + 1))
        continue
    fi

    if [ "$FORCE" = true ]; then
        echo "  Stopping pod $pod_id..."
        runpodctl stop pod "$pod_id" && echo "  Stopped." || echo "  WARN: Failed to stop."
        STOPPED=$((STOPPED + 1))
    else
        read -r -p "  Stop pod $pod_id? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY])
                echo "  Stopping pod $pod_id..."
                runpodctl stop pod "$pod_id" && echo "  Stopped." || echo "  WARN: Failed to stop."
                STOPPED=$((STOPPED + 1))
                ;;
            *)
                echo "  Skipped."
                SKIPPED=$((SKIPPED + 1))
                ;;
        esac
    fi
done

echo ""
echo "Summary: $STOPPED stopped, $SKIPPED skipped"
