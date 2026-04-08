#!/usr/bin/env bash
set -euo pipefail

# collect_billing.sh — Collect billing/cost info for a pod or all pods.
#
# Usage:
#   ./scripts/runpod/collect_billing.sh [--pod <pod_id>]
#
# Without --pod, reports billing for all active pods.
# Appends cost snapshot to the run manifest if a matching run directory exists.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

POD_ID=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pod)
            POD_ID="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--pod <pod_id>]"
            exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1 ;;
    esac
done

echo "Collecting RunPod billing information..."
echo ""

if [ -n "$POD_ID" ]; then
    # Billing for a specific pod
    echo "Pod: $POD_ID"
    runpodctl get pod "$POD_ID" || {
        echo "ERROR: Could not retrieve info for pod $POD_ID"
        exit 1
    }

    # Try to find and update the matching run manifest
    MATCHING_RUN=""
    for run_dir in "$REPO_ROOT"/.agent/runs/runpod-*/; do
        if [ -f "$run_dir/meta.json" ]; then
            FOUND_POD=$(python3 -c "import json; print(json.load(open('${run_dir}meta.json')).get('pod_id',''))" 2>/dev/null || echo "")
            if [ "$FOUND_POD" = "$POD_ID" ]; then
                MATCHING_RUN="$run_dir"
                break
            fi
        fi
    done

    if [ -n "$MATCHING_RUN" ]; then
        echo ""
        echo "Appending cost snapshot to: ${MATCHING_RUN}meta.json"
        python3 -c "
import json
from datetime import datetime, timezone
meta_path = '${MATCHING_RUN}meta.json'
with open(meta_path, 'r') as f:
    meta = json.load(f)
meta['cost_snapshot'] = {
    'collected_at': datetime.now(timezone.utc).isoformat(),
    'note': 'Manual billing collection — check RunPod console for exact cost'
}
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print('Cost snapshot recorded.')
" || echo "WARN: Could not update manifest"
    fi
else
    # Billing for all pods
    echo "All active pods:"
    runpodctl get pod || {
        echo "ERROR: Could not retrieve pod listing"
        exit 1
    }
fi

echo ""
echo "For detailed billing, visit: https://www.runpod.io/console/billing"
