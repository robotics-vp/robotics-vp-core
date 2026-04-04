#!/usr/bin/env bash
set -euo pipefail

# launch_pod.sh — Launch a RunPod pod for a given pod class.
#
# Usage:
#   ./scripts/runpod/launch_pod.sh --class train --gpu A100-80GB [--volume $RUNPOD_VOLUME_ID] [--template $RUNPOD_TEMPLATE_ID] [--timeout 3600]
#
# Pod classes: loop, provider, train, refactor
# Records pod metadata to .agent/runs/runpod-<timestamp>/meta.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Defaults ---
POD_CLASS=""
GPU_TYPE=""
VOLUME_ID="${RUNPOD_VOLUME_ID:-}"
TEMPLATE_ID="${RUNPOD_TEMPLATE_ID:-}"
TIMEOUT="${RUNPOD_POD_TIMEOUT:-14400}"
POD_NAME=""

# --- Class defaults ---
declare -A CLASS_GPU_DEFAULT=(
    [loop]="NVIDIA A40"
    [provider]="NVIDIA A100 40GB"
    [train]="NVIDIA A100 80GB"
    [refactor]="NVIDIA A40"
)

declare -A CLASS_TIMEOUT_DEFAULT=(
    [loop]=14400      # 4 hours
    [provider]=3600   # 1 hour
    [train]=28800     # 8 hours
    [refactor]=3600   # 1 hour
)

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --class)
            POD_CLASS="$2"; shift 2 ;;
        --gpu)
            GPU_TYPE="$2"; shift 2 ;;
        --volume)
            VOLUME_ID="$2"; shift 2 ;;
        --template)
            TEMPLATE_ID="$2"; shift 2 ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --name)
            POD_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --class <loop|provider|train|refactor> --gpu <GPU_TYPE> [--volume ID] [--template ID] [--timeout SECS] [--name NAME]"
            exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1 ;;
    esac
done

# --- Validate ---
if [ -z "$POD_CLASS" ]; then
    echo "ERROR: --class is required (loop|provider|train|refactor)"
    exit 1
fi

case "$POD_CLASS" in
    loop|provider|train|refactor) ;;
    *)
        echo "ERROR: Invalid pod class '$POD_CLASS'. Must be one of: loop, provider, train, refactor"
        exit 1 ;;
esac

# Apply class defaults if not overridden
if [ -z "$GPU_TYPE" ]; then
    GPU_TYPE="${CLASS_GPU_DEFAULT[$POD_CLASS]}"
    echo "Using default GPU for class '$POD_CLASS': $GPU_TYPE"
fi

if [ "$TIMEOUT" = "${RUNPOD_POD_TIMEOUT:-14400}" ]; then
    TIMEOUT="${CLASS_TIMEOUT_DEFAULT[$POD_CLASS]}"
fi

# Require volume for loop and train
if [[ "$POD_CLASS" == "loop" || "$POD_CLASS" == "train" ]] && [ -z "$VOLUME_ID" ]; then
    echo "ERROR: Pod class '$POD_CLASS' requires a persistent volume."
    echo "  Set RUNPOD_VOLUME_ID or pass --volume <id>"
    exit 1
fi

# Generate run ID and metadata directory
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_ID="runpod-${TIMESTAMP}-$(openssl rand -hex 3)"
RUN_DIR="$REPO_ROOT/.agent/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

# Default pod name
if [ -z "$POD_NAME" ]; then
    POD_NAME="vp-core-${POD_CLASS}-${TIMESTAMP}"
fi

COMMIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"

echo "Launching RunPod pod..."
echo "  Class:    $POD_CLASS"
echo "  GPU:      $GPU_TYPE"
echo "  Timeout:  ${TIMEOUT}s"
echo "  Run ID:   $RUN_ID"

# --- Build runpodctl command ---
CMD=(runpodctl create pod
    --name "$POD_NAME"
    --gpuType "$GPU_TYPE"
    --gpuCount 1
    --imageName "nvidia/cuda:12.1.0-runtime-ubuntu22.04"
)

if [ -n "$VOLUME_ID" ]; then
    CMD+=(--volumeId "$VOLUME_ID" --volumePath "/workspace")
fi

if [ -n "$TEMPLATE_ID" ]; then
    CMD+=(--templateId "$TEMPLATE_ID")
fi

# Execute
echo "Running: ${CMD[*]}"
POD_OUTPUT=$("${CMD[@]}" 2>&1) || {
    echo "ERROR: Pod creation failed:"
    echo "$POD_OUTPUT"
    exit 1
}

echo "$POD_OUTPUT"

# Try to extract pod ID from output
POD_ID=$(echo "$POD_OUTPUT" | grep -oE '[a-z0-9]{20,}' | head -1 || echo "unknown")

# --- Write metadata ---
cat > "$RUN_DIR/meta.json" <<EOF
{
  "run_id": "$RUN_ID",
  "pod_id": "$POD_ID",
  "pod_class": "$POD_CLASS",
  "pod_name": "$POD_NAME",
  "gpu_type": "$GPU_TYPE",
  "volume_id": "$VOLUME_ID",
  "template_id": "$TEMPLATE_ID",
  "timeout_seconds": $TIMEOUT,
  "commit_sha": "$COMMIT_SHA",
  "branch": "$BRANCH",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "launched"
}
EOF

echo ""
echo "Pod metadata written to: $RUN_DIR/meta.json"
echo "Pod ID: $POD_ID"
echo "Run ID: $RUN_ID"
