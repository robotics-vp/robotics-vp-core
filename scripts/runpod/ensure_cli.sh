#!/usr/bin/env bash
set -euo pipefail

# ensure_cli.sh — Verify that runpodctl is installed and configured.
# Exit 0 if ready, 1 if not.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

errors=0

# 1. Check runpodctl is installed
if ! command -v runpodctl &>/dev/null; then
    echo "ERROR: runpodctl is not installed."
    echo "  Install: brew install runpod/runpodctl/runpodctl"
    echo "  Or see: https://github.com/runpod/runpodctl"
    errors=$((errors + 1))
else
    echo "OK: runpodctl found at $(command -v runpodctl)"
fi

# 2. Check RUNPOD_API_KEY is set (never print the value)
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY is not set."
    echo "  Set it: export RUNPOD_API_KEY=\"your-key-here\""
    errors=$((errors + 1))
else
    echo "OK: RUNPOD_API_KEY is set"
fi

# 3. Optional: check volume ID
if [ -z "${RUNPOD_VOLUME_ID:-}" ]; then
    echo "WARN: RUNPOD_VOLUME_ID is not set. Required for loop/train pod classes."
else
    echo "OK: RUNPOD_VOLUME_ID is set"
fi

# 4. Run runpodctl version as a basic connectivity check
if command -v runpodctl &>/dev/null; then
    if runpodctl version &>/dev/null; then
        echo "OK: runpodctl version check passed"
    else
        echo "WARN: runpodctl version check failed — CLI may need re-authentication"
    fi
fi

if [ "$errors" -gt 0 ]; then
    echo ""
    echo "FAILED: $errors prerequisite(s) missing. Fix the above and re-run."
    exit 1
fi

echo ""
echo "All RunPod prerequisites satisfied."
exit 0
