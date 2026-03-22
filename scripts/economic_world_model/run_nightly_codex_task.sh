#!/usr/bin/env bash
#
# Run or queue the single highest-value next additive roadmap task.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="auto"
ENV_ID="${CODEX_CLOUD_ENV_ID:-${CODEX_CLOUD_ENV:-}}"
TIMEOUT=1800
QUEUE_ONLY=false
SKIP_AUDIT=false
AUDIT_JSON="$REPO_ROOT/artifacts/economic_world_model/nightly_audit_summary.json"
AUDIT_MARKDOWN="$REPO_ROOT/artifacts/economic_world_model/nightly_audit_summary.md"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --env)
            ENV_ID="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --queue-only)
            QUEUE_ONLY=true
            shift
            ;;
        --skip-audit)
            SKIP_AUDIT=true
            shift
            ;;
        --audit-json)
            AUDIT_JSON="$2"
            shift 2
            ;;
        --audit-markdown)
            AUDIT_MARKDOWN="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run or queue the single nightly economic-world-model task."
            echo ""
            echo "Options:"
            echo "  --mode MODE        auto|cli|cloud (default: auto)"
            echo "  --env ENV_ID       Codex cloud environment ID when --mode cloud"
            echo "  --timeout SECS     Codex timeout in seconds (default: 1800)"
            echo "  --queue-only       Add the task to the local Codex queue instead of executing"
            echo "  --skip-audit       Reuse existing audit artifact instead of regenerating it"
            echo "  --audit-json PATH  Audit JSON path"
            echo "  --audit-markdown PATH Audit markdown path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$AUDIT_JSON")"

if [ "$SKIP_AUDIT" = false ] || [ ! -f "$AUDIT_JSON" ]; then
    python3 "$REPO_ROOT/scripts/economic_world_model/nightly_audit.py" \
        --output-json "$AUDIT_JSON" \
        --output-markdown "$AUDIT_MARKDOWN"
fi

eval "$(python3 - "$AUDIT_JSON" <<'PY'
import json
import shlex
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
task = data.get("next_task", {})

def emit(key, value):
    print(f"{key}={shlex.quote(str(value))}")

emit("EXECUTE_NOW", str(bool(task.get("execute_now"))).lower())
emit("NEXT_TITLE", task.get("title", ""))
emit("NEXT_CLASSIFICATION", task.get("classification", ""))
emit("NEXT_RATIONALE", task.get("rationale", ""))
emit("SUMMARY_DIGEST", data.get("summary_digest", ""))
PY
)"

if [ "$EXECUTE_NOW" != "true" ]; then
    echo "No safe additive task selected by the audit."
    echo "Summary digest: $SUMMARY_DIGEST"
    exit 0
fi

TASK_PROMPT="$(cat <<EOF
Follow codex_skills/economic-world-model-roadmap/SKILL.md.
Read docs/economic_world_model/architecture_gap_analysis.md, docs/economic_world_model/roadmap.md, docs/economic_world_model/progress_log.md, docs/economic_world_model/nightly_audit.md, docs/economic_world_model/implementation_notes.md, and $AUDIT_MARKDOWN.
Execute exactly one task: $NEXT_TITLE.
Classification: $NEXT_CLASSIFICATION.
Rationale: $NEXT_RATIONALE.
Keep VLA and foundation-model paths external and sidecar/advisory.
Do not touch frozen zones: checkpoints/stable_world_model.pt, legacy baseline world-model math, trust_net, w_econ lattice math, lambda controller equations, or src/controllers/synthetic_weight_controller.py core logic.
Additive successor modules in src/world_model/ are allowed only when they preserve the stable baseline as the rollback anchor and stay advisory/governed.
Prefer additive docs, scaffolding, tests, and sidecars before invasive rewrites.
After changes, run verification, update docs/economic_world_model/progress_log.md and docs/economic_world_model/implementation_notes.md, and summarize blockers and the next recommended task.
If you create a commit, publish it before finishing by running bash scripts/economic_world_model/publish_codex_change.sh --base-branch main --feature-prefix codex/ewm-nightly.
Prefer a direct push to origin/main when it is a safe fast-forward. If main rejects the push, let the helper publish a timestamped feature branch instead, and include the published ref or exact push blocker in the final summary.
EOF
)"

echo "Selected task: $NEXT_TITLE"
echo "Classification: $NEXT_CLASSIFICATION"
echo "Summary digest: $SUMMARY_DIGEST"

if [ "$QUEUE_ONLY" = true ]; then
    QUEUE_TASK="$TASK_PROMPT"
    if [ "$MODE" = "cloud" ]; then
        QUEUE_TASK="[cloud] $QUEUE_TASK"
        echo "Queue mode assumes CODEX_CLOUD_ENV_ID is set for the worker."
    fi
    "$REPO_ROOT/scripts/codex/enqueue.sh" --mode "$MODE" "$QUEUE_TASK"
    exit 0
fi

if [ "$MODE" = "cloud" ]; then
    if [ -z "$ENV_ID" ]; then
        echo "Cloud mode requires --env or CODEX_CLOUD_ENV_ID" >&2
        exit 1
    fi
    "$REPO_ROOT/scripts/codex/run.sh" \
        --mode cloud \
        --env "$ENV_ID" \
        --apply \
        --timeout "$TIMEOUT" \
        "$TASK_PROMPT"
    exit 0
fi

"$REPO_ROOT/scripts/codex/run.sh" \
    --mode "$MODE" \
    --timeout "$TIMEOUT" \
    "$TASK_PROMPT"
