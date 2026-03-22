#!/usr/bin/env bash
#
# Publish a completed Codex change to origin/main when possible, or fall back
# to a timestamped feature branch when direct main publication is rejected.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE="origin"
BASE_BRANCH="main"
FEATURE_PREFIX="codex/ewm-nightly"
DRY_RUN=false

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Publish the current HEAD to the remote base branch when possible."
    echo "If pushing to the base branch fails from local main, fall back to a"
    echo "timestamped feature branch so the work is still visible remotely."
    echo ""
    echo "Options:"
    echo "  --remote NAME          Remote name (default: origin)"
    echo "  --base-branch NAME     Base branch to publish directly (default: main)"
    echo "  --feature-prefix NAME  Fallback feature branch prefix"
    echo "                         (default: codex/ewm-nightly)"
    echo "  --dry-run              Print the intended publish target without pushing"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)
            REMOTE="$2"
            shift 2
            ;;
        --base-branch)
            BASE_BRANCH="$2"
            shift 2
            ;;
        --feature-prefix)
            FEATURE_PREFIX="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null

CURRENT_BRANCH="$(git -C "$REPO_ROOT" symbolic-ref --quiet --short HEAD || true)"
if [ -z "$CURRENT_BRANCH" ]; then
    echo "Detached HEAD is not supported for publication." >&2
    exit 1
fi

if [ "$DRY_RUN" = false ] && [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    echo "Working tree is dirty; commit or stash changes before publishing." >&2
    exit 1
fi

push_ref() {
    local local_ref="$1"
    local remote_branch="$2"

    if [ "$DRY_RUN" = true ]; then
        echo "DRY_RUN_REMOTE=$REMOTE"
        echo "DRY_RUN_BRANCH=$remote_branch"
        echo "DRY_RUN_REFSPEC=$local_ref:refs/heads/$remote_branch"
        return 0
    fi

    git -C "$REPO_ROOT" push "$REMOTE" "$local_ref:refs/heads/$remote_branch"
}

emit_result() {
    local mode="$1"
    local branch="$2"
    local prefix="PUBLISHED"

    if [ "$DRY_RUN" = true ]; then
        prefix="PLANNED"
    fi

    echo "${prefix}_MODE=$mode"
    echo "${prefix}_BRANCH=$branch"
    echo "${prefix}_REF=$REMOTE/$branch"
}

if [ "$CURRENT_BRANCH" != "$BASE_BRANCH" ]; then
    push_ref HEAD "$CURRENT_BRANCH"
    emit_result "current_branch" "$CURRENT_BRANCH"
    exit 0
fi

if push_ref HEAD "$BASE_BRANCH"; then
    emit_result "direct_main" "$BASE_BRANCH"
    exit 0
fi

FALLBACK_BRANCH="${FEATURE_PREFIX}-$(date +%Y%m%d-%H%M%S)"

# Preserve the current HEAD under a remote-visible branch when main rejects it.
if [ "$DRY_RUN" = false ]; then
    git -C "$REPO_ROOT" branch --force "$FALLBACK_BRANCH" HEAD >/dev/null
fi

push_ref "${FALLBACK_BRANCH:-HEAD}" "$FALLBACK_BRANCH"
emit_result "fallback_feature_branch" "$FALLBACK_BRANCH"
