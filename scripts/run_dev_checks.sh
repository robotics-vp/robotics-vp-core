#!/usr/bin/env bash
set -euo pipefail

# Convenience script to run basic linting + core smokes.
# This is advisory-only and does NOT touch Phase B math or reward behavior.

echo "[dev-checks] Running formatters (black/isort/ruff)..."
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit run --all-files || true
else
  echo "[dev-checks] pre-commit not installed; skipping formatter hooks."
fi

echo "[dev-checks] Running core smokes..."
python3 scripts/smoke_test_dependency_hierarchy.py
python3 scripts/smoke_test_pareto_frontier.py
python3 scripts/smoke_test_semantic_feedback_loop.py
python3 scripts/smoke_test_reward_builder.py
if [ -f "scripts/smoke_test_vision_backbone.py" ]; then
  python3 scripts/smoke_test_vision_backbone.py
fi

echo "[dev-checks] Done."
