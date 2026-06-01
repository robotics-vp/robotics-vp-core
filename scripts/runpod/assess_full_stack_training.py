#!/usr/bin/env python3
"""Assess full-stack training readiness against the checked-in Runpod bundle backlog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

from scripts.runpod.full_stack_training import (
    DEFAULT_CONFIG_PATH,
    discover_workspace_state,
    evaluate_bundles,
    load_bundle_config,
    select_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--bundle",
        type=str,
        default="auto",
        help="Specific bundle to highlight, or auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_bundle_config(Path(args.config))
    state = discover_workspace_state()
    assessments = evaluate_bundles(config, state)
    selected = select_bundle(assessments, bundle_id=args.bundle)
    summary = {
        "bundle_config_path": str(Path(args.config).resolve()),
        "workspace_state": state,
        "bundle_assessments": assessments,
        "selected_bundle": selected,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
