#!/usr/bin/env python3
"""Build the Economic WM neural architecture manifest scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model import (  # noqa: E402
    EconomicWMNeuralArchitectureManifest,
    build_economic_wm_neural_architecture_manifest_from_path,
)


def _write_markdown(path: Path, manifest: EconomicWMNeuralArchitectureManifest) -> None:
    payload = manifest.to_dict()
    lines = [
        "# Economic WM Neural Architecture Manifest",
        "",
        f"- Manifest ID: `{payload['manifest_id']}`",
        f"- Lower-WM preflight ID: `{payload['lower_wm_preflight_id']}`",
        f"- Architecture stage: `{payload['architecture_stage']}`",
        f"- Authority class: `{payload['authority_class']}`",
        f"- Ready for training scaffold: `{str(payload['ready_for_training_scaffold']).lower()}`",
        f"- Ready for GPU training: `{str(payload['ready_for_gpu_training']).lower()}`",
        f"- GPU training ready: `{str(payload['gpu_training_ready']).lower()}`",
        f"- Provider bring-up ready: `{str(payload['provider_bringup_ready']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        "",
        "## Components",
        "",
        "| Component | Family | Runtime plane | Outputs |",
        "| --- | --- | --- | --- |",
    ]
    for component in payload["components"]:
        outputs = ", ".join(component["output_surfaces"][:4])
        if len(component["output_surfaces"]) > 4:
            outputs += ", ..."
        lines.append(
            "| "
            f"`{component['component_key']}` | "
            f"{component['model_family']} | "
            f"`{component['runtime_plane']}` | "
            f"{outputs} |"
        )
    lines.extend(["", "## Training blockers"])
    lines.extend(f"- `{blocker}`" for blocker in payload["training_blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a neural topology and training-contract manifest only. It names future learned surfaces, inputs, outputs, losses, and gates, but it does not instantiate weights, run GPU training, run provider bring-up, promote a model, or mutate frozen reward/trust/`w_econ`/lambda math.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_lower_wm_preflight(
    *,
    output_root: Path,
    lower_wm_preflight_path: Optional[str | Path],
    run_if_missing: bool,
) -> Path:
    preflight = Path(
        lower_wm_preflight_path
        or "artifacts/economic_world_model/economic_wm_lower_wm_consumption_preflight/economic_wm_lower_wm_consumption_preflight_v1.json"
    )
    if preflight.exists():
        return preflight
    if not run_if_missing:
        raise FileNotFoundError(preflight)

    from scripts.economic_world_model.prepare_economic_wm_lower_wm_consumption_preflight import (  # noqa: E501
        run_prepare_economic_wm_lower_wm_consumption_preflight,
    )

    payload = run_prepare_economic_wm_lower_wm_consumption_preflight(
        output_dir=output_root / "lower_wm_consumption_preflight"
    )
    return Path(payload["artifact_refs"]["preflight_path"])


def run_build_economic_wm_neural_architecture_manifest(
    *,
    output_dir: str | Path,
    lower_wm_preflight_path: Optional[str | Path] = None,
    run_lower_wm_preflight_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_preflight_path = _resolve_lower_wm_preflight(
        output_root=output_root,
        lower_wm_preflight_path=lower_wm_preflight_path,
        run_if_missing=run_lower_wm_preflight_if_missing,
    )
    manifest_path = output_root / "economic_wm_neural_architecture_manifest_v1.json"
    markdown_path = output_root / "economic_wm_neural_architecture_manifest_v1.md"
    manifest = build_economic_wm_neural_architecture_manifest_from_path(
        lower_wm_preflight_path=resolved_preflight_path,
        output_path=manifest_path,
        metadata={"source": "build_economic_wm_neural_architecture_manifest_script"},
    )
    payload = manifest.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "manifest_path": str(manifest_path),
        "markdown_path": str(markdown_path),
        "lower_wm_preflight_path": str(resolved_preflight_path),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(
        markdown_path, EconomicWMNeuralArchitectureManifest.from_dict(payload)
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_neural_architecture_manifest",
        help="Directory for Economic WM neural architecture manifest artifacts.",
    )
    parser.add_argument("--lower-wm-preflight", default=None)
    parser.add_argument(
        "--no-run-lower-wm-preflight",
        action="store_true",
        help="Do not run the lower-WM consumption preflight if the input is missing.",
    )
    args = parser.parse_args()
    payload = run_build_economic_wm_neural_architecture_manifest(
        output_dir=args.output_dir,
        lower_wm_preflight_path=args.lower_wm_preflight,
        run_lower_wm_preflight_if_missing=not args.no_run_lower_wm_preflight,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["authority_class"] == "neural_manifest_only"
        and payload["ready_for_training_scaffold"]
        and not payload["ready_for_gpu_training"]
        and not payload["gpu_training_ready"]
        and not payload["provider_bringup_ready"]
        and not payload["promotion_eligible"]
        and not payload["reward_math_mutation"]
        and payload["aggregate_counts"].get("component_count", 0) >= 4
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
