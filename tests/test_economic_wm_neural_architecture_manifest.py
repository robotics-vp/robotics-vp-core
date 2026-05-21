from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.build_economic_wm_neural_architecture_manifest import (
    run_build_economic_wm_neural_architecture_manifest,
)
from src.world_model.economic_world_model import (
    EconomicWMLowerWMConsumptionPreflight,
    build_economic_wm_neural_architecture_manifest,
    load_economic_wm_neural_architecture_manifest,
)


def _preflight() -> EconomicWMLowerWMConsumptionPreflight:
    return EconomicWMLowerWMConsumptionPreflight(
        preflight_id="ewm_lower_wm_consumption_test",
        corpus_id="ewm_corpus_test",
        row_count=2,
        consumption_rows_path="canonical_consumption_rows.jsonl",
        status="ok",
        all_required_wms_referenced=True,
        ready_for_neural_manifest=True,
        ready_for_training=False,
        promotion_eligible=False,
        missing_reference_count=0,
        compiled_reference_count=6,
        direct_reference_count=0,
        summary_only_reference_count=0,
        consumption_row_ids=["row_a", "row_b"],
        aggregate_counts={"row_count": 2.0, "compiled_reference_count": 6.0},
        artifact_refs={
            "consumption_rows_path": "canonical_consumption_rows.jsonl",
            "corpus_manifest_path": "manifest.json",
        },
    )


def test_neural_architecture_manifest_names_full_component_topology() -> None:
    manifest = build_economic_wm_neural_architecture_manifest(
        lower_wm_preflight=_preflight()
    )

    component_keys = {component.component_key for component in manifest.components}
    assert component_keys == {
        "datapack_composition_network",
        "economic_state_estimator",
        "economic_dynamics_model",
        "distributional_pareto_allocator",
        "discrete_receding_horizon_allocator",
        "governance_reciprocity_compiler",
    }
    assert manifest.ready_for_training_scaffold is True
    assert manifest.ready_for_gpu_training is False
    assert manifest.gpu_training_ready is False
    assert manifest.provider_bringup_ready is False
    assert manifest.promotion_eligible is False
    assert manifest.reward_math_mutation is False
    assert manifest.authority_class == "neural_manifest_only"
    assert manifest.aggregate_counts["component_count"] == 6.0
    assert manifest.aggregate_counts["gpu_train_required_count"] == 5.0

    estimator = next(
        component
        for component in manifest.components
        if component.component_key == "economic_state_estimator"
    )
    assert "PerceptionGroundingWorldState" in estimator.input_surfaces
    assert "SimSynthPhysicsWorldState" in estimator.input_surfaces
    assert "EmbodimentActuationWorldState" in estimator.input_surfaces
    assert "EconomicRegime" in estimator.output_surfaces

    allocator = next(
        component
        for component in manifest.components
        if component.component_key == "distributional_pareto_allocator"
    )
    assert "ParetoFrontierSlice" in allocator.output_surfaces
    assert "ShadowPriceField" in allocator.output_surfaces

    for component in manifest.components:
        payload = component.to_dict()
        assert payload["authority_class"] == "neural_scaffold_only"
        assert payload["training_ready"] is False
        assert payload["promotion_eligible"] is False
        assert payload["runtime_plane"]
        assert "gpu_training_runtime_receipt" in payload["promotion_gates"]


def test_neural_architecture_manifest_script_roundtrip(tmp_path) -> None:
    preflight_path = tmp_path / "lower_preflight.json"
    preflight_path.write_text(
        json.dumps(_preflight().to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    payload = run_build_economic_wm_neural_architecture_manifest(
        output_dir=tmp_path / "manifest",
        lower_wm_preflight_path=preflight_path,
        run_lower_wm_preflight_if_missing=False,
    )

    assert payload["authority_class"] == "neural_manifest_only"
    assert payload["ready_for_training_scaffold"] is True
    assert payload["ready_for_gpu_training"] is False
    assert payload["promotion_eligible"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["aggregate_counts"]["component_count"] == 6.0
    assert Path(payload["artifact_refs"]["manifest_path"]).exists()
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()

    loaded = load_economic_wm_neural_architecture_manifest(
        payload["artifact_refs"]["manifest_path"]
    )
    assert loaded.manifest_id == payload["manifest_id"]
    assert len(loaded.components) == 6
