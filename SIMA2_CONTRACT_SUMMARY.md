# SIMA-2 Contract Summary

## Components & IO
- **Client (`src/sima2/client.py`)**: produces rollouts with `episode_id`, `task`, `events/primitives`, plus SIMA-2 provenance (`sima2_backend_id`, `sima2_model_version`, `sima2_task_spec`). Configured via `config/sima2.yaml`.
- **Segmentation Engine (`src/sima2/segmentation_engine.py`)**: consumes rollouts and emits `segments`, `segment_boundaries`, `subtask_tags`; segmentation thresholds pulled from `config/sima2.yaml`.
- **Primitive Extractor (`src/sima2/semantic_primitive_extractor.py`)**: converts rollout events/segments into `SemanticPrimitive` records carrying provenance.
- **Ontology Update Engine (`src/sima2/ontology_update_engine.py`)**: maps primitives to ontology proposals; provenance threaded into proposal metadata.
- **Semantic Tag Propagator (`src/sima2/semantic_tag_propagator.py`)**: generates enrichment proposals using segmentation outputs, ontology proposals, and task graph proposals.

## Configuration & Provenance
- Central SIMA-2 config: `config/sima2.yaml` (backend mode/id/version, task distributions, segmentation thresholds).
- Provenance helper: `src/sima2/config.py` (load config, build/extract provenance).
- Provenance fields appear on rollouts, primitives, datapacks/episodes (`sima2_backend_id`, `sima2_model_version`, `sima2_task_spec`, `sima2_backend_mode`).

## Artifacts & Outputs
- Segmentation/primitive/tag outputs: produced by pipeline runners (e.g., `scripts/run_stage2_sima2_pipeline.py`, `scripts/stress_test_sima2_pipeline.py`) under `results/`.
- TrustMatrix: `results/sima2/trust_matrix.json`; loader in `src/analytics/econ_correlator.py`; injected into policies via `src/policies/registry.py`.
- Stress policy datasets: `results/policy_datasets_sima2_stress/` (created by stress script with `--emit-policy-datasets`).
- Contract smoke: `scripts/smoke_test_sima2_contract.py` exercises client → segmenter → primitives → ontology → tags path.
