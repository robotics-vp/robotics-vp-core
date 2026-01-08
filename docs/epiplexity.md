# Epiplexity / Prequential-MDL

## Purpose
Epiplexity is a compute-bounded proxy for learnable structure. Under a fixed compute budget, the system separates:
- `S_T_proxy`: learnable structure (prequential improvement)
- `H_T_proxy`: residual entropy (final NLL proxy)

These diagnostics are advisory: they inform data valuation, representation selection, and orchestrator scheduling without changing reward math.

## Key Components
- `EpiplexityTracker`: runs probe learners, caches results, and returns `S_T_proxy`, `H_T_proxy`, and `epi_per_flop`.
- `PrequentialAUCLossEstimator`: MVP estimator using area-under-loss-curve proxy.
- `TokenizerAblationHarness`: compares representations on the same dataset slice and writes leaderboards.

## Datapack Metadata
Epiplexity results are stored in datapack metadata:
- `epiplexity[repr][budget][seed]`: per-run metrics and version hashes
- `epiplexity_summary[repr][budget]`: mean/std/confidence summary
- `epiplexity_summary._default`: default repr/budget selector for downstream use

By default, only summaries are attached to datapacks; full per-run details live in the cache (`artifacts/epiplexity_cache/`). To store full runs in datapack metadata, pass `--store-full-runs` to the CLI (debug only).

## CLI
Run a synthetic evaluation:

```bash
python -m scripts.run_epiplexity_eval --synthetic --dataset-slice-id demo_slice
```

Custom inputs can be provided via `--episode-jsonl` (each line is a JSON dict with token keys).

For deterministic probe runs:

```bash
VPE_DETERMINISTIC=1 VPE_DETERMINISTIC_SEED=0 python -m scripts.run_epiplexity_eval --synthetic
```

## Curated Slices
To evaluate geometry on targeted slices (occluded / dynamic / static), use the curated-slice runner:

```bash
python -m scripts.run_epiplexity_curated_slices --datapack-dir /path/to/datapacks --task drawer_vase
```

This compares `vision_rgb`, `geometry_scene_graph`, `geometry_bev`, and `canonical_tokens` under a fixed compute budget.

**Raw vs portable datapacks:** Curated epiplexity slices run in one of two modes. If `raw_data_path` is present and accessible, the runner rehydrates raw streams (RGB, scene tracks) and computes slices directly. If raw data is absent but portable artifacts are embedded (`scene_tracks_v1`, `rgb_features_v1`, `slice_labels_v1`), the runner operates in portable mode, consuming stored artifacts without raw rehydration. If neither raw data nor portable artifacts are available, the runner fails fast with an explicit diagnostic.

If raw streams are not available, curated slices can run on portable datapacks that embed `scene_tracks_v1`, `rgb_features_v1`, and `slice_labels_v1`. Use the exporter to generate them:

```bash
python -m scripts.export_portable_datapacks --datapack-dir /path/to/datapacks --task drawer_vase
```

For a synthetic smoke run:

```bash
python -m scripts.run_epiplexity_curated_slices --synthetic
```

## Orchestrator Hook
When enabled (`config/pipeline.yaml`):
- `orchestrator.use_epiplexity_term = true`
- `orchestrator.epi_alpha` scales the advisory term
- `orchestrator.epi_budget_id` selects which compute budget to read
- `orchestrator.epi_baseline_repr` selects the baseline representation

The semantic orchestrator surfaces this as an advisory scheduling term.
