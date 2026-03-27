# Epiplexity / Prequential-MDL

## Purpose
Epiplexity is a compute-bounded proxy for learnable structure. Under a fixed compute budget, the system separates:
- `S_T_proxy`: learnable structure (prequential improvement)
- `H_T_proxy`: residual entropy (final NLL proxy)

These diagnostics currently operate as bounded inferential overlays: they inform data valuation, representation selection, replay weighting, and orchestrator scheduling without changing legacy reward math.

The target posture is narrower than "purely advisory forever":

- full probe runs and experimental diagnostics should remain sidecars
- portable epiplexity summaries should graduate into canonical learnability metadata
- learnability class should later help drive training admission, adaptation work orders, and promotion evidence

So the current repo posture is:

- advisory / bounded today
- candidate canonical learnability class later

## Key Components
- `EpiplexityTracker`: runs probe learners, caches absolute runs, records compute accounting (`flops_estimate`), and derives baseline-relative deltas only at read time.
- `PrequentialAUCLossEstimator`: area-under-loss-curve proxy with deterministic compute estimation.
- `RequentialEstimator`: online evaluate-then-update variant for nonzero requential scoring.
- `TokenizerAblationHarness`: compares representations on the same dataset slice and writes leaderboards.

## Datapack Metadata
Epiplexity results are stored in datapack metadata:
- `epiplexity[repr][budget][seed]`: per-run metrics and version hashes
- `epiplexity_summary[repr][budget]`: mean/std/confidence summary
- `epiplexity_summary._default`: default repr/budget selector for downstream use
- `data/datapacks/epiplexity_overlays.jsonl`: additive overlay sidecar loaded automatically by `DataPackRepo`

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
When run against a datapack repo, it also emits `epiplexity_overlays.jsonl` beside the datapacks so repo reloads, samplers, and replay-side advisory consumers can see the same canonical summaries.

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
- `orchestrator.epi_repr_id` optionally pins a representation; otherwise the datapack’s `_default` selector is used

The semantic orchestrator currently surfaces this as a bounded scheduling term.

Longer term, the economic-world-model roadmap intends epiplexity to do more than scheduling. Once replay/datapack/train-manifest contracts carry an explicit learnability class, the same signal should be able to participate in:

- training admissibility
- adaptation / recollection work orders
- sim/synth agenda ranking
- promotion evidence
