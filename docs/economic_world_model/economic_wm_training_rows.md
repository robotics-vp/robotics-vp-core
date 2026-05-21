# Economic WM Training Rows

Date: 2026-05-21

## Purpose

This pass materializes local Economic WM row surfaces from the scaffold and Stage-1 proposal-admission evidence. The rows are deterministic feature/target records for future Economic WM training and evaluation; they are not a training run.

The row corpus emits:

- `economic_wm_replay_feature_rows_v1.jsonl`: one row per Stage-1 admission with benchmark/shadow truth, typed feature vectors, target vectors, sidecar refs, and denied-promotion reasons.
- `economic_wm_training_corpus_manifest_v1.json`: corpus counts, scaffold linkage, blocker posture, and artifact refs.
- `economic_wm_training_corpus_manifest_v1.md`: human-readable summary.

## Executable path

Run from an existing scaffold report:

```bash
python3 scripts/economic_world_model/materialize_economic_wm_training_rows.py \
  --output-dir artifacts/economic_world_model/economic_wm_training_rows \
  --scaffold-report artifacts/economic_world_model/economic_wm_scaffold/economic_wm_scaffold_report_v1.json
```

If no scaffold report is supplied, the script can run the scaffold builder first. That may also run the entry preflight and Stage-1 bridge-readiness sweep.

## Current local result

The current local corpus has:

- `row_count=5`
- `benchmark_ready_count=2`
- `shadow_only_count=3`
- `ready_for_training=false`
- `promotion_eligible=false`

Rows preserve the same benchmark vs shadow split as the Stage-1 bridge-readiness sweep. Benchmark-ready rows receive benchmark/reconstruction target weights. Shadow-only rows receive gap-collection target weights and blocked-precondition reasons.

## Boundary

These rows are allowed to support local feature extraction, training-corpus inspection, and future trainer/evaluator development.

They do not:

- run GPU training
- bring up a provider
- promote an Economic WM
- mutate reward, trust-net, `w_econ`, or lambda-controller math
- treat external teacher/provider outputs as native truth
