# External LeRobot Corpus Import

This is the first real external-corpus import proof after the post-gap
readiness manifest. It downloads a small LeRobot-format Parquet dataset slice,
converts selected rows into repo-native replay records, and emits the split,
index, quality, label-gap, governance-label, and Economic WM shadow-ingestion
artifacts that the readiness plan called for.

Command:

```bash
python3 scripts/economic_world_model/import_lerobot_corpus_slice.py \
  --repo-id lerobot/pusht_keypoints \
  --output-dir artifacts/economic_world_model/external_lerobot_import \
  --max-episodes 3 \
  --max-steps-per-episode 200
```

Primary artifact:

- `artifacts/economic_world_model/external_lerobot_import/external_lerobot_corpus_import_report_v1.json`

Current result:

- `status=ok_external_corpus_slice_imported_shadow_only`
- `dataset_id=lerobot/pusht_keypoints`
- `source_total_bytes=2253085`
- `selected_episode_count=3`
- `selected_step_count=420`
- `replay_episode_count=3`
- `replay_step_count=420`
- `quality_receipt_count=11`
- `quality_passed_count=11`
- `label_gap_count=5`
- `governance_label_count=4`
- `ingestion_row_count=1`
- `ready_for_shadow_eval=true`
- `ready_for_training=false`
- `provider_executed=false`
- `gpu_training_executed=false`
- `unitree_hardware_truth=false`
- `promotion_eligible=false`
- `phase7_authority_granted=false`

Artifacts emitted:

- `external_lerobot_rows.jsonl`
- `replay_dataset/manifest.json`
- `replay_dataset/episodes.jsonl`
- `replay_dataset/steps.jsonl`
- `replay_dataset/windows.jsonl`
- `train_eval_split_manifest.json`
- `replay_index.jsonl`
- `data_quality_receipts.jsonl`
- `label_gap_ledger.jsonl`
- `governance_label_specs.jsonl`
- `economic_wm_external_corpus_ingestion_rows.jsonl`

## Boundary

This is a real external corpus import, but it remains shadow-only. It proves
download, source digests, Parquet decoding, LeRobot row conversion, repo-native
replay export, split/index generation, quality receipts, label-gap ledger,
false-veto/false-allow label specs, and Economic WM shadow ingestion.

It does not prove Unitree hardware behavior, provider execution, GPU training,
perception video handling, benchmark-grade corpus scale, reward-math mutation,
promotion, or Phase 7 authority.
