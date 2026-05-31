# Economic WM Post-Gap Readiness

[ad-hoc note]

This document records the planning-only readiness layer for the post-gap items
in `2026-05-25-cpu-capable-august-gap-items.md`. It covers GPU day-one
runbooks, dataset/corpus prep, benchmark gates, provider packaging, replay-loop
prep, G1/R1 purchase readiness, and evidence hygiene before GPU/provider/data or
hardware access exists.

Compiler:

```bash
python3 scripts/economic_world_model/compile_post_gap_readiness.py \
  --output-dir artifacts/economic_world_model/post_gap_readiness
```

Primary artifacts:

- `post_gap_readiness_report_v1.json`
- `post_gap_readiness_v1.md`
- `gpu_day_one_runbooks_v1.jsonl`
- `external_dataset_corpus_plan_v1.jsonl`
- `corpus_prep_artifact_plans_v1.jsonl`
- `benchmark_gate_specs_v1.jsonl`
- `provider_runtime_packaging_specs_v1.jsonl`
- `perception_embodiment_replay_loop_specs_v1.jsonl`
- `g1_r1_purchase_readiness_v1.jsonl`
- `evidence_hygiene_specs_v1.jsonl`

## Status Boundary

This layer is not an execution claim.

- No external datasets were downloaded.
- No provider was launched.
- No GPU training was run.
- No robot was purchased or operated.
- No model was promoted.
- No Phase 7 concepts or authority surfaces were expanded.

The report is ready when every post-gap checklist item has an explicit typed
plan, receipt shape, fail-closed gate, and remaining blocker.

## External Datasets To Bring In

The current corpus plan identifies these source families for staged import:

| Dataset | Priority | Use | Import posture |
| --- | --- | --- | --- |
| Open X-Embodiment / RT-X | P0 | broad real robot manipulation and cross-embodiment rows | planned manifest only; per-source license and storage review required |
| DROID | P0 | in-the-wild manipulation, scene diversity, language/task labels | planned manifest only; large download and calibration/version review required |
| BridgeData V2 | P0 | kitchen/tabletop/drawer-style manipulation | planned manifest only; source adapter and license review required |
| Hugging Face LeRobot curated datasets | P0 | small adapter fixtures and standardized format validation | planned manifest only; dataset-by-dataset license review required |
| RoboMIND | P1 | humanoid/multi-embodiment transfer watchlist | planned manifest only; source/schema review required |
| RH20T | P1 | contact-rich force/audio/multimodal manipulation | sensitive planned manifest only; multi-TB storage and privacy review required |
| Ego4D / Ego-Exo4D | P2 | perception-only human-object video priors | perception-only; no robot action truth |
| AgiBot World | P2 | humanoid/manipulation watchlist | source/terms/schema review required before import |
| Local robotics-vp-core artifacts | P0 | fixture replay, receipts, event spine, Economic WM shadow rows | local manifest ready, not training-grade |

Every dataset row has planned normalization steps, train/eval split policy,
replay indexer keys, data-quality receipt checks, label-gap ledger entries,
false-veto/false-allow label specs, and transport/meta-node training-corpus
mapping. External rows remain `ready_for_training=false` until the actual
download, license review, digesting, schema conversion, and quality receipts are
complete.

## GPU Day-One Runbooks

The runbooks cover:

- RunPod provider proof-of-life for the first hour.
- RunPod Economic WM GPU shape run for the first eight hours.
- RunPod weekend corpus and benchmark candidate.
- Local Linux runtime preflight.
- Codex cloud code-only readiness audit.

Each runbook carries provider bring-up commands, verification commands, expected
artifacts, failure receipts, cost/time estimates, checkpoint paths, artifact
storage paths, and explicit stop conditions. Each remains
`launch_allowed=false` until a real non-stub provider command replaces the guard
and a real run manifest is recorded.

## Benchmark Gates

The gate specs are fail-closed until evidence exists:

- transport eval acceptance thresholds
- perception replay consistency metrics
- command/timing/safety benchmark reports
- economic allocation shadow benchmarks
- Phase 7 governance outcome scoring
- promotion gate evidence

The Phase 7 gate consumes existing shadow outcome labels only. It does not add
new Phase 7 vocabulary.

## G1/R1 Purchase Readiness

The purchase-readiness specs cover variant decision criteria, workspace safety,
e-stop and recovery, network/DDS, companion compute, camera/sensor mounting,
storage/logging, calibration, first-week bring-up, and do-not-run-until gates.

The current decision posture is:

- prefer Unitree G1 developer/EDU-style variants when secondary development,
  onboard developer compute, sensor depth/LiDAR, and hand options are required;
- treat R1 developer variants as lower-cost experimentation only if vendor
  confirms secondary-development API and sensor access;
- do not treat any consumer or non-developer configuration as SDK/control proof
  without vendor confirmation.

The purchase rows are planning receipts only. Hardware-grade proof still needs a
robot or honest sim runtime, safety inspection, calibration, and command echo
receipts.

## Evidence Hygiene

The hygiene specs cover nightly audit hardening, artifact retention, remote-run
manifests, stale artifact detection, claim-vs-evidence checking, focused CI
suites, and this readiness report generator.

Claims remain invalid unless the matching receipt exists. In particular,
GPU/provider/hardware/promotion claims require provider truth, GPU runtime,
hardware or honest sim, benchmark evidence, and run manifests.
