# Pre-Economic-WM Integration Stage

Date: 2026-05-20

## Stage name

This stage is the **Post-Phase-3 / Pre-Economic-WM Integration Stage**. It sits after local structural closure of Perception / Grounding, Sim / Synth / Physics, and Embodiment / Actuation, and before any new Economic WM implementation or training claim.

## Purpose

The job of this stage is to make lower-WM outputs replayable, exportable, and trainable without bespoke joins. The Economic WM should not be asked to learn allocation from scattered sidecars; it should consume stable packet, evidence, governance, value, embodiment, and dataset-bridge surfaces.

## What must be true before Economic WM starts

1. **Stage-1 governed video emits complete supervision refs**: runtime packets, event spine rows, decision ledger rows, governance traces, counterfactual evals, value target packs, value-ledger receipts, reconstruction sidecars, and explicit teacher-runtime sidecars.
2. **Fallbacks are explicit**: unavailable teacher runtimes, heuristic video diffusion, missing calibration, passthrough SceneTracks, and shadow-only benchmark posture must be represented as typed evidence rather than silent success.
3. **Replay import preserves the refs**: governed-video admission logs and rollout bundles must carry runtime/evidence/governance/value/reconstruction/teacher refs into canonical replay records.
4. **Public dataset bridges preserve internal sidecar pointers**: RLDS and LeRobot exports are lossy adapters, but they must keep internal sidecar refs so richer repo-native training remains possible.
5. **No economic scalar takeover**: value targets and counterfactuals may become Economic WM supervision, but they do not rewrite frozen reward math, trust-net, `w_econ`, or lambda-controller equations.

## Landed local tranche

The first 2026-05-20 integration pass added Stage-1 teacher-runtime contract/action/trace sidecars, stopped fabricating calibration refs in default reconstruction sidecars, threaded reconstruction/teacher refs through replay discovery, and added a governed-video bridge export script that writes canonical replay plus RLDS and LeRobot rows while preserving internal sidecar refs. The follow-up pass added `reconstruction_grounding_report_v1` so calibration class, grounding class, missing refs, training eligibility, and benchmark readiness are explicit sidecars rather than inferred from reconstruction-file presence. The next local pass tightened Stage-1 benchmark gates so real SceneTracks and real vision cannot become benchmark-ready unless camera calibration is present before sidecar emission. Replay/bridge exports now preserve benchmark-gate and future-training truth alongside internal sidecar refs. A local Stage-1 bridge-readiness sweep now exercises five manifest shapes across Stage-1, canonical replay, RLDS, and LeRobot exports. The Economic WM entry preflight now reports `ready_for_scaffold=true` and `ready_for_training=false` from that evidence, the first native Economic WM scaffold emits deterministic `economic_state_v1`, `allocation_envelope_v1`, and `economic_wm_scaffold_report_v1` artifacts, the first row materializer emits `economic_wm_replay_feature_row_v1` / `economic_wm_training_corpus_manifest_v1` from Stage-1 proposal admissions, and the first shadow allocation eval ranks local allocation candidates without executing them.

## Remaining before Economic WM

- Keep the Stage-1 bridge-readiness sweep green as new manifest shapes are added; the current local sweep covers calibrated real, top-level calibration refs, missing calibration, unknown artifact refs, and passthrough fallback cases.
- Keep tightening real SceneTracks and calibration truth at ingestion/runner boundaries; Stage-1 now has explicit reconstruction grounding reports and calibration-aware benchmark gates, but broader ingestion paths still need the same discipline.
- Economic WM scaffold, local row materialization, and shadow allocation evals may continue behind the entry preflight; training remains blocked until GPU/provider bring-up, non-stub teacher/runtime evidence, and promotion-grade benchmark evidence exist.
- GPU/provider work remains external: non-stub teacher runtime, V-JEPA/DreamGen-style video-state training, SAM3D-scale grounding, and promotion-grade benchmark runs.
