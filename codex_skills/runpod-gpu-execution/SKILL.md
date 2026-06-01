---
name: runpod-gpu-execution
description: Use when a task requires GPU-backed execution on RunPod — training runs, provider bring-up, workcell loop episodes, or heavy refactor validation against CUDA/Isaac stacks. Do not use for code-only work (use Codex cloud) or lightweight validation (use local).
type: execution
---

# RunPod GPU Execution

Use this skill when the task requires GPU hardware that is not available locally or on Codex cloud. RunPod is the GPU execution plane for this repo.

## When to Use Each Execution Plane

| Plane | Use When | Examples |
|-------|----------|----------|
| **Local** | No GPU needed, fast iteration, lightweight tests | `pytest`, `ruff`, `python3 -m compileall`, smoke tests without GPU |
| **Codex cloud** | Code-only parallel work, no GPU dependency | Refactors, documentation, test writing, schema work |
| **RunPod** | GPU required for training, inference, provider models, or CUDA-dependent validation | SAC training, SAM 3 bring-up, Isaac Lab builds, replay generation |

If in doubt: if the task needs `torch.cuda.is_available() == True` to succeed, use RunPod.

## Classification Axes

A run should be classified along two axes:

### Run Class

- `loop`
- `provider`
- `train`
- `refactor`

### Epistemic Status

- `smoke`
- `proof_of_life`
- `benchmark_candidate`
- `promotion_candidate`
- `deployment_candidate`

A run is not fully described by `pod_class` alone. If `epistemic_status` is omitted, interpret the run conservatively as no stronger than `smoke`.

## Pod Classes

### `loop` — Workcell Loop Runs

- **Work**: Episode collection, replay generation, workcell environment runs, data flywheel steps
- **GPU**: A40 or L40
- **Volume**: Persistent — replay data and episode buffers accumulate across runs
- **Lifetime**: Medium (hours). Stop when episode budget is exhausted.
- **Stays local**: Episode config authoring, replay analysis scripts, plotting
- **Artifacts**: Episode replay files, environment logs, feature extraction outputs
- **Receipts**: Episode count, total steps, reward summary statistics, disk usage on volume

### `provider` — Provider Bring-Up

- **Work**: Integration testing of perception providers (SAM 3/3.1, V-JEPA 2, DINOv2/SigLIP, Depth Anything V2)
- **GPU**: A100-40GB or higher (provider models need VRAM headroom)
- **Volume**: None required — short-lived pods for smoke testing
- **Lifetime**: Short (minutes to low hours). Terminate after smoke tests pass or fail.
- **Stays local**: Provider adapter code, seam wiring, contract schemas
- **Artifacts**: Smoke test logs, latency benchmarks, VRAM profiles, sample outputs
- **Receipts**: Pass/fail per provider mode, inference latency p50/p99, peak VRAM

### `train` — Training Runs

- **Work**: SAC training, seam training, policy training, imitation learning
- **GPU**: A100-80GB (training needs full memory for batch sizes)
- **Volume**: Persistent — checkpoints and training logs must survive pod restarts
- **Lifetime**: Long (hours to days). Use pod timeout to cap cost.
- **Stays local**: Config authoring, hyperparameter search design, checkpoint analysis
- **Artifacts**: Checkpoints, training logs, tensorboard events, evaluation metrics
- **Receipts**: Final loss, reward curves, episode count, wall-clock time, cost snapshot

### `refactor` — Heavy Refactor Validation

- **Work**: Validating refactors against GPU/CUDA dependency stacks (Isaac Lab, torch compile, CUDA builds)
- **GPU**: A40 or higher
- **Volume**: None required — ephemeral validation
- **Lifetime**: Short (minutes). Terminate after validation passes or fails.
- **Stays local**: The refactor itself (done on Codex cloud or local), only validation runs here
- **Artifacts**: Build logs, test output, compile diagnostics
- **Receipts**: Pass/fail, build time, test summary

## Prerequisites

1. **runpodctl installed**: `brew install runpod/runpodctl/runpodctl` or see https://github.com/runpod/runpodctl
2. **RUNPOD_API_KEY set**: `export RUNPOD_API_KEY="..."` — never commit this value
3. **Network volume configured** (for `loop` and `train` classes): set `RUNPOD_VOLUME_ID`
4. **Template configured** (optional): set `RUNPOD_TEMPLATE_ID` or use `Dockerfile.runpod` as the base

Verify readiness:

```bash
./scripts/runpod/ensure_cli.sh
```

## Workflow

1. **Prepare launch manifest**: `python3 scripts/runpod/prepare_launch_manifest.py --profile <provider_bringup|g1_loop_run|g1_sac_training> [--volume-id "$RUNPOD_VOLUME_ID"]`
2. **Launch**: `bash .agent/runs/<run_id>/launch_command.sh`
3. **Sync repo up**: `./scripts/runpod/sync_up.sh --pod <pod_id>`
4. **Execute**: `./scripts/runpod/exec_remote.sh --pod <pod_id> -- <command>`
5. **Sync results down**: `./scripts/runpod/sync_down.sh --pod <pod_id> --remote-path /workspace/results --local-path results/run_registry/<run_id>/`
6. **Collect billing**: `./scripts/runpod/collect_billing.sh --pod <pod_id>`
7. **Cleanup**: `./scripts/runpod/cleanup_idle.sh` or stop the pod manually

## Run Manifests

Every RunPod execution must produce a run manifest at `.agent/runs/runpod-<timestamp>/manifest.json`. See `docs/agent_ergonomics/run_manifest_schema.md` for the schema.

The manifest should record not only pod class and commands, but also epistemic status, cost/time fields, queue-prioritization fields where known, and whether the run justified itself.

Example manifests: `configs/runpod/examples/train_sac_manifest.json`, `configs/runpod/examples/provider_bringup_manifest.json`, `configs/runpod/examples/benchmark_candidate_training_manifest_v2.json`.

## Comparison Artifacts

Launching runs is not the hard part. Comparing them cleanly is.

Every meaningful run family should eventually yield a comparison artifact with fields like:

- `baseline`
- `candidate_runs`
- `what_changed`
- `what_improved`
- `what_regressed`
- `confidence_level`
- `promotion_implication`
- `roadmap_implication`
- `next_recommended_action`

Use `results/run_registry/templates/run_comparison_template.md` as the default shape.

## Verification Commands

```bash
# Check CLI and credentials
./scripts/runpod/ensure_cli.sh

# List active pods
runpodctl get pod

# Check billing
./scripts/runpod/collect_billing.sh

# Clean up idle pods
./scripts/runpod/cleanup_idle.sh --dry-run
```

## Safety Rules

1. **No secrets in commands**: Never pass API keys, tokens, or credentials as command-line arguments. Use environment variables on the pod.
2. **Cost awareness**: Always set `--timeout` on long-running pods. Check billing after runs. The `train` class defaults to a 4-hour timeout.
3. **Cleanup idle pods**: Run `./scripts/runpod/cleanup_idle.sh` after work sessions. Do not leave GPU pods idling overnight.
4. **Volume hygiene**: Persistent volumes accumulate data. Monitor disk usage and prune old checkpoints/replays.
5. **Commit before launch**: Always launch from a clean, committed state so the manifest `commit_sha` is meaningful.
6. **No force-push from pods**: Pods are for execution, not for git operations that modify remote history.

## Stage-Appropriate Philosophy

- enough structure for recurring multi-GPU work
- but not a giant orchestration platform
- not fake automation
- not premature serverless/platform building

The thin-wrapper + manifest + registry + skill posture remains the correct level.
