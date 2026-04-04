# RunPod Execution Plane

This document describes the RunPod GPU execution plane for the robotics-vp-core stack.

## Architecture: Three Execution Planes

| Plane | Hardware | Scope | Agent | Persistence |
|-------|----------|-------|-------|-------------|
| **Local** | CPU (developer laptop) | Lint, type-check, unit tests, lightweight smoke tests | Claude, human | Filesystem |
| **Codex cloud** | CPU (cloud sandbox) | Code-only parallel work: refactors, docs, test authoring, schema changes | Codex | Git branches |
| **RunPod** | GPU (rented) | Training, provider bring-up, episode collection, CUDA-dependent validation | Codex or human | Volumes + artifact sync |

The split is economic: local is free, Codex cloud is per-task, RunPod is per-GPU-hour. Use the cheapest plane that can do the job.

## Prerequisites

### Credentials

| Variable | Required | Purpose |
|----------|----------|---------|
| `RUNPOD_API_KEY` | Yes | Authentication with RunPod API |
| `RUNPOD_VOLUME_ID` | For `loop`/`train` | Persistent network volume for data/checkpoints |
| `RUNPOD_TEMPLATE_ID` | Optional | Pre-built pod template (defaults to `Dockerfile.runpod`) |
| `RUNPOD_POD_TIMEOUT` | Optional | Default pod timeout in seconds (overridable per launch) |

### Tooling

- `runpodctl` CLI installed and authenticated
- `rsync` available locally (for sync scripts)
- SSH key registered with RunPod (for `exec_remote.sh` and sync scripts)

Verify with:

```bash
./scripts/runpod/ensure_cli.sh
```

## Classification Axes

A run should be classified along two axes:

### 1. Run Class

This answers: **what kind of machine work is this?**

- `loop`
- `provider`
- `train`
- `refactor`

### 2. Epistemic Status

This answers: **what inferential weight should this run carry?**

- `smoke`
- `proof_of_life`
- `benchmark_candidate`
- `promotion_candidate`
- `deployment_candidate`

A run is not fully described by `pod_class` alone. A `train` + `proof_of_life` run is not benchmark-credible. A `provider` + `smoke` run is just bring-up. A `promotion_candidate` run should normally feed a comparison artifact before any promotion claim is made.

If `epistemic_status` is omitted, interpret the run conservatively as no stronger than `smoke`.

## Pod Classes

| Class | GPU | Volume | Lifetime | Work | Artifacts |
|-------|-----|--------|----------|------|-----------|
| `loop` | A40 / L40 | Persistent | Hours | Episode collection, replay generation | Replay files, env logs |
| `provider` | A100-40GB+ | None | Minutes-hours | Provider smoke tests (SAM 3, V-JEPA 2, DINOv2, Depth Anything) | Smoke logs, latency benchmarks |
| `train` | A100-80GB | Persistent | Hours-days | SAC training, seam training, policy training | Checkpoints, training logs |
| `refactor` | A40+ | None | Minutes | CUDA/Isaac build validation | Build logs, test output |

## Workflow

### 1. Launch

```bash
./scripts/runpod/launch_pod.sh --class train --gpu A100-80GB --timeout 14400
```

This creates the pod and writes metadata to `.agent/runs/runpod-<timestamp>/meta.json`.

### 2. Sync

```bash
./scripts/runpod/sync_up.sh --pod <pod_id>
```

Pushes the repo (respecting `.gitignore`) to `/workspace/` on the pod.

### 3. Execute

```bash
./scripts/runpod/exec_remote.sh --pod <pod_id> -- pip install -r requirements-gpu.txt
./scripts/runpod/exec_remote.sh --pod <pod_id> -- python train_sac.py --config configs/sac/default.yaml
```

Stdout/stderr are captured to `.agent/runs/runpod-<run_id>/`.

### 4. Collect results

```bash
./scripts/runpod/sync_down.sh --pod <pod_id> \
  --remote-path /workspace/results \
  --local-path results/run_registry/<run_id>/
```

### 5. Record billing

```bash
./scripts/runpod/collect_billing.sh --pod <pod_id>
```

Appends a cost snapshot to the run manifest.

### 6. Cleanup

```bash
./scripts/runpod/cleanup_idle.sh
```

Or stop a specific pod: `runpodctl stop pod <pod_id>`.

## Run Manifests and Receipts

Every remote run produces a manifest at `.agent/runs/<run_id>/manifest.json`. The schema is documented in [run_manifest_schema.md](run_manifest_schema.md).

Manifests make remote runs agent-legible: any agent can inspect what ran, what it produced, what inferential weight it should carry, and what it cost.

At minimum, completed runs should preserve:

- `gpu_class`
- `wall_clock_seconds`
- `estimated_cost_usd`
- `artifact_size_bytes`
- `storage_or_checkpoint_size_bytes`
- `justified_itself`

Example manifests:

- `configs/runpod/examples/train_sac_manifest.json`
- `configs/runpod/examples/provider_bringup_manifest.json`
- `configs/runpod/examples/benchmark_candidate_training_manifest_v2.json`

## Comparison Artifacts

Launching runs is not the hard part. Comparing them honestly is.

Meaningful run families should eventually produce a comparison artifact such as `results/run_registry/templates/run_comparison_template.md` with fields for:

- baseline
- candidate run(s)
- what changed
- what improved
- what regressed
- confidence level
- promotion implication
- roadmap implication
- next recommended action

A `benchmark_candidate` run should normally have a named baseline. A `promotion_candidate` run should not be treated as promotion-credible without a comparison artifact or equivalent receipt.

## Cost Management

1. **Set timeouts**: Every `launch_pod.sh` invocation should include `--timeout`. Defaults are set per pod class but can be overridden.
2. **Collect billing after runs**: `collect_billing.sh` appends cost snapshots to the manifest so spend is tracked per task.
3. **Clean up idle pods**: `cleanup_idle.sh` identifies pods that have been idle beyond a threshold and offers to stop them.
4. **Prefer short-lived pods**: Use `provider` and `refactor` classes for quick validation, then terminate. Only `loop` and `train` justify long-lived pods.
5. **Volume pruning**: Persistent volumes accumulate data. Periodically review and prune old checkpoints and replay buffers.

## Queue Prioritization Posture

As concurrent GPU windows multiply, runs should be sortable by more than timestamp.

Prioritization-ready manifests should gradually expose:

- `wm`
- `subsystem`
- `blocker`
- `run_class`
- `epistemic_status`
- `expected_value`
- `estimated_cost_usd`
- `dependency_chain`
- `urgency`

This does not require a scheduler right now. It only requires preserving the fields needed for later scheduling discipline.

## Decision Tree

```
Is the task code-only (no GPU needed)?
  YES --> Does it benefit from parallel execution?
    YES --> Codex cloud
    NO  --> Local
  NO  --> Does it need GPU?
    YES --> What kind of work?
      Training (SAC, seam, policy)      --> RunPod: train class
      Provider smoke (SAM, V-JEPA, ...) --> RunPod: provider class
      Episode collection / replay       --> RunPod: loop class
      CUDA/Isaac build validation       --> RunPod: refactor class
    NO  --> Local (even if slow, it's cheaper)
```

## Manual Steps That Cannot Be Automated

The following require human action and cannot be performed by agents:

1. **RunPod account creation and billing setup** — requires payment method
2. **API key generation** — must be done in the RunPod web console
3. **Network volume creation** — initial volume provisioning in a specific region
4. **SSH key registration** — uploading your public key to RunPod settings
5. **Template creation from Dockerfile** — first-time setup of `Dockerfile.runpod` as a RunPod template
6. **Cost approval for large runs** — agents should report estimated cost; humans approve

## Stage-Appropriate Philosophy

The intended posture remains:

- enough structure for recurring multi-GPU work
- comparison-friendly and decision-oriented records
- no fake automation
- no premature platform building

The thin-wrapper + manifest + registry + skill model remains correct.

## Related Documents

- [RunPod GPU Execution Skill](../../codex_skills/runpod-gpu-execution/SKILL.md)
- [Run Manifest Schema](run_manifest_schema.md)
- [Run Comparison Template](../../results/run_registry/templates/run_comparison_template.md)
- [Example: SAC Training Manifest](../../configs/runpod/examples/train_sac_manifest.json)
- [Example: Provider Bring-Up Manifest](../../configs/runpod/examples/provider_bringup_manifest.json)
- [Example: Benchmark Candidate Manifest](../../configs/runpod/examples/benchmark_candidate_training_manifest_v2.json)
