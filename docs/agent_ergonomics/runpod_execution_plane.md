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

## Run Manifests

Every remote run produces a manifest at `.agent/runs/<run_id>/manifest.json`. The schema is documented in [run_manifest_schema.md](run_manifest_schema.md).

Manifests make remote runs agent-legible: any agent can inspect what ran, what it produced, and what it cost.

## Cost Management

1. **Set timeouts**: Every `launch_pod.sh` invocation should include `--timeout`. Defaults are set per pod class but can be overridden.
2. **Collect billing after runs**: `collect_billing.sh` appends cost snapshots to the manifest so spend is tracked per task.
3. **Clean up idle pods**: `cleanup_idle.sh` identifies pods that have been idle beyond a threshold and offers to stop them.
4. **Prefer short-lived pods**: Use `provider` and `refactor` classes for quick validation, then terminate. Only `loop` and `train` justify long-lived pods.
5. **Volume pruning**: Persistent volumes accumulate data. Periodically review and prune old checkpoints and replay buffers.

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNPOD_API_KEY` | (none) | Required. RunPod API key for authentication. |
| `RUNPOD_VOLUME_ID` | (none) | Network volume ID for persistent storage. Required for `loop` and `train`. |
| `RUNPOD_TEMPLATE_ID` | (none) | Pod template ID. If unset, scripts use `Dockerfile.runpod` as the image reference. |
| `RUNPOD_POD_TIMEOUT` | `14400` | Default pod timeout in seconds (4 hours). |

## Manual Steps That Cannot Be Automated

The following require human action and cannot be performed by agents:

1. **RunPod account creation and billing setup** — requires payment method
2. **API key generation** — must be done in the RunPod web console
3. **Network volume creation** — initial volume provisioning in a specific region
4. **SSH key registration** — uploading your public key to RunPod settings
5. **Template creation from Dockerfile** — first-time setup of `Dockerfile.runpod` as a RunPod template
6. **Cost approval for large runs** — agents should report estimated cost; humans approve

## Related Documents

- [RunPod GPU Execution Skill](../../codex_skills/runpod-gpu-execution/SKILL.md)
- [Run Manifest Schema](run_manifest_schema.md)
- [Example: SAC Training Manifest](../../configs/runpod/examples/train_sac_manifest.json)
- [Example: Provider Bring-Up Manifest](../../configs/runpod/examples/provider_bringup_manifest.json)
