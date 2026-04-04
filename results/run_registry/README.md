# Run Registry

Central location for run manifests and artifacts from remote execution.

## Purpose

The run registry stores outputs from RunPod GPU runs, organized by run ID. Each run directory contains a manifest and any artifacts synced down from the remote pod.

This is the canonical place to look for training checkpoints, provider smoke test results, episode replay data, and other artifacts produced by GPU-backed execution.

## Directory Structure

```
results/run_registry/
  <run_id>/
    manifest.json      # Run manifest (see docs/agent_ergonomics/run_manifest_schema.md)
    checkpoints/       # Training checkpoints (if applicable)
    logs/              # Training or execution logs
    metrics/           # Evaluation metrics, reward curves
    provider_output/   # Provider smoke test outputs
    ...                # Other artifacts specific to the run
```

## Querying Past Runs

Agents and humans can find runs by:

```bash
# List all runs
ls results/run_registry/

# Find runs by pod class
grep -rl '"pod_class": "train"' results/run_registry/*/manifest.json

# Find completed runs
grep -rl '"status": "completed"' results/run_registry/*/manifest.json

# Find runs on a specific branch
grep -rl '"branch": "main"' results/run_registry/*/manifest.json
```

## Relationship to `.agent/runs/`

- **`.agent/runs/`** stores run metadata (meta.json, stdout/stderr logs) for all execution — including Codex cloud runs, RunPod launch metadata, and exec logs. This is the operational record.
- **`results/run_registry/`** stores artifacts and final manifests from remote runs. This is the artifact archive.

The two are complementary: `.agent/runs/` tells you what happened during execution; `results/run_registry/` holds what was produced.
