# Run Manifest Schema

## Purpose

Every remote execution run (RunPod, Codex cloud, or any non-local plane) produces a manifest file. The manifest makes the run agent-legible: any agent or human can inspect what ran, on what code, with what config, what it produced, and what it cost.

Manifests are the primary mechanism by which remote GPU work becomes traceable and auditable.

## Location

```
.agent/runs/<run_id>/manifest.json
```

Run IDs follow the pattern: `<mode>-<YYYYMMDD>-<HHMMSS>-<hex>`.

Examples:
- `runpod-20260901-120000-abc123`
- `codex-cloud-20260901-140000-def456`

## Schema

```json
{
  "run_id": "string — unique identifier, matches the directory name",
  "mode": "string — one of: local, codex_cloud, runpod",
  "pod_class": "string | null — RunPod pod class: loop, provider, train, refactor. Null for non-RunPod runs.",
  "commit_sha": "string — short git SHA of the code that was executed",
  "branch": "string — git branch at launch time",
  "task": "string — human-readable description of what this run does",
  "config_paths": ["string — paths to config files used, relative to repo root"],
  "seeds": ["integer — random seeds used for reproducibility"],
  "image": "string — container image or base environment",
  "template": "string — RunPod template name or ID",
  "pod_id": "string | null — RunPod pod ID, null before launch",
  "volume_id": "string | null — RunPod volume ID if persistent storage is attached",
  "commands": ["string — ordered list of commands executed on the remote"],
  "artifact_paths": ["string — paths (remote or local) where artifacts are stored"],
  "status": "string — one of: pending, running, completed, failed",
  "started_at": "string | null — ISO 8601 timestamp when execution began",
  "finished_at": "string | null — ISO 8601 timestamp when execution ended",
  "cost_snapshot": "object | null — billing data collected after the run",
  "rollback_notes": "string — notes on how to revert if results are bad",
  "replay_notes": "string — notes on how to reproduce this run"
}
```

### Field Details

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | string | Yes | Unique run identifier. Format: `<mode>-<YYYYMMDD>-<HHMMSS>-<hex>` |
| `mode` | string | Yes | Execution plane: `local`, `codex_cloud`, `runpod` |
| `pod_class` | string/null | RunPod only | Pod class: `loop`, `provider`, `train`, `refactor` |
| `commit_sha` | string | Yes | Git short SHA at launch time |
| `branch` | string | Yes | Git branch at launch time |
| `task` | string | Yes | Human-readable task description |
| `config_paths` | array | Yes | Config files used (may be empty) |
| `seeds` | array | Yes | Random seeds (may be empty if not applicable) |
| `image` | string | Yes | Container image or environment |
| `template` | string | Yes | Template name or ID |
| `pod_id` | string/null | RunPod only | Pod ID, null until pod is created |
| `volume_id` | string/null | If volume used | Network volume ID |
| `commands` | array | Yes | Commands executed in order |
| `artifact_paths` | array | Yes | Where outputs are stored |
| `status` | string | Yes | One of: `pending`, `running`, `completed`, `failed` |
| `started_at` | string/null | When running | ISO 8601 UTC timestamp |
| `finished_at` | string/null | When done | ISO 8601 UTC timestamp |
| `cost_snapshot` | object/null | After billing | Cost data from `collect_billing.sh` |
| `rollback_notes` | string | Yes | How to revert (empty string if N/A) |
| `replay_notes` | string | Yes | How to reproduce (empty string if N/A) |

### `cost_snapshot` Object

```json
{
  "collected_at": "ISO 8601 timestamp",
  "gpu_hours": "number or null",
  "estimated_cost_usd": "number or null",
  "note": "string — freeform note about billing source"
}
```

## How Agents Should Read Manifests

1. **Find runs**: List `.agent/runs/runpod-*/manifest.json` to find RunPod runs.
2. **Filter by status**: Check `status` field to find completed or failed runs.
3. **Check artifacts**: Use `artifact_paths` to locate outputs.
4. **Trace code**: Use `commit_sha` and `branch` to identify what code produced the run.
5. **Check cost**: Read `cost_snapshot` to understand spend.

## How Agents Should Write Manifests

1. Create the manifest with `status: "pending"` before launching the pod.
2. Update `status` to `"running"` and set `started_at` when execution begins.
3. Update `status` to `"completed"` or `"failed"` and set `finished_at` when done.
4. Run `collect_billing.sh` and let it append `cost_snapshot`.
5. Fill in `rollback_notes` and `replay_notes` after reviewing results.

## Examples

- SAC training: `configs/runpod/examples/train_sac_manifest.json`
- Provider bring-up: `configs/runpod/examples/provider_bringup_manifest.json`
