# Run Manifest Schema

## Purpose

Every remote execution run (RunPod, Codex cloud, or any non-local plane) produces a manifest file. The manifest makes the run agent-legible: any agent or human can inspect what ran, on what code, with what config, what it produced, what inferential weight it should carry, and what it cost.

Manifests are the primary mechanism by which remote GPU work becomes traceable and auditable.

## Location

```
.agent/runs/<run_id>/manifest.json
```

Run IDs follow the pattern: `<mode>-<YYYYMMDD>-<HHMMSS>-<hex>`.

Examples:
- `runpod-20260901-120000-abc123`
- `codex-cloud-20260901-140000-def456`

## Classification Axes

A run should be classified along two axes:

1. **Run class** — what kind of machine work is this?
   - `loop`
   - `provider`
   - `train`
   - `refactor`

2. **Epistemic status** — what inferential weight should this run carry?
   - `smoke`
   - `proof_of_life`
   - `benchmark_candidate`
   - `promotion_candidate`
   - `deployment_candidate`

`pod_class` remains the concrete RunPod-facing field for compatibility. `run_class` is the more general decision-facing label. If `epistemic_status` is omitted, interpret the run conservatively as no stronger than `smoke`.

## Schema

```json
{
  "run_id": "string — unique identifier, matches the directory name",
  "mode": "string — one of: local, codex_cloud, runpod",
  "pod_class": "string | null — RunPod pod class: loop, provider, train, refactor. Null for non-RunPod runs.",
  "run_class": "string | null — decision-facing class: loop, provider, train, refactor",
  "epistemic_status": "string | null — smoke, proof_of_life, benchmark_candidate, promotion_candidate, deployment_candidate",
  "commit_sha": "string — short git SHA of the code that was executed",
  "branch": "string — git branch at launch time",
  "task": "string — human-readable description of what this run does",
  "wm": "string | null — world model primarily served by the run",
  "subsystem": "string | null — subsystem primarily served by the run",
  "blocker": "string | null — blocker this run is intended to burn down",
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
  "gpu_class": "string | null — GPU class used for the run",
  "wall_clock_seconds": "number | null — elapsed runtime in seconds",
  "artifact_size_bytes": "number | null — total size of synced-down artifacts",
  "storage_or_checkpoint_size_bytes": "number | null — storage footprint materially retained from the run",
  "expected_value": "string | null — bounded EV note for prioritization",
  "estimated_cost_usd": "number | null — pre-run or post-run cost estimate",
  "dependency_chain": ["string — explicit dependencies or preconditions"],
  "urgency": "string | null — low, medium, high, critical",
  "justified_itself": "string | null — yes, no, unclear",
  "rollback_notes": "string — notes on how to revert if results are bad",
  "replay_notes": "string — notes on how to reproduce this run"
}
```

### Future Sim-to-Real Training References

The GR00T / VIRAL / DoorMan borrowing doctrine may later extend manifests
with optional references for teacher/student and sim-to-real runs. These are
future docs-only field reservations, not current required schema fields:

- `teacher_checkpoint_ref`
- `student_checkpoint_ref`
- `domain_randomization_profile_ref`
- `dataset_reset_profile_ref`
- `eval_export_gate_ref`

See
`docs/economic_world_model/doctrine_groot_visualsim2real_borrowings.md` for
the borrowing posture. The run manifest remains the ledger anchor; external
experiment directories, W&B runs, checkpoint files, and export artifacts should
attach to it rather than replacing it.

### Field Details

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | string | Yes | Unique run identifier. Format: `<mode>-<YYYYMMDD>-<HHMMSS>-<hex>` |
| `mode` | string | Yes | Execution plane: `local`, `codex_cloud`, `runpod` |
| `pod_class` | string/null | RunPod only | Pod class: `loop`, `provider`, `train`, `refactor` |
| `run_class` | string/null | Preferred | Decision-facing run class, usually matching `pod_class` |
| `epistemic_status` | string/null | Preferred | `smoke`, `proof_of_life`, `benchmark_candidate`, `promotion_candidate`, `deployment_candidate` |
| `commit_sha` | string | Yes | Git short SHA at launch time |
| `branch` | string | Yes | Git branch at launch time |
| `task` | string | Yes | Human-readable task description |
| `wm` | string/null | Optional | WM primarily served by the run |
| `subsystem` | string/null | Optional | Subsystem primarily served by the run |
| `blocker` | string/null | Optional | Blocker being burned down |
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
| `cost_snapshot` | object/null | After billing | Cost data from `collect_billing.sh` or equivalent |
| `gpu_class` | string/null | Preferred | GPU class used by the run |
| `wall_clock_seconds` | number/null | Preferred | End-to-end runtime |
| `artifact_size_bytes` | number/null | Preferred | Size of produced artifacts |
| `storage_or_checkpoint_size_bytes` | number/null | Preferred | Persistent storage or checkpoint footprint |
| `expected_value` | string/null | Optional | Bounded EV note for queue prioritization |
| `estimated_cost_usd` | number/null | Optional | Cost estimate or measured cost |
| `dependency_chain` | array | Optional | Declared upstream dependencies |
| `urgency` | string/null | Optional | `low`, `medium`, `high`, `critical` |
| `justified_itself` | string/null | Preferred | `yes`, `no`, `unclear` |
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

## Comparison Artifacts

Meaningful run families should eventually produce a comparison artifact, typically in `results/run_registry/templates/run_comparison_template.md` shape, with at least:

- `baseline`
- `candidate_runs`
- `what_changed`
- `what_improved`
- `what_regressed`
- `confidence_level`
- `promotion_implication`
- `roadmap_implication`
- `next_recommended_action`

A `promotion_candidate` run should not be treated as promotion-credible without a comparison artifact or equivalent receipt.

## How Agents Should Read Manifests

1. **Find runs**: List `.agent/runs/runpod-*/manifest.json` to find RunPod runs.
2. **Filter by status**: Check `status` field to find completed or failed runs.
3. **Check epistemic status**: Do not over-interpret smoke or proof-of-life runs.
4. **Check artifacts**: Use `artifact_paths` to locate outputs.
5. **Trace code**: Use `commit_sha` and `branch` to identify what code produced the run.
6. **Check cost/time**: Read `cost_snapshot`, `wall_clock_seconds`, and storage fields to understand allocative burden.
7. **Check justification**: Read `justified_itself` before treating the run as a decision-worthy success.

## How Agents Should Write Manifests

1. Create the manifest with `status: "pending"` before launching the pod.
2. Record `run_class` and `epistemic_status` up front.
3. Update `status` to `"running"` and set `started_at` when execution begins.
4. Update `status` to `"completed"` or `"failed"` and set `finished_at` when done.
5. Run `collect_billing.sh` and let it append `cost_snapshot`.
6. Fill in `wall_clock_seconds`, artifact/storage sizes, and `justified_itself` after results are reviewed.
7. If the run is benchmark-, promotion-, or deployment-oriented, attach or create a comparison artifact.

## Examples

- SAC training: `configs/runpod/examples/train_sac_manifest.json`
- Provider bring-up: `configs/runpod/examples/provider_bringup_manifest.json`
- Benchmark candidate training: `configs/runpod/examples/benchmark_candidate_training_manifest_v2.json`
