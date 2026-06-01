# GPU / Provider Run Hygiene

This is the local preflight layer for provider bring-up, loop runs, and future
training runs. It validates manifest-shaped runs before launch and emits
receipts; it does not launch pods, execute providers, train models, or grant
promotion authority.

Primary commands:

```bash
python3 scripts/economic_world_model/check_gpu_run_hygiene.py \
  --manifest-dir configs/runpod/examples \
  --output-dir artifacts/economic_world_model/gpu_run_hygiene

python3 scripts/economic_world_model/check_wm_surface_hygiene.py \
  --output-dir artifacts/economic_world_model/wm_surface_hygiene
```

Current local results:

- GPU run hygiene:
  - `status=ok_gpu_run_hygiene_passed`
  - `manifest_count=3`
  - `receipt_count=51`
  - `blocking_issue_count=0`
  - `safe_to_queue_count=3`
- WM surface hygiene:
  - `status=ok_wm_surface_hygiene_passed`
  - `scanned_file_count=326`
  - `python_file_count=243`
  - `doc_file_count=75`
  - `manifest_file_count=3`
  - `blocking_issue_count=0`
  - `risky_true_claim_count=0`
  - `protected_change_count=0`
  - `oversized_python_file_count=3`
  - `todo_marker_count=0`

The sweep inventories large files and TODO markers as information, not as
launch blockers. Blocking failures are reserved for things that would create
bad evidence: missing run-manifest fields, invalid run class or epistemic
status, inline secrets, generic checkpoint sinks, protected baseline path
references, risky execution claims set to true, or broken example manifests.

## Required Posture Before GPU Work

- Every RunPod/Codex-cloud/provider/training run must have a manifest under
  `.agent/runs/<run_id>/manifest.json` or an equivalent reviewed manifest before
  launch.
- `run_class` and `epistemic_status` must be explicit. Smoke and
  proof-of-life runs must not be read as benchmark or promotion evidence.
- Pending manifests must not contain runtime truth: no pod id, start time,
  finish time, billing truth, or post-hoc justification before execution.
- Training or provider manifests must use run-scoped artifact paths, not generic
  `checkpoints/` sinks.
- Commands must not inline API keys, tokens, passwords, or secrets.
- Stable Phase B checkpoint and frozen controller/reward surfaces remain
  protected.
- Promotion/deployment candidates need comparison or benchmark artifacts before
  their results can carry promotion-grade weight.

## Boundary

These checks make future GPU/provider work less error-prone. They do not prove
GPU availability, provider availability, training quality, Unitree hardware
truth, benchmark promotion, live policy control, reward-math changes, or Phase 7
authority.
