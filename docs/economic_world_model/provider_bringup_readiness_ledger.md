# Provider Bring-Up Readiness Ledger

Date: 2026-06-02

## Purpose

This pass materializes a cross-WM provider bring-up ledger so the first
provider/GPU window starts from explicit owner WMs, commands, expected receipts,
unavailable posture, RunPod profile, blockers, and local verification commands.

It is a template-only readiness surface. It does not download weights, launch
RunPod, execute providers, run GPU jobs, operate hardware, train, write
weights, mutate reward math, grant authority, or claim promotion.

## Landed surfaces

```text
src/world_model/economic_world_model/provider_bringup_ledger.py
scripts/economic_world_model/compile_provider_bringup_readiness_ledger.py
tests/test_provider_bringup_readiness_ledger.py
```

The Economic WM package exports the ledger builders, loaders, saver, and
validator.

## Artifact path

```bash
python3 scripts/economic_world_model/compile_provider_bringup_readiness_ledger.py
```

Current local artifact result:

- `entry_count=7`
- `covered_required_family_count=6`
- `required_family_count=6`
- `launch_allowed_count=0`
- `provider_bringup_ready_count=0`
- `local_verification_available_count=7`
- `runpod_template_count=7`
- `missing_prerequisite_count=23`
- `all_entries_fail_closed=true`
- `provider_executed=false`
- `gpu_executed=false`
- `runpod_launched=false`
- `weights_downloaded=false`
- `weights_written=false`
- `training_executed=false`
- `hardware_executed=false`
- `promotion_eligible=false`

Current local prerequisite probes:

- `runpodctl_on_path=false`
- `RUNPOD_API_KEY_set=false`
- `RUNPOD_VOLUME_ID_set=false`
- `cuda_visible_devices_set=false`

## Covered Provider Families

| Provider key | Family | Owner WM | RunPod profile | Status |
| --- | --- | --- | --- | --- |
| `sam_sam3d_scene_ir` | `sam_sam3d` | `perception_grounding` | `provider` | `blocked_template_only` |
| `dino_siglip_vision_backbone` | `dino_siglip` | `perception_grounding` | `provider` | `blocked_template_only` |
| `vjepa2_sim_synth_predictive` | `vjepa2` | `sim_synth_physics` | `provider` | `blocked_template_only` |
| `vjepa2_perception_temporal` | `vjepa2` | `perception_grounding` | `provider` | `blocked_template_only` |
| `openvla_semantic_teacher` | `openvla` | `perception_grounding` | `provider` | `blocked_template_only` |
| `isaac_unitree_runtime` | `isaac_unitree` | `sim_synth_physics_and_embodiment_actuation` | `loop` | `blocked_template_only` |
| `holosoma_runtime` | `holosoma` | `embodiment_actuation` | `loop` | `blocked_template_only` |

## Verification

```bash
python3 -m pytest -q tests/test_provider_bringup_readiness_ledger.py
python3 scripts/economic_world_model/compile_provider_bringup_readiness_ledger.py
python3 -m ruff check src/world_model/economic_world_model/provider_bringup_ledger.py scripts/economic_world_model/compile_provider_bringup_readiness_ledger.py tests/test_provider_bringup_readiness_ledger.py
python3 -m mypy src/world_model/economic_world_model/provider_bringup_ledger.py scripts/economic_world_model/compile_provider_bringup_readiness_ledger.py --show-error-codes --no-error-summary
```

Focused test result: `2 passed`.

## Boundary

The ledger is safe for template storage only. It is intentionally not safe for
launch. Real provider proof still requires replacing template guards, recording
run manifests, and producing real provider/runtime receipts on the appropriate
RunPod/provider/hardware plane.
