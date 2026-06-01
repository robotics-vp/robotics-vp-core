# G1 Primary Environment

The repo-wide primary environment target is `bipedal_whole_body_unitree_g1`.

That means:

- `unitree_g1` / `unitree_g1_r1_class` is the default robot target;
- `bipedal_whole_body` is the primary posture;
- `stable_base_mobile_manipulator` is a safety fallback / degraded mode;
- `fixed_base_tabletop` envs remain curriculum, smoke, and regression sources.

Legacy surfaces such as `dishwashing`, `drawer_vase`, and `workcell` should not
be described as primary or final readiness evidence. They may feed SAC,
semantic, replay, and economic plumbing, but their receipts must keep the
curriculum boundary visible.

The checked-in profile is `configs/humanoid/g1_primary_env.yaml`.

Local hygiene sweep:

```bash
python3 scripts/economic_world_model/check_g1_primary_env_hygiene.py \
  --output-dir artifacts/economic_world_model/g1_primary_env_hygiene
```

RunPod launch manifests should use the G1 profiles emitted by:

```bash
python3 scripts/runpod/prepare_launch_manifest.py --profile provider_bringup
python3 scripts/runpod/prepare_launch_manifest.py --profile g1_loop_run
python3 scripts/runpod/prepare_launch_manifest.py --profile g1_sac_training
```

These commands only prepare manifests and launch commands. They do not create
pods, run providers, train weights, or prove hardware readiness.
