# Isaac Integration Outline (Scaffold)

> **Architectural context**: Isaac Lab integration enters the multi-WM topology
> as a motor backend consumed by the **Embodiment / Actuation WM**
> (`docs/actuation_embodiment_world_model.md`). We borrow Isaac Lab's scalable
> embodiment-aware task environments, contact-rich simulation surfaces, and
> sensor/backend/task abstraction patterns. We do NOT import Isaac Lab's
> environment abstraction as the master ontology — our WM boundaries remain
> separate. Isaac backend availability is a typed provider-truth surface
> inside the Embodiment WM, not an implicit assumption.

Goal: swap the physics backend while keeping the economics + orchestrator + datapack stack unchanged.

## PhysicsBackend API
- `make_backend(engine_type, env_name, ...)` returns a `PhysicsBackend`.
- Must provide: `reset()`, `step(action)`, `get_episode_info()` (EpisodeInfoSummary), `get_info_history()`.
- IsaacBackend currently stubbed; implement matching fields:
  - obs shape compatible with drawer_vase_* configs.
  - info fields sufficient to build EpisodeInfoSummary (mpl/error/energy, termination_reason, media_refs/episode_id).

## Engine flow
- `engine_type` flag threads through CLI → env/backend construction → ConditionProfile/Objectives → datapacks.
- If engine_type != pybullet, scripts should warn/NotImplemented, but keep tags correct for future use.

## Episode/Datapack schema requirements
- EpisodeInfoSummary must include energy fields, episode_id, media_refs.
- Datapacks record engine_type via ConditionProfile and ObjectiveProfile, plus media_refs for video/sim traces.
- Guidance/Diffusion requests carry env/engine info for future video generation.

## Isaac Gym parity checklist
- Action/obs mapping identical to PyBullet bench envs.
- Energy attribution (joint τ·ω) with per-limb/joint outputs.
- Termination reasons consistent (max_steps, success, collisions).
- EpisodeInfoSummary serialization unchanged.

## Integration steps (future)
1) Implement IsaacBackend with Isaac task matching drawer_vase_arm_env.
2) Update make_backend to construct real IsaacBackend when engine_type="isaac".
3) Ensure datapacks/guidance/diffusion requests reflect engine_type="isaac".
4) Reuse orchestrator/run-specs to select backend; no change to rewards or Phase B economics.
