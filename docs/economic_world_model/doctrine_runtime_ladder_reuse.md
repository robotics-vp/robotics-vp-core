# Doctrine Note: Runtime Ladder Reuse in Embodiment / Hardware-Facing Control

## Status

- **Type**: forward-looking doctrine constraint
- **Applies to**: Phase 3 (Embodiment / Actuation WM) and Phase 4A/4E (real-time / companion-compute layers)
- **Does NOT authorize**: premature normalization, cross-WM abstraction, or Embodiment WM implementation starting now

## The Pattern

Phase 1 Sim / Synth / Physics WM developed a 10-rung runtime ladder that decomposes "can we execute on this backend?" into distinct maturity questions:

```
1. backend binding           — what backend are we targeting?
2. deployment contract       — what modes does this backend support?
3. upstream runtime pack     — what's available on the host?
4. runtime binding           — which surfaces are relevant to this mode?
5. executable-adapter request — what does the WM want the adapter to do?
6. executable-adapter consumer — who takes responsibility for the request?
7. adapter execution         — which execution path (local / external)?
8. adapter realization       — is the chosen path concretely realizable?
9. local materialization / external launch — actual execution
10. harvested runtime outcomes — what came back?
```

Each rung answers a question that was previously collapsed into "backend available / unavailable."

## Why This Pattern Should Reuse

The Embodiment / Actuation WM will face the same decomposition problem for hardware-facing control:

- "Can we control the G1?" is not one question. It is:
  - Do we have the Unitree SDK2 / ROS2 middleware installed? (upstream runtime pack)
  - Is the target deployment mode supported — sim, teleop, autonomous? (deployment contract)
  - Do we have the right URDF, calibration, joint naming, and controller gains for this mode? (runtime binding — mode-scoped surface selection)
  - What does the WM want the control adapter to do — whole-body, locomotion-only, manipulation-only? (adapter request)
  - Which control pipeline takes responsibility — reflex controller, skill policy, high-level planner? (adapter consumer)
  - Is the chosen pipeline concretely runnable on this compute placement? (adapter realization)
  - Did it produce control outputs and receipts? (harvested outcomes)

Without this decomposition, Embodiment will repeat the same fake-readiness failure mode that Sim/Synth/Physics had: "hardware unavailable" doing the work of 5+ different missing-component classes.

## What Should Transfer

**The structural decomposition** — the idea that host truth, mode-scoped binding, WM intent, consumer ownership, execution routing, realization, and outcome harvest are separate maturity rungs.

**The status vocabulary** — `binding_ready / binding_partial / binding_blocked`, `realization_blocked`, `pack_ready / pack_partial`, etc. The same honest three-way distinction between "ready," "partially ready," and "blocked" should apply to hardware control surfaces.

**The mode-scoped missing-component filtering** — the Holosoma pattern of filtering pack-level gaps by deployment mode was the single highest-value change in the Tier 2 tranche. Embodiment should do the same: a locomotion-only mode should not be blocked by missing manipulation calibration.

**The test pattern** — structural tests that verify binding compilation and missing-component propagation without requiring real hardware.

## What Should NOT Transfer

**The per-backend implementation details.** Isaac and Holosoma runtime bindings have backend-specific surface names (`policy_surface`, `motion_surface`, `retargeting_surface`). The Embodiment WM will have different surface names (`joint_state_surface`, `control_command_surface`, `safety_envelope_surface`, `sensor_fusion_surface`). Do not try to normalize these into a shared vocabulary now.

**The adapter file structure.** Phase 1 has `adapters/isaac_unitree_runtime_binding.py` and `adapters/holosoma_runtime_binding.py` as separate files per backend. Phase 3 may have a different factoring — one robot model may need multiple control-rate adapters rather than one file per hardware target. Let the factoring emerge from the Embodiment WM's actual ownership boundaries.

**The execution routing semantics.** Phase 1 routes between `local_python_bridge` and `external_launch` as execution paths. Phase 3 will route between `onboard_reflex`, `companion_gpu`, `offboard_planning`, and potentially `degraded_mode`. The routing categories are domain-specific; only the ladder structure is shared.

## What Should NOT Happen Now

1. **Do not create a shared `RuntimeLadder` base class or abstraction.** The pattern is a structural discipline, not a library. If both WMs need the decomposition, both should implement it against their own domain — not inherit from a premature common ancestor.

2. **Do not add Embodiment-facing fields to the existing Sim/Synth/Physics binding modules.** The current `isaac_unitree_runtime_binding.py` should remain Sim/Synth-scoped. When Phase 3 needs Unitree hardware binding, it should create its own `src/world_model/embodiment_actuation/adapters/` tree.

3. **Do not normalize the two backend binding ontologies into a cross-WM contract yet.** Per the Q1 doctrinal answer in `claude_to_comment_on.md`: keep backend-local until the Embodiment / Actuation WM is active. The Embodiment WM should own the cross-backend normalization when it arrives.

4. **Do not inject compute / battery / placement awareness into the existing Sim/Synth binding layer.** Per the Q2 doctrinal answer: wait until Phase 4A/4E makes QoS consequences real. The insertion point exists (the `_required_surfaces()` pattern), but exercising it now would produce degradation signals nothing can act on.

## When This Note Becomes Actionable

This note becomes actionable when:

- Phase 1 structural closure is declared (per `phase1_closure_standard.md`)
- Phase 2 (Perception / Grounding WM) schema work is underway or complete
- Phase 3 (Embodiment / Actuation WM) implementation begins

At that point, the first Embodiment tranche spec should reference this note and use the ladder decomposition as a structural template — not a code dependency.

## Relationship to Unitree Readiness Sequencing

The runtime ladder makes Unitree readiness more plannable. Instead of one milestone ("can we control the G1?"), the program can track:

| Rung | Unitree G1 question | Expected timing |
|------|---------------------|-----------------|
| Upstream runtime pack | Is Unitree SDK2 installed? URDF available? | Phase 3 bring-up |
| Deployment contract | Which modes — sim, teleop, autonomous — are supported? | Phase 3 |
| Runtime binding | For this mode, which joint groups, sensors, and controllers are bound? | Phase 3 |
| Adapter request | What does the WM want the control pipeline to do? | Phase 3 |
| Adapter consumer | Which control pipeline takes responsibility? | Phase 3 |
| Adapter execution | Local onboard vs companion GPU vs offboard? | Phase 4A/4E |
| Adapter realization | Is compute placement concretely viable? | Phase 4A/4E |
| Harvested outcomes | Are we getting real control receipts back? | Post-hardware (July 2027+) |

This table should be expanded into a concrete Unitree readiness checklist when Phase 3 begins, complementing `humanoid_target_readiness.md`.
