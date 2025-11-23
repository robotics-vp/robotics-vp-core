# To-Do for Claude: SIMA-2 Hardening & Phase H Kickoff

**Context**: Antigravity has completed the architectural review of SIMA-2 and the design phase for the Text-Front-Door and ObservationAdapter. The system is ready for implementation, but critical gaps in the SIMA-2 pipeline must be fixed first.

## 1. Immediate Priority: Fix SIMA-2 Gaps
**Reference**: `specs/sima2_hardening/SIMA2_HARDENING_REVIEW.md`

The current SIMA-2 implementation is a "Potemkin Village"—it looks right but lacks the internal logic to actually validate robustness.

*   [ ] **Upgrade `Sima2Client`**: Modify `src/sima2/client.py` to support `template="failure"|"recovery"|"mixed"`. It must generate trajectories that actually fail and recover, not just successful ones.
*   [ ] **Implement `EconCorrelator`**: Fill in `src/analytics/econ_correlator.py` with the statistical logic to compute `RiskPremium` and `RecoveryValue` from Datapacks.
*   [ ] **Verify Invariants**: Ensure `SemanticTagPropagator` flags "Lucky" successes (High Risk + Success + No Recovery).

## 2. Second Priority: Text-Front-Door Scaffolding
**Reference**: `specs/phase_h_design/TEXT_FRONT_DOOR_DESIGN.md`

*   [ ] Create `src/tfd/` module.
*   [ ] Implement `TextOp` entry point.
*   [ ] Implement a regex-based `SemanticCompiler` to map basic commands ("careful", "fast") to `ConditionVector` overrides.

## 3. Third Priority: ObservationAdapter Fusion
**Reference**: `specs/phase_h_design/OBSERVATION_ADAPTER_DESIGN.md`

*   [ ] Implement `ConditionedVisionAdapter` in `src/vision/`.
*   [ ] Wire it into `ObservationAdapter` to modulate visual features based on the `ConditionVector`.

## 4. Phase H Roadmap
**Reference**: `specs/phase_h_design/PHASE_H_FRAMEWORK.md`

*   [ ] Review the "Economic Learner" framework for future dynamic skill acquisition work.
