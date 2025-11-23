# Implementation Handoff: SIMA-2 Hardening & Phase H

**Date**: 2025-11-23
**Architect**: Claude (Semantic Design)
**Implementer**: Codex (Code Generation)

---

## Executive Summary

This document provides a complete handoff from semantic/economic architecture (Claude) to implementation (Codex). All specifications are **unambiguous, deterministic, and ready for direct implementation** without further design decisions.

---

## 1. Completed Work (Ready for Production)

### SIMA-2 Core Pipeline ✅

**Implemented by Codex under Claude's direction**:

1. **Task Library & Rollout Templates** ([src/sima2/client.py](src/sima2/client.py))
   - 4 templates: success, failure, recovery, mixed
   - 3 task families: drawer_open, dish_place, wipe_surface
   - Deterministic generation with seed support
   - Ground truth failure/recovery event markers

2. **Heuristic Segmentation** ([src/sima2/heuristic_segmenter.py](src/sima2/heuristic_segmenter.py))
   - Physics-based primitive detection (gripper, contact, velocity)
   - Failure → recovery pattern recognition
   - Flag-gated for backward compatibility

3. **OOD & Recovery Tags** ([src/sima2/tags/ood_recovery_tags.py](src/sima2/tags/ood_recovery_tags.py))
   - OODTag, RecoveryTag dataclasses
   - Pattern detection functions

4. **EconCorrelator** ([src/analytics/econ_correlator_impl.py](src/analytics/econ_correlator_impl.py))
   - Statistical correlation: E[Damage | Tag]
   - TrustMatrix generation with trust scores
   - JSON-safe artifact output

---

## 2. Specification Documents (Implementation-Ready)

### SIMA-2 Specifications

**[SIMA2_INVARIANTS_AND_PHASE_H_HOOKS.md](specs/sima2_hardening/SIMA2_INVARIANTS_AND_PHASE_H_HOOKS.md)**

Defines:
- **Exact firing conditions** for OODTag (visual/kinematic/temporal) and RecoveryTag
- **TrustMatrix tier semantics**: Trusted (>0.8), Provisional (0.5-0.8), Untrusted (<0.5)
- **Module integration rules**: DataPackRLSampler (5x/1.5x/1x multipliers), SemanticOrchestrator (SafetyStop authority), Auditor (trust-weighted scoring)
- **Phase H skill quality signals**: success_rate, recovery_rate, fragility_score
- **Phase H exploration signals**: ood_rate, failure_diversity, recovery_density
- **Must-enforce invariants**: OOD Containment, Recovery Imperative, Advisory Boundary

**Implementation Tasks**:
- [ ] Wire TrustMatrix into DataPackRLSampler with sampling multipliers
- [ ] Implement OODTag firing logic (3 branches: visual, kinematic, temporal)
- [ ] Implement RecoveryTag pattern matching (fail → recovery → success)
- [ ] Wire trust scores into SemanticOrchestrator SafetyStop logic
- [ ] Add `extract_skill_quality()` and `extract_exploration_value()` to SIMA-2 pipeline
- [ ] Create smoke tests for invariants

---

### Text-Front-Door Specifications

**[TEXT_FRONT_DOOR_COMPLETE_SEMANTICS.md](specs/phase_h_design/TEXT_FRONT_DOOR_COMPLETE_SEMANTICS.md)**

Defines:
- **Instruction taxonomy**: 10 instruction types (MODIFY_RISK, SPEED_MODE, ENERGY_CONSTRAINT, etc.)
- **Regex patterns** for intent recognition (deterministic, no LLM needed initially)
- **Intent → ConditionVector mapping**: Exact field assignments for each instruction type
- **Safety clamping rules**: risk_tolerance ∈ [0.1, 0.9], energy_budget ≥ 10 Wh, time_budget ≥ 5 sec
- **Rejection patterns**: "ignore safety", "override econ", etc.
- **20 canonical examples** with exact JSON outputs
- **Module integration**: SIMA-2 modulation, SemanticOrchestrator proposals, EconController advisory

**Implementation Tasks**:
- [ ] Create `src/tfd/` package with TextOp, SemanticCompiler, SafetyFilter
- [ ] Implement `parse_instruction()` using INTENT_PATTERNS regex
- [ ] Implement `compile_to_condition_vector()` with deterministic field mapping
- [ ] Implement `apply_safety_constraints()` with clamping rules
- [ ] Implement `should_reject_instruction()` with REJECTED_PATTERNS
- [ ] Add smoke test for all 20 canonical examples
- [ ] Verify determinism: same text → same ConditionVector

---

### Vision Adapter Specifications

**[CONDITIONED_VISION_ADAPTER_SEMANTICS.md](specs/phase_h_design/CONDITIONED_VISION_ADAPTER_SEMANTICS.md)**

Defines:
- **Modulation mechanisms**: FiLM (feature-wise modulation), attention reweighting, RNN gating
- **Field-specific rules**: risk_tolerance → risk saliency scaling, novelty_tier → OOD amplification, skill_mode → attention routing
- **Three canonical regimes**:
  - Safety-first: risk amplification (1.3x), fine features boosted (2x), temporal smoothing high
  - Frontier-explore: novelty amplification (3x), mid-scale features, temporal smoothing low
  - Energy-efficient: spatial downsampling (50%), top-k sparsity (30%), temporal reuse high
- **Invariants**: Base representation (z_v) unchanged, deterministic, bounded scales [0.1, 10.0]
- **Architecture**: RegNet → FiLM layers → BiFPN (weighted) → Spatial RNN (gated) → Risk/Affordance heads (scaled)

**Implementation Tasks**:
- [ ] Create `src/vision/conditioned_adapter.py` with ConditionedVisionAdapter class
- [ ] Implement FiLM modulation layers
- [ ] Implement `compute_fusion_weights()` from ConditionVector.skill_mode
- [ ] Implement `compute_rnn_gate()` from ConditionVector
- [ ] Implement `modulate_risk_map()` and `modulate_ood_saliency()`
- [ ] Wire into ObservationAdapter with flag-gating
- [ ] Add smoke test: verify z_v unchanged, regime outputs differ

---

### Phase H Specifications

**[PHASE_H_ECONOMIC_LEARNER_DESIGN.md](specs/phase_h_design/PHASE_H_ECONOMIC_LEARNER_DESIGN.md)**

Defines:
- **Skill dataclass**: mpl_baseline, mpl_current, mpl_target, success_rate, failure_rate, recovery_rate, fragility_score, ood_exposure, training_cost_usd
- **Skill lifecycle**: EXPLORATION → TRAINING → MATURE → DEPRECATED (transition rules specified)
- **Budget allocation logic**: proportional to MPL gap, adjusted by quality penalty and novelty bonus
- **ROI calculation**: (productivity value + energy savings + damage savings - training cost) / training cost * 100
- **Learner cycle** (every 1000 episodes): measure returns → compute ROI → reallocate budgets → update statuses → generate reports
- **SIMA-2 integration**: Extract skill quality from success_rate, recovery_rate, fragility_score
- **TFD integration**: Deploy skills via "use [skill] skill" instruction
- **Artifacts**: skill_market_state.json, exploration_budget.json, skill_returns.json
- **Advisory boundaries**: EconController/RewardFunction/Ontology CANNOT consume Phase H artifacts

**Implementation Tasks**:
- [ ] Create `src/phase_h/` package with Skill, ExplorationBudget, SkillReturns dataclasses
- [ ] Implement `allocate_exploration_budget()` with MPL gap + quality penalty + novelty bonus
- [ ] Implement `compute_skill_roi()` with productivity/efficiency/quality returns
- [ ] Implement `EconomicLearner.run_cycle()` with budget reallocation
- [ ] Implement `update_skill_from_sima2()` to consume SIMA-2 quality signals
- [ ] Implement `deploy_skill_from_tfd()` to handle TFD skill activation
- [ ] Generate JSON artifacts (skill_market_state.json, etc.)
- [ ] Enforce advisory boundaries (validation checks)
- [ ] Add smoke tests for skill lifecycle transitions

---

## 3. Remaining Implementation Tasks

### Priority 1: Core SIMA-2 Hardening

**Estimated Effort**: 2-3 hours

1. **OODTag Implementation**
   - File: `src/sima2/tags/ood_recovery_tags.py`
   - Add visual/kinematic/temporal OOD detection branches
   - Wire into SegmentationEngine
   - Test: Verify firing on high embedding distance, high velocity, long duration

2. **RecoveryTag Implementation**
   - File: `src/sima2/tags/ood_recovery_tags.py`
   - Add pattern matching for fail → recovery → success
   - Wire into SegmentationEngine
   - Test: Verify detection on recovery rollouts

3. **TrustMatrix Wiring**
   - Files: `src/sampling/datapack_sampler.py`, `src/orchestrator/semantic_orchestrator.py`, `src/auditing/datapack_auditor.py`
   - Add trust-weighted sampling multipliers (5x, 1.5x, 1x)
   - Add SafetyStop authority check (trust > 0.8 required)
   - Add audit score adjustments
   - Test: Verify sampling distribution changes with trust scores

4. **Phase H Hooks**
   - File: `src/sima2/phase_h_integration.py` (new)
   - Implement `extract_skill_quality()` and `extract_exploration_value()`
   - Wire into SIMA-2 pipeline output
   - Test: Verify correct quality metrics extraction

---

### Priority 2: Text-Front-Door

**Estimated Effort**: 3-4 hours

1. **TFD Package Creation**
   - Create `src/tfd/` directory structure
   - Implement `text_op.py`, `semantic_compiler.py`, `safety_filter.py`, `patterns.py`
   - Test: Run all 20 canonical examples, verify JSON output matches spec

2. **Integration Points**
   - Wire TFD into `ConditionVectorBuilder`
   - Add `--instruction` CLI flag to relevant scripts
   - Test: Verify instruction changes ConditionVector fields

---

### Priority 3: Conditioned Vision Adapter

**Estimated Effort**: 4-5 hours

1. **ConditionedVisionAdapter**
   - Create `src/vision/conditioned_adapter.py`
   - Implement FiLM layers, fusion weight computation, RNN gating
   - Test: Verify determinism, regime output differences

2. **ObservationAdapter Wiring**
   - Modify `src/observation/adapter.py` to use ConditionedVisionAdapter
   - Add `use_conditioned_vision` flag
   - Test: Verify backward compatibility (flag off = original behavior)

---

### Priority 4: Phase H Economic Learner

**Estimated Effort**: 5-6 hours

1. **Phase H Package**
   - Create `src/phase_h/` directory with Skill, EconomicLearner, budget allocation
   - Implement learner cycle with ROI computation
   - Test: Run 5000 episodes, verify budget reallocation occurs

2. **Artifact Generation**
   - Implement JSON artifact writers for skill_market_state, exploration_budget, skill_returns
   - Test: Verify JSON schema matches spec

3. **Integration**
   - Wire `update_skill_from_sima2()` to SIMA-2 pipeline
   - Wire `deploy_skill_from_tfd()` to TFD
   - Test: End-to-end flow from SIMA-2 quality signal → budget reallocation

---

## 4. Testing Strategy

### Smoke Tests (Required for Each Module)

1. **SIMA-2**:
   - `scripts/smoke_test_ood_recovery_tags.py`: Verify OODTag/RecoveryTag firing
   - `scripts/smoke_test_trust_matrix_integration.py`: Verify sampling multipliers
   - `scripts/smoke_test_phase_h_hooks.py`: Verify skill quality extraction

2. **TFD**:
   - `scripts/smoke_test_tfd_canonical_examples.py`: Run all 20 examples, verify JSON
   - `scripts/smoke_test_tfd_determinism.py`: Same input → same output
   - `scripts/smoke_test_tfd_safety_filter.py`: Verify rejection patterns

3. **Vision**:
   - `scripts/smoke_test_conditioned_vision_regimes.py`: Verify regime outputs differ
   - `scripts/smoke_test_vision_invariants.py`: Verify z_v unchanged, bounded scales

4. **Phase H**:
   - `scripts/smoke_test_economic_learner_cycle.py`: Verify budget reallocation
   - `scripts/smoke_test_skill_lifecycle.py`: Verify EXPLORATION → TRAINING → MATURE transitions
   - `scripts/smoke_test_phase_h_artifacts.py`: Verify JSON schema

---

### Stress Tests

1. **SIMA-2 10k Rollout** (`scripts/stress_test_sima2_pipeline.py`):
   - Process 10,000 rollouts in < 1 hour
   - Memory usage < 2GB
   - No OOM errors
   - Deterministic segmentation for same seed

2. **TFD 1000 Instructions** (`scripts/stress_test_tfd_throughput.py`):
   - Process 1000 instructions in < 1 second
   - Deterministic outputs
   - No regex timeout errors

---

## 5. Configuration Flags

**All new features are flag-gated for backward compatibility**:

```yaml
# config/sima2.yaml
segmentation:
  use_heuristic_segmenter: true  # Enable physics-based segmentation

# config/vision.yaml
vision:
  use_conditioned_vision: true  # Enable ConditionedVisionAdapter

# config/tfd.yaml
tfd:
  enabled: true  # Enable Text-Front-Door
  safety_filter: strict  # Rejection rule strictness

# config/phase_h.yaml
phase_h:
  enabled: true  # Enable Economic Learner
  reallocation_period_episodes: 1000
  total_exploration_budget_usd: 10000.0
```

**When flags are OFF, system behaves identically to baseline (no regressions).**

---

## 6. Advisory Boundaries (Critical)

**The following modules MUST NOT be modified by Phase H work**:

1. **EconController**: Pricing and wage calculations are independent of skill training
2. **Reward Functions**: Rewards are task-based, not skill-based
3. **Task Ordering**: Curriculum is managed separately from skill budgets

**Phase H artifacts are advisory inputs to**:
- DataPackRLSampler (sampling weights)
- CurriculumManager (skill prioritization)
- TFD (skill deployment)
- Dashboards/Reports (visibility)

---

## 7. Success Criteria

### SIMA-2 Hardening Complete When:
- [ ] OODTag fires on visual/kinematic/temporal anomalies per spec
- [ ] RecoveryTag detects fail → recovery → success patterns
- [ ] TrustMatrix influences sampling with 5x/1.5x/1x multipliers
- [ ] Skill quality signals extracted from SIMA-2 output
- [ ] All smoke tests pass
- [ ] 10k rollout stress test completes in < 1 hour

### Text-Front-Door Complete When:
- [ ] All 20 canonical examples produce correct JSON
- [ ] Safety filter rejects prohibited instructions
- [ ] TFD outputs deterministic ConditionVectors
- [ ] Integration with ConditionVectorBuilder works
- [ ] All smoke tests pass

### Vision Adapter Complete When:
- [ ] Three regimes produce different outputs
- [ ] Base representation (z_v) unchanged across regimes
- [ ] Integration with ObservationAdapter works
- [ ] Flag-gating maintains backward compatibility
- [ ] All smoke tests pass

### Phase H Complete When:
- [ ] Skill lifecycle transitions work (EXPLORATION → TRAINING → MATURE)
- [ ] Budget reallocation occurs every 1000 episodes
- [ ] ROI computed correctly from productivity/efficiency/quality returns
- [ ] SIMA-2 + TFD integration works
- [ ] JSON artifacts generated with correct schema
- [ ] Advisory boundaries enforced (no EconController consumption)
- [ ] All smoke tests pass

---

## 8. File Manifest

**Specification Documents** (Claude):
- `specs/sima2_hardening/SIMA2_INVARIANTS_AND_PHASE_H_HOOKS.md`
- `specs/phase_h_design/TEXT_FRONT_DOOR_COMPLETE_SEMANTICS.md`
- `specs/phase_h_design/CONDITIONED_VISION_ADAPTER_SEMANTICS.md`
- `specs/phase_h_design/PHASE_H_ECONOMIC_LEARNER_DESIGN.md`

**Implemented Code** (Codex):
- `src/sima2/client.py` (task library)
- `src/sima2/heuristic_segmenter.py` (segmentation)
- `src/sima2/segmentation_engine.py` (engine)
- `src/sima2/tags/ood_recovery_tags.py` (tags)
- `src/analytics/econ_correlator_impl.py` (correlator)

**Pending Implementation** (Codex):
- `src/tfd/` (Text-Front-Door package)
- `src/vision/conditioned_adapter.py` (Vision adapter)
- `src/phase_h/` (Economic Learner package)
- Integration wiring (sampler, orchestrator, auditor, observation adapter)
- Smoke tests and stress tests

---

## 9. Next Steps for Codex

1. **Implement OODTag/RecoveryTag firing logic** using specs in SIMA2_INVARIANTS_AND_PHASE_H_HOOKS.md
2. **Wire TrustMatrix** into DataPackRLSampler, SemanticOrchestrator, DatapackAuditor
3. **Create src/tfd/ package** following TEXT_FRONT_DOOR_COMPLETE_SEMANTICS.md
4. **Create src/vision/conditioned_adapter.py** following CONDITIONED_VISION_ADAPTER_SEMANTICS.md
5. **Create src/phase_h/ package** following PHASE_H_ECONOMIC_LEARNER_DESIGN.md
6. **Write smoke tests** for each module
7. **Run stress tests** to validate scaling

All specifications are complete and unambiguous. No further design decisions are needed. Proceed with direct implementation.

---

**End of Handoff Document**
