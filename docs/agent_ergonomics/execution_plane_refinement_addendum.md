# Execution Plane Refinement Addendum

This document is an additive refinement pass on top of the Sep 2026 execution-plane standup. It does **not** replace the thin-wrapper + manifest + registry posture already documented elsewhere.

The goal is to make the execution plane more decision-oriented without turning it into a fake internal platform.

## Non-Goals

- Do **not** build a large orchestration layer.
- Do **not** pretend that comparison, billing, or prioritization are fully automated if they are not.
- Do **not** let small smoke runs quietly inherit benchmark or promotion credibility.
- Do **not** turn the roadmap companion into an essay engine.

## 1. Two Classification Axes

A run should be classified along **both** of these axes:

### A. Run Class

The existing execution class answers: *what kind of machine work is this?*

- `loop`
- `provider`
- `train`
- `refactor`

### B. Epistemic Status

A second axis answers: *what inferential weight should this run carry?*

- `smoke`
- `proof_of_life`
- `benchmark_candidate`
- `promotion_candidate`
- `deployment_candidate`

### Doctrine

A run is not fully described by `pod_class` alone.

Examples:

- a `provider` + `smoke` run means adapter or runtime bring-up only
- a `train` + `proof_of_life` run means the training path executed but is not yet benchmark-credible
- a `train` + `benchmark_candidate` run means it is intended for disciplined comparison
- a `loop` + `promotion_candidate` run means the resulting artifacts may justify subsystem or policy promotion if the comparison record supports it
- a `refactor` + `deployment_candidate` run should be rare and only used when the validation bar is unusually high

Default rule: if a run's epistemic status is omitted, treat it conservatively as **no stronger than `smoke`**.

## 2. Additive Manifest Fields

These fields are additive extensions to the existing run-manifest doctrine. They can be recorded in `manifest.json`, a sidecar receipt, or both.

```json
{
  "run_class": "loop | provider | train | refactor",
  "epistemic_status": "smoke | proof_of_life | benchmark_candidate | promotion_candidate | deployment_candidate",
  "wm": "string | null",
  "subsystem": "string | null",
  "blocker": "string | null",
  "expected_value": "string | null",
  "estimated_cost_usd": "number | null",
  "dependency_chain": ["string"],
  "urgency": "low | medium | high | critical | null",
  "gpu_class": "string | null",
  "wall_clock_seconds": "number | null",
  "artifact_size_bytes": "number | null",
  "storage_or_checkpoint_size_bytes": "number | null",
  "justified_itself": "yes | no | unclear | null"
}
```

### Notes

- `run_class` is the doctrinal name for the machine-work axis. Existing `pod_class` values remain valid; `run_class` is the more general decision-facing label.
- `epistemic_status` prevents accidental over-interpretation.
- `wm`, `subsystem`, and `blocker` make the run queue legible at roadmap scale.
- `expected_value`, `estimated_cost_usd`, `dependency_chain`, and `urgency` help with queue prioritization once there are multiple GPU windows.
- `gpu_class`, `wall_clock_seconds`, `artifact_size_bytes`, and `storage_or_checkpoint_size_bytes` preserve allocative facts needed later by the Economic WM and broader control plane.
- `justified_itself` is intentionally blunt. It forces a bounded judgment instead of a vague post-hoc narrative.

## 3. Run Comparison as a First-Class Artifact

Launching runs is not the hard part. Comparing them honestly is.

Every meaningful run family should eventually yield a comparison artifact with the following bounded shape:

```json
{
  "comparison_id": "string",
  "baseline": "string",
  "candidate_runs": ["string"],
  "what_changed": ["string"],
  "what_improved": ["string"],
  "what_regressed": ["string"],
  "confidence_level": "low | medium | high",
  "promotion_implication": "string",
  "roadmap_implication": "string",
  "next_recommended_action": "string"
}
```

### Doctrine

- A `proof_of_life` run normally does **not** justify a comparison artifact unless it is explicitly being compared against another proof-of-life path.
- A `benchmark_candidate` run should normally have a named baseline before launch.
- A `promotion_candidate` run should not be considered promotion-relevant without a comparison artifact.
- A `deployment_candidate` run should have both a comparison artifact and a justification judgment.

## 4. Cost and Time Belong in Receipts

The execution plane is not just ops plumbing. It is an early allocative surface.

At minimum, a completed run should preserve:

- `gpu_class`
- `wall_clock_seconds`
- `estimated_cost_usd`
- `artifact_size_bytes`
- `storage_or_checkpoint_size_bytes`
- `justified_itself`

These can be approximate early on, but they should not be omitted from the doctrine.

## 5. Roadmap Companion Posture

The roadmap execution companion should prefer **bounded outputs** over long-form commentary.

Preferred outputs:

- `ranked_next_actions`
- `bottleneck_report`
- `claim_vs_code_audit`
- `run_comparison_summary`

Companion anti-pattern:

- elegant but non-actionable essays that increase operator entropy

If the companion cannot reduce uncertainty into a bounded output shape, it should say what is missing rather than expand rhetorically.

## 6. Queue Prioritization Posture

As concurrent GPU windows multiply, runs should be sortable by more than timestamp.

A prioritization-ready record should expose:

- `wm`
- `subsystem`
- `blocker`
- `run_class`
- `epistemic_status`
- `expected_value`
- `estimated_cost_usd`
- `dependency_chain`
- `urgency`

This does **not** require a scheduler right now. It only requires that the execution plane preserve the fields needed for later scheduling discipline.

## 7. Stage-Appropriate Philosophy

The intended posture remains:

- enough structure for recurring multi-GPU work
- comparison-friendly and decision-oriented records
- no fake automation
- no premature platform building

The thin-wrapper + manifest + registry + skill model remains correct.

This addendum only sharpens the ergonomics so the execution plane becomes more than a nicer launcher while still staying far short of an internal platform.
