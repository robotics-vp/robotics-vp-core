# Roadmap Execution Companion

## Purpose

The roadmap execution companion is an agent pattern for strategic bottleneck detection and next-action recommendation across the multi-WM roadmap. It reads the full repo state (docs, code, tests, artifacts, run logs, configs, backlogs) and produces bounded, ranked outputs that tell the developer or agent what to do next, why, and how to verify.

It complements but does not replace the nightly audit. The nightly audit is an incremental safe-step executor; the companion is a strategic advisor.

## Relationship to Nightly Audit

| Dimension | Nightly Audit | Roadmap Execution Companion |
|-----------|--------------|----------------------------|
| **Cadence** | Nightly (automated) | On-demand (after tranches, before planning, when stuck) |
| **Scope** | One safe additive step | Full roadmap bottleneck scan |
| **Output** | Code change + progress log update | Ranked recommendations, audits, comparisons |
| **Autonomy** | Can land commits autonomously | Advisory only; does not land code |
| **Risk tolerance** | Conservative (one scoped change) | Can recommend larger structural moves |
| **Phase awareness** | Follows current phase priority | Scans across phases but respects sequencing discipline |

The nightly audit answers: "What is the single safest next step?" The companion answers: "Where are we actually stuck, what should we do about it, and are our docs honest?"

## Preferred Output Types

The companion should prefer bounded artifacts over essays.

### `ranked_next_actions`

Ranked list of 3-5 highest-leverage tasks. Each item should include:

- **What**
- **Why now**
- **Unblocks**
- **Verify**
- **Do NOT**
- **Confidence**
- **Blocking**

### `bottleneck_report`

Ranked table of roadmap bottlenecks with severity, type, and suggested resolution.

### `claim_vs_code_audit`

Line-by-line comparison of doc claims against code/test/artifact reality. Each claim should be marked `verified`, `unverified`, `blocked`, or `inferred`.

### `run_comparison_summary`

Bounded summary of a benchmark-, promotion-, or deployment-oriented run family. It should mirror the run comparison artifact shape:

- baseline
- candidate run(s)
- what changed
- what improved
- what regressed
- confidence level
- promotion implication
- roadmap implication
- next recommended action

### Additional Supported Outputs

- `upstream_comparison`
- `experiment_matrix`
- `refactor_recommendation`

These remain valid, but the companion should default toward the four bounded output types above when possible.

## Integration with Execution Planes

### Codex Cloud Execution

The companion can recommend tasks suitable for Codex cloud execution. When it does, the recommendation includes:

- The Codex prompt (scoped, with verification command)
- The approval policy (`deny-all`, `read-only`, or as appropriate)
- Whether `--wait` or `--apply` is appropriate
- Expected output artifacts

The companion does not invoke Codex directly. It produces specs that the developer or orchestrating agent can feed to Codex.

### RunPod GPU Execution

When a recommendation requires GPU resources (model inference, training runs, benchmark evaluation), the companion should flag:

- required `run_class`
- expected `epistemic_status`
- RunPod configuration needed (GPU type, container image)
- script or command to run
- artifacts to collect
- where results should be written (`results/run_registry/`)
- whether a comparison artifact should be expected on completion

See `codex_skills/runpod-gpu-execution/` for the RunPod execution skill.

## Queue-Prioritization Awareness

When reasoning across multiple runnable ideas, the companion should use or surface fields such as:

- `wm`
- `subsystem`
- `blocker`
- `run_class`
- `epistemic_status`
- `expected_value`
- `estimated_cost_usd`
- `dependency_chain`
- `urgency`

This does not make the companion a scheduler. It simply ensures that recommendations are queue-legible once there are multiple concurrent GPU windows.

## Example Companion Session

A typical session after a Codex tranche lands:

1. Read the full Read First list from the skill definition.
2. Run `python3 -m compileall src/ && pytest tests/ -v` to confirm repo state.
3. Produce a `claim_vs_code_audit` comparing the tranche's claimed outputs against actual code and tests.
4. Produce `ranked_next_actions` ranking the 3-5 highest-leverage follow-up tasks.
5. If any recommendation requires GPU resources, specify run class, epistemic status, and RunPod configuration.
6. If any recommendation is benchmark- or promotion-oriented, state whether a comparison artifact should be emitted.
7. If any recommendation involves an upstream approach, produce an `upstream_comparison`.

The session output is a set of structured documents, not code changes. The developer or Codex acts on the recommendations.

## Anti-Patterns

- **Using the companion to generate implementation code.** The companion produces specs and recommendations. Codex implements.
- **Treating recommendations as mandates.** Recommendations are advisory. The developer decides priority based on full context.
- **Ignoring phase sequencing.** The companion respects the phase exit rule. If it recommends Phase 3 work while Phase 2 is open, that recommendation must be clearly marked as spec-only, not implementation.
- **Running the companion without reading the repo first.** The companion's value depends entirely on grounding in current repo state. Stale context produces stale recommendations.
- **Producing elegant but non-actionable essays.** If the companion cannot reduce uncertainty into a bounded output shape, it should state what is missing rather than expand rhetorically.
