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

## Output Types

### bottleneck_report

Ranked table of roadmap bottlenecks with severity (high/medium/low), type (structural/external), and suggested resolution. Use when progress has stalled or priorities are unclear.

### next_actions

Ranked list of 3-5 highest-leverage tasks. Each item includes what to do, why now, what it unblocks, a verification command, and an explicit scope boundary. Use before planning sessions or after a tranche lands.

### claim_audit

Line-by-line comparison of doc claims against code/test/artifact reality. Each claim is marked `verified`, `unverified`, `blocked`, or `inferred`. Use when doc drift is suspected or before a phase closure review.

### upstream_comparison

Structured comparison table for evaluating whether to adopt, adapt, or ignore an approach from an upstream project (LeRobot, Habitat, Feynman, etc.). Use when a specific upstream pattern is under consideration.

### experiment_matrix

Proposed experiment grid for a subsystem, including hypotheses, inputs, expected outputs, verification commands, and priority. Use when a subsystem needs empirical validation before committing to an approach.

### refactor_recommendation

Specific refactor proposal with scope, motivation, before/after description, verification commands, risk assessment, and scope boundary. Use when structural debt is blocking progress.

## Integration with Execution Planes

### Codex Cloud Execution

The companion can recommend tasks suitable for Codex cloud execution. When it does, the recommendation includes:

- The Codex prompt (scoped, with verification command)
- The approval policy (`deny-all`, `read-only`, or as appropriate)
- Whether `--wait` or `--apply` is appropriate
- Expected output artifacts

The companion does not invoke Codex directly. It produces specs that the developer or orchestrating agent can feed to Codex.

### RunPod GPU Execution

When a recommendation requires GPU resources (model inference, training runs, benchmark evaluation), the companion flags it as `type: external` and specifies:

- What RunPod configuration is needed (GPU type, container image)
- What script or command to run
- What artifacts to collect
- Where results should be written (`results/run_registry/`)

See `codex_skills/runpod-gpu-execution/` for the RunPod execution skill.

## When to Invoke

| Trigger | Recommended Output Types |
|---------|-------------------------|
| After a Codex tranche lands | `claim_audit`, `next_actions` |
| Before a planning session | `bottleneck_report`, `next_actions` |
| When progress stalls | `bottleneck_report`, `upstream_comparison` |
| Weekly strategic review | `claim_audit`, `bottleneck_report` |
| Before phase closure review | `claim_audit` |
| Evaluating an upstream approach | `upstream_comparison` |
| Subsystem needs empirical validation | `experiment_matrix` |
| Structural debt blocking progress | `refactor_recommendation` |

## Example Companion Session

A typical session after a Codex tranche lands:

1. Read the full Read First list from the skill definition.
2. Run `python3 -m compileall src/ && pytest tests/ -v` to confirm repo state.
3. Produce a `claim_audit` comparing the tranche's claimed outputs against actual code and tests.
4. Produce `next_actions` ranking the 3-5 highest-leverage follow-up tasks.
5. If any recommendation requires GPU resources, flag it with RunPod configuration.
6. If any recommendation involves an upstream approach, produce an `upstream_comparison`.

The session output is a set of structured documents, not code changes. The developer or Codex acts on the recommendations.

## Anti-Patterns

- **Using the companion to generate implementation code.** The companion produces specs and recommendations. Codex implements.
- **Treating recommendations as mandates.** Recommendations are advisory. The developer decides priority based on full context.
- **Ignoring phase sequencing.** The companion respects the phase exit rule. If it recommends Phase 3 work while Phase 2 is open, that recommendation must be clearly marked as spec-only, not implementation.
- **Running the companion without reading the repo first.** The companion's value depends entirely on grounding in current repo state. Stale context produces stale recommendations.
- **Producing unbounded output.** The companion produces ranked lists of 3-5 items. If the output grows beyond that, the companion is being used wrong.
