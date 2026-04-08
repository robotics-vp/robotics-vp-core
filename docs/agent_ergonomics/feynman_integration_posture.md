# Feynman Integration Posture

## Decision

Adopt the useful patterns now; defer the full fork.

Feynman (https://github.com/getcompanion-ai/feynman) is a deep-research agent framework built on Pi/Companion AI runtime. It has good workflow patterns for research integrity, provenance tracking, and structured output. It also has heavy runtime coupling to a platform we do not use. This document records what we borrow, what we adapt, what we ignore, and why.

## What We Adopt Now (Pattern-Level, No Code Dependency)

These patterns are adopted directly into this repo's agent infrastructure. No Feynman code is imported.

### SKILL.md YAML-frontmatter convention

Already in use in this repo (`codex_skills/*/SKILL.md`). Feynman uses the same convention, confirming it as a reasonable cross-agent standard. No action needed.

### Researcher integrity protocol

Core rules borrowed from Feynman's research agent posture:

- **URL or it didn't happen**: every factual claim in a research output must link to a source URL. Claims without sources are marked `inferred` or `unverified`, never presented as fact.
- **Read before summarize**: do not summarize a source you have not actually fetched and read. If a source is unreachable, say so.
- **Never fabricate sources**: no hallucinated URLs, no invented paper titles, no fake DOIs.
- **Mark verification status honestly**: use `verified`, `blocked`, `unverified`, `inferred`. Do not round up.

These rules apply to all agent-produced research outputs in this repo, including upstream comparisons, architecture reviews, and literature surveys.

### Deep-research workflow pattern

The workflow structure: plan, fan-out (parallel sub-queries), synthesize, verify, provenance sidecar. This maps naturally to:

1. **Plan**: define the research question and sub-questions
2. **Fan-out**: parallel Codex tasks or Claude Code agents for sub-queries
3. **Synthesize**: merge sub-query results into a structured output
4. **Verify**: check each claim against its source
5. **Provenance sidecar**: emit a `.provenance.md` file alongside the output

### Provenance tracking

Convention: any research output file `docs/research/topic.md` should have a companion `docs/research/topic.provenance.md` containing:

- Date of research
- Agent and model version
- Sources consulted (URLs, with access timestamps)
- Verification status of each major claim
- Known gaps or unreachable sources

### Output organization

Structured output directories with slugged naming. Research outputs go in `docs/research/` or `results/` with names like `upstream-comparison-lerobot-act-policy.md`, not ambiguous names like `notes.md` or `research.md`.

### Evidence table format

Structured evidence tables for research outputs:

```
| Source ID | URL | Claim | Claim Type | Confidence | Verified |
|-----------|-----|-------|-----------|------------|----------|
| S1 | https://... | ... | empirical / architectural / anecdotal | high / medium / low | yes / no / blocked |
```

### CHANGELOG-as-lab-notebook

For multi-session research tasks, maintain a CHANGELOG section at the top of the output file recording what was done in each session, what remains, and what changed. This enables resumable research without context loss.

### Honest verification labels

Four labels, no others:

- `verified`: claim checked against source, source accessible, claim holds
- `blocked`: source exists but is inaccessible (paywalled, requires auth, network error)
- `unverified`: claim not yet checked against source
- `inferred`: claim derived from reasoning, not directly stated in any source

## What We Adapt Carefully

### Subagent delegation pattern

Feynman delegates sub-queries to Pi-runtime-specific subagents (`pi-subagents`). We adapt this to our execution model:

- **Parallel Codex tasks**: for implementation-oriented sub-queries, use `./scripts/codex/enqueue.sh` or Codex cloud execution
- **Claude Code agents**: for analysis-oriented sub-queries, use parallel Claude Code sessions
- **Coordination**: the orchestrating agent (Claude copilot or companion skill) synthesizes results from sub-queries

The adaptation preserves the fan-out/synthesize pattern without importing Pi runtime.

### Replication workflow with environment selection

Feynman supports local/Docker/RunPod execution environments for replication tasks. We adapt this to our execution plane model:

- **Local**: `python3 -m compileall src/ && pytest tests/ -v` for structural verification
- **RunPod**: for GPU-requiring replication (model inference, training runs), use `codex_skills/runpod-gpu-execution/`
- **Codex cloud**: for sandboxed execution of implementation tasks

### Session search

Feynman maintains session search across research history. We adapt this to our existing patterns:

- `.agent/runs/`: Codex execution logs with structured metadata
- `results/run_registry/`: experiment results and artifacts
- `docs/economic_world_model/progress_log.md`: narrative progress history

No new session-search infrastructure is needed. The existing patterns are sufficient.

## What We Do NOT Import

### Pi runtime

`pi-subagents`, `pi-web-access`, `pi-docparser`, and other Pi-runtime-specific modules are platform-coupled. We do not use Pi/Companion AI infrastructure. Importing these would create a dependency on a platform we do not control and do not need.

### Alpha search integration

Feynman's academic search integration (Semantic Scholar, arXiv, PubMed APIs) is domain-specific to academic literature research. If we need academic search in the future, we will build a minimal adapter, not import Feynman's full search stack.

### The `feynman` CLI binary

We have our own agent infrastructure (`codex_skills/`, `.agent/`, `scripts/codex/`). A second CLI binary adds confusion without capability.

### Full Feynman repo tree

No vendoring. No git subtree. No submodule. The useful patterns are documented here and implemented natively. The code stays in its upstream repo.

## Future Posture

If we later need a dedicated research agent for upstream comparison, architecture review, or literature survey at a scale that exceeds what the roadmap execution companion can do:

1. Fork Feynman into a separate repo (e.g., `robotics-vp/feynman-research-agent`)
2. Strip Pi-runtime dependencies and replace with Codex/Claude Code execution
3. Wire it as an external tool callable from this repo's agent infrastructure
4. Do not vendor it into `robotics-vp-core`

Until that need materializes, the repo-local companion skill absorbs the useful workflow patterns without the runtime dependency.

## Explicit Non-Goals

- **Do not vendor Feynman into this repo.** No `vendor/`, no `third_party/`, no submodule.
- **Do not create a dependency on Pi/Companion AI infrastructure.** We use Codex and Claude Code.
- **Do not duplicate Feynman's full subagent orchestration.** Codex parallel tasks and Claude Code agents already provide parallel execution. Adding another orchestration layer creates confusion.
- **Do not import Feynman's search integrations.** If we need academic search, we build a minimal adapter scoped to our needs.
