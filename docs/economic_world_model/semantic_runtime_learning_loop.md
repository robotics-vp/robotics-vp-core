# Semantic Runtime Learning Loop

Date: 2026-03-24

## Goal

Create a production-ready semantic loop before a learned controller is allowed to take authority.

That loop must do four things:

1. ingest semantic evidence from DINO/SceneTracks/Map-First and OpenVLA/teacher lanes
2. compile that evidence into a canonical semantic world model
3. let orchestration/meta-transformer shells act over that state in bounded ways
4. feed execution outcomes and counterfactual regret back into future learning and inferential scoring

## Runtime Topology

```text
SceneTracks / Map-First / DINO-like semantic lane
  -> track labels, confidence, topology, object identity
  -> semantic grounding priors for SemanticWorldModelState

OpenVLA / teacher semantic lane
  -> teacher trace, object refs, affordance hints, risk hints
  -> VLA semantic evidence sidecars
  -> action/affordance priors for SemanticWorldModelState

Semantic fusion
  -> confidence-weighted agreement / disagreement
  -> evidence bus / belief state
  -> fusion confidence back into SemanticWorldModelState

Semantic world model
  -> objects, relations, capabilities, topology, meta nodes
  -> SemanticSnapshot
  -> runtime semantic packet

Transformer shells
  -> MetaTransformer bounded routing packet
  -> OrchestrationTransformer bounded tool/activation packet
  -> execution preconditions + work orders

Runtime / replay / receipts
  -> observed outcomes
  -> blocked vs executed work orders
  -> reward/quality/readiness evidence
  -> future-training artifacts/signals

Semantic runtime learning corpus
  -> canonical rows
  -> shadow counterfactuals
  -> meta-transformer runtime dataset
  -> orchestration runtime dataset
  -> future scorer/controller training
```

## Semantic WM Relation To DINO / OpenVLA Annotation

The semantic world model should be the meeting point, not a third disconnected lane.

- DINO / SceneTracks / Map-First side:
  - contributes object persistence, confidence, geometry, relation hints, and stable scene identity
  - is strongest at object memory, topology, and non-linguistic visual consistency
- OpenVLA / teacher side:
  - contributes instruction-conditioned affordance, action, recovery, and risk hints
  - is strongest at action semantics and instruction-conditioned affordance meaning
- semantic fusion:
  - measures agreement, disagreement, and confidence asymmetry
  - should eventually supervise authority selection and calibration
- semantic world model:
  - absorbs both lanes into one object/relation/meta-node packet
  - is the canonical substrate the transformer shells should consume

This means the right learned authority problem is not “DINO vs OpenVLA in isolation.” It is:

- given semantic grounding quality
- given VLA affordance confidence
- given fusion agreement/disagreement
- given downstream outcome evidence
- which lane should dominate which bounded decision?

## What Landed In Code

- `src/orchestrator/semantic_runtime_learning.py`
  - builds canonical semantic runtime rows from replay datasets and live sidecars
  - loads semantic world model, teacher trace, VLA semantic evidence, SceneTracks metadata, and replay outcomes
  - emits:
    - semantic summaries
    - VLA/OpenVLA summaries
    - DINO/SceneTracks proxy summaries
    - transformer targets
    - feedback summaries
    - shadow counterfactuals
    - inferential labels
- `scripts/export_semantic_runtime_learning_corpus.py`
  - exports the runtime corpus
  - exports meta-transformer runtime dataset
  - exports orchestration runtime dataset
- `src/orchestrator/semantic_runtime_scorers.py`
  - trains lightweight route-success / authority-calibration / counterfactual-value / regret scorers from the same runtime rows
  - scores live semantic-WM plus transformer packets in shadow mode
- `src/orchestrator/semantic_runtime_scorer_training.py`
  - builds an explicit scorer-training dataset over the runtime row schema
  - provides an optional torch multitask trainer/checkpoint path for heavier learned reranking work
- `scripts/train_semantic_runtime_scorers.py`
  - exports the scorer-training dataset
  - trains the lightweight scorer package
  - optionally trains and checkpoints the heavier torch scorer net

## Learning Pipeline Needed For End-To-End Production

### 1. Runtime Corpus

Every production-semantic run should leave behind:

- semantic world model packet
- semantic snapshot
- orchestrator/meta-transformer/orchestration execution packets
- replay/outcome evidence
- artifact refs for VLA evidence, teacher trace, SceneTracks, fusion sidecars

### 2. Supervised Runtime Datasets

Two runtime-backed datasets should exist continuously:

- meta-transformer dataset
  - VLA embedding/features
  - DINO/semantic embedding/features
  - semantic tokens
  - authority target
  - bounded routing target
- orchestration dataset
  - semantic-aware orchestrator context
  - bounded tool sequence target
  - execution/readiness target

### 3. Inferential Dataset

Each runtime row should also carry:

- counterfactual objective-preset alternatives
- authority alternative
- predicted route score
- estimated regret
- semantic/fusion gain labels

This is the immediate pre-training inferential substrate.

### 4. Offline Scorers Before Full Controller Training

Train these before a full learned controller:

- route-success classifier
- authority calibrator
- counterfactual value/regret scorer
- execution-readiness predictor

This repo now has the first three instantiated:

- a lightweight live-shadow scorer package for immediate inference
- a heavyweight scorer-training dataset plus optional torch checkpoint path
- a backlog-tracked training entrypoint at `scripts/train_semantic_runtime_scorers.py`

### 5. Promotion Evidence

Production promotion should depend on:

- execution count
- ready vs blocked rate
- route success rate
- counterfactual regret
- calibration error
- semantic grounding quality

## Inferential Pipeline Needed For End-To-End Production

The inferential path should run even when controller learning is not active.

It should:

1. take a semantic runtime row
2. generate bounded candidate alternatives
3. score them from current semantic/fusion/outcome evidence
4. compute regret against the chosen bounded route
5. publish that result as future training evidence

This is what lets the stack learn from production traces before real transformer fine-tuning begins.

## Recommended Next Layers

1. add execution-result joins for actual transformer work orders once those start being persisted in replay
2. promote the scorer-training path into the full regal/promotion training envelope once broader training windows open
3. use scorer outputs to rerank bounded decisions in shadow mode against denser real execution traces
4. promote the best-calibrated transformer lane from deterministic routing to learned reranking
