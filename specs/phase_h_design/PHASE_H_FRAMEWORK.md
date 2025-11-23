# Phase H: Dynamic Skill Acquisition & Online Adaptation

**Status**: Conceptual Draft
**Owner**: Antigravity
**Context**: The "Self-Modifying" Phase

## 1. Teleology: The Economic Learner

Phase H marks the transition from a static set of skills to a **dynamic, growing ontology**. The system stops being just a "worker" and becomes a "learner".

**Core Principle**: Learning is an investment.
*   **Cost**: Energy, compute, risk of failure during exploration.
*   **Return**: New skills (MPL), higher efficiency, better robustness.
*   **The Econ Controller decides when to learn.** If the ROI of learning "how to open a new latch" is positive, the system allocates a budget for it.

## 2. Conceptual Framework

### 2.1. Dynamic Skill Heads
Instead of a fixed `AffordanceHead` with N classes, we introduce a **Dynamic Head** architecture.
*   **Structure**: A transformer-based head that takes a `TaskEmbedding` as a query and attends to visual features.
*   **Growth**: When a new skill is discovered (e.g., via SIMA-2 OOD recovery), a new `TaskEmbedding` is added to the registry.
*   **No Retraining**: The backbone remains frozen. Only the lightweight adapter weights or the embedding registry are updated.

### 2.2. Condition-Driven Task Embedding
The `ConditionVector` becomes the "key" for the dynamic head.
*   **Input**: `ConditionVector` (contains `task_id`, `skill_mode`).
*   **Process**: Map `task_id` -> `TaskEmbedding`.
*   **Output**: The policy focuses on features relevant to *that specific task*.

### 2.3. Online Adaptation Loop
1.  **Detection**: `SIMA-2` flags an OOD event ("Latch is stuck").
2.  **Proposal**: `OntologyUpdateEngine` proposes `NewSkill: UnstickLatch`.
3.  **Valuation**: `EconController` prices the learning cost (e.g., "100 failures allowed").
4.  **Exploration**: `Sampler` switches to `FrontierMode`. The agent tries variations.
5.  **Consolidation**: Once success > threshold, the successful trajectory is "distilled" into a new `TaskEmbedding` or a small MLP adapter.

## 3. Integration Blueprint

### 3.1. Safety & OOD Handling
*   **The Safety Net**: During adaptation, the `SafetyPolicy` (sovereign layer) runs in "Paranoid Mode". Any collision or force spike triggers an immediate stop.
*   **Semantic Orchestrator**: Manages the "Learning Session". It ensures the agent doesn't drift into unrelated tasks.

### 3.2. Econ-Directed Continual Learning
*   **Learning Budget**: The system has a `LearningWallet`. It earns credits by completing routine tasks (high MPL). It spends credits on exploration (zero MPL, high risk).
*   **Bankruptcy**: If the agent fails too often during learning, the `EconController` revokes the budget and forces a return to known skills.

## 4. Roadmap for Codex

1.  **Phase H1 (The Hook)**: Implement the `DynamicSkillHead` interface (even if it just wraps static heads for now).
2.  **Phase H2 (The Wallet)**: Implement `LearningBudget` in the `EconController`.
3.  **Phase H3 (The Loop)**: Connect `SIMA-2` OOD signals to the `Sampler` to trigger automatic exploration sessions.
