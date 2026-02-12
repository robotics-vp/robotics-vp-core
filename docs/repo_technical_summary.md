# Repository Technical Summary (robotics-vp-core)

## 1) Architectural overview

`robotics-vp-core` is an economics-first robotics stack that treats economic signals (MPL, error, energy, wage parity, risk) as first-class control inputs rather than post-hoc reporting. The core loop is intentionally end-to-end: video/observation input → representation/semantics → simulation + policy improvement → economic telemetry and pricing signals → data valuation and orchestrated sampling decisions.

The architecture is staged:
- **Current implemented core**: RL environments and training, reward shaping scaffolding, telemetry, valuation/orchestration interfaces, and determinism infrastructure.
- **Target architecture**: first-class ObjectiveTensor/EconTensor contracts, pricing sentinel, and governance (Regal nodes) as the default runtime control plane.

## 2) Module map (major directories)

### Root-level
- `src/`: Main Python packages for environments, RL/training, valuation, orchestration, and economics.
- `tests/`: Unit/integration/smoke coverage across envs, economics, vision, process_reward, plugins, and MuJoCo gates.
- `scripts/`: Smoke tests, docs utilities, MCP/agent tooling, and runnable verification scripts.
- `docs/`: Design references (HRL/VLA, embodiment, energy bench, ontology, reward schema, scene tracking).
- `.github/workflows/`: CI and nightly CI automation.
- `configs/` + `config/`: Runtime/training/scenario/objective configuration files.
- `specs/`: Phase-level design specs (e.g., Phase H, SIMA2 hardening).
- `checkpoints/`: Saved model assets, including stable world model artifacts.
- `artifacts/`, `reports/`, `results/`, `plots/`: Generated outputs and analysis artifacts.

### Selected `src/` packages (functional map)
- `src/envs/`: Core env implementations, including dishwashing, drawer-vase, and workcell variants.
- `src/rl/`: RL training utilities, reward shaping contracts, sampling/curriculum logic.
- `src/valuation/`: Datapack schema/valuation, reward builder interfaces, valuation-oriented utilities.
- `src/economics/`: Economic reward decomposition and telemetry primitives.
- `src/orchestrator/`: Advisory orchestration and diffusion request wiring.
- `src/vision/`: Scene IR tracking, map-first supervision, and vision artifact plumbing.
- `src/vla/`, `src/hrl/`, `src/sima/`, `src/sima2/`: Phase C+ scaffolding for semantic control and hierarchy.
- `src/controllers/`: Synthetic weighting and budget control components (with frozen constraints in Phase B).
- `src/world_model/`: Frozen world-model math (explicitly non-modifiable per project constraints).
- `src/ontology/`: Economic/episode dataclasses and persistence schema.
- `src/observation/`, `src/representation/`, `src/encoders/`: Observation/embedding construction and modality bridging.
- `src/motor_backend/`, `src/physics/`, `src/envs/physics/`: Execution backends and physics adapters.
- `src/process_reward/`: Process-level reward analysis and regression gates.
- `src/regal/`: Governance/safety-control abstractions around objective and pricing integrity.
- `src/training/`, `src/inference/`, `src/deployment/`: Runtime training/inference/deploy plumbing.

## 3) Active vs experimental components

### Active / stable-enough
- CI-backed fast and nightly test lanes, including syntax checks and smoke gates.
- High test volume in `tests/` and regular compile/test workflows.
- Environment stack (`src/envs/`) plus workcell smoke and MuJoCo integration tests.
- Economic telemetry and reward-shaping scaffolding integrated into training/evaluation paths.

### Experimental / unfinished / advisory
- Objective-conditioned SAC path is explicitly marked as a **stub** with TODO integration points and no behavior-change default.
- Some physics backend pathways (Isaac/MuJoCo abstraction notes) are marked TODO.
- Video-to-policy docs indicate staged migration from toy state vectors toward real visual modality.
- Several valuation/orchestrator components are described as advisory scaffolding rather than hardened default runtime behavior.

## 4) Data flow (high-level)

Typical current flow:
1. **Environment reset/step** from `src/envs/*` or backend factory (`src/envs/physics/backend_factory.py`).
2. **Observation path** through observation/encoder layers (`src/observation`, `src/encoders`, `src/representation`).
3. **Policy action loop** in RL training utilities (`src/rl/*`).
4. **Reward decomposition** via reward shaping + economics (`src/rl/reward_shaping.py`, `src/economics/reward_engine.py`, `src/valuation/reward_builder.py`).
5. **Episode summaries / ontology persistence** in `src/ontology/*` and logging paths.
6. **Valuation + orchestrator advisory decisions** (datapacks, sampling hints, diffusion request hooks) in `src/valuation` and `src/orchestrator`.
7. **Artifacts and reports** emitted into `artifacts/`, `reports/`, and `results/`.

Planned target flow expands this into explicit ObjectiveTensor/ConstraintSet/EconTensor/PricingSentinel contracts before scalarization and ledger writes.

## 5) Dependencies summary

Core runtime dependencies are intentionally lean in `pyproject.toml`:
- `torch`: model/training backbone.
- `numpy`: numerical operations.
- `scikit-learn`: ML utilities.
- `pydantic>=2`: typed data models/contracts.
- `pyyaml`: config loading.
- `matplotlib`: plotting/analysis.

Developer/test dependencies in `requirements-dev.txt`:
- `pytest`, `ruff`, `mypy`.

Optional extra physics dependency in `requirements-extra.txt`:
- `mujoco==3.2.5` for dedicated MuJoCo CI and integration lanes.

## 6) Tests and CI

- Test suite is substantial (`tests/` includes unit + integration + smoke + plugin-specific tests).
- CI (`.github/workflows/ci.yml`) runs:
  - agent ergonomics verification,
  - compile checks,
  - fast pytest lane (`not slow and not no_mujoco`),
  - MuJoCo missing-dependency UX test,
  - workcell smoke script.
- Separate MuJoCo job installs extras and runs MuJoCo-specific integration gates.
- Nightly CI (`ci-nightly.yml`) runs full test suite + compile checks on schedule.

## 7) README & docs evaluation

Strengths:
- README clearly states thesis, current-vs-target architecture, constraints, and operational commands.
- Explicit staged roadmap and definitions (pricing/data-sharing contract primitives).
- Good practical verification loop and smoke-test references.

Gaps:
- README itself acknowledges there is not yet a fully unified production roboeconomic runtime.
- Architecture breadth is large; some module-level docs are design-forward and future-facing rather than implementation-complete.
- Multiple parallel phase documents imply active evolution; onboarding still requires cross-reading several docs.

## 8) Roadmap inference (near-term)

From roadmap/design artifacts, likely next milestones:
- Complete objective-conditioned training wiring (RewardBuilder integration flip from stub path).
- Strengthen video-conditioned policy path and modality migration from state vectors.
- Harden Phase C/Phase H HRL-VLA-SIMA scaffolding into more production-like loops.
- Promote pricing/data valuation from advisory overlays to tighter runtime control/accounting paths.
- Expand simulation/physics realism while keeping deterministic replay and economic telemetry guarantees.

## 9) Risk and complexity profile

Primary risk areas:
- **Architecture breadth vs integration depth**: many modules/phases increase coupling surface.
- **Stub-to-production transitions**: objective-conditioned SAC and some backend adapters still have TODO seams.
- **Frozen-zone constraints**: Phase B lock protects stability but can force workaround complexity in adjacent modules.
- **Physics/backend variability**: optional dependency pathways (MuJoCo/Isaac) introduce env-specific behavior and CI matrix complexity.
- **Economic truthfulness challenge**: ensuring pricing/valuation signals remain robust against proxy gaming as stack scales.

## 10) TODO / FIXME summary

Notable explicit TODOs/FIXMEs identified:
- `src/rl/train_sac_objective.py`: step summary structure, policy conditioning, RewardBuilder integration point.
- `src/rl/hydra_losses.py`: actor/critic loss placeholders slated for PPO/SAC and TD(λ)/GAE-style replacements.
- `src/config/internal_profile.py`: migration TODO to full `PolicyProfile`.
- `src/motor_backend/holosoma_backend.py`: preset mapping, evaluation criteria mapping, and real trajectory capture placeholders.
- `src/envs/physics/isaac_backend.py` + physics package notes: backend initialization/availability TODOs.
- `src/scene/vector_scene/tiled.py`: TODO note in tiled scene implementation.
- `src/valuation/REWARD_INTEGRATION_DESIGN.md`: implementation TODO checklist for objective-conditioned reward rollout.

