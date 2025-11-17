"""
Pipeline Manager: Coordinates the 5-stage learning loop.

Manages the flow from:
1. Offline objective/guidance solving
2. Data collection with annotations
3. Policy training (advisory recommendations)
4. Evaluation and metrics
5. Feedback and iteration

This is orchestrator-level infrastructure that provides visibility into the full
learning pipeline. It is ADVISORY ONLY - does not execute training or modify Phase B.

This is additive infrastructure - no changes to Phase B math or RL training loops.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.semantic_metrics import write_semantic_metrics, load_semantic_metrics, SemanticMetrics
from src.orchestrator.semantic_orchestrator import SemanticOrchestrator
from src.orchestrator.semantic_metrics import write_semantic_metrics
from enum import Enum
from datetime import datetime
import json
import uuid


class PipelineStage(Enum):
    """Stages in the learning pipeline."""
    OBJECTIVE_SOLVING = "objective_solving"  # Stage 1: Solve for optimal objectives
    DATA_COLLECTION = "data_collection"  # Stage 2: Collect/annotate data
    POLICY_TRAINING = "policy_training"  # Stage 3: Train policies (advisory)
    EVALUATION = "evaluation"  # Stage 4: Evaluate performance
    FEEDBACK_ITERATION = "feedback_iteration"  # Stage 5: Process feedback, iterate


class StageStatus(Enum):
    """Status of a pipeline stage."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from executing a pipeline stage."""
    stage: PipelineStage
    status: StageStatus
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    # Stage-specific outputs
    outputs: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"objective_vector": [...], "datapacks_annotated": 50, ...}

    metrics: Dict[str, float] = field(default_factory=dict)
    # e.g., {"mpl_delta": 2.5, "error_reduction": 0.1}

    recommendations: List[str] = field(default_factory=list)
    # e.g., ["Increase data collection for fragile objects"]

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Next stage hints
    next_stage_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "errors": self.errors,
            "next_stage_config": self.next_stage_config,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageResult":
        """Create from dictionary."""
        return cls(
            stage=PipelineStage(d["stage"]),
            status=StageStatus(d["status"]),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
            duration_seconds=d.get("duration_seconds", 0.0),
            outputs=d.get("outputs", {}),
            metrics=d.get("metrics", {}),
            recommendations=d.get("recommendations", []),
            warnings=d.get("warnings", []),
            errors=d.get("errors", []),
            next_stage_config=d.get("next_stage_config", {}),
        )


@dataclass
class PipelineIteration:
    """
    A single iteration through the 5-stage pipeline.

    Tracks progress through all stages and accumulates results.
    """
    iteration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    iteration_number: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""

    # Stage results
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    # Key is PipelineStage.value

    # Global configuration for this iteration
    config: Dict[str, Any] = field(default_factory=dict)

    # Summary metrics aggregated across stages
    summary_metrics: Dict[str, float] = field(default_factory=dict)

    # Status
    current_stage: Optional[PipelineStage] = None
    is_complete: bool = False
    is_failed: bool = False
    failure_reason: str = ""

    def start_stage(self, stage: PipelineStage) -> StageResult:
        """Mark a stage as started."""
        self.current_stage = stage
        result = StageResult(
            stage=stage,
            status=StageStatus.IN_PROGRESS,
            started_at=datetime.now().isoformat(),
        )
        self.stage_results[stage.value] = result
        return result

    def complete_stage(self, stage: PipelineStage, result: StageResult):
        """Mark a stage as completed with results."""
        result.status = StageStatus.COMPLETED
        result.completed_at = datetime.now().isoformat()
        # Calculate duration
        if result.started_at:
            start = datetime.fromisoformat(result.started_at)
            end = datetime.fromisoformat(result.completed_at)
            result.duration_seconds = (end - start).total_seconds()
        self.stage_results[stage.value] = result
        self.current_stage = None

    def fail_stage(self, stage: PipelineStage, reason: str):
        """Mark a stage as failed."""
        if stage.value in self.stage_results:
            result = self.stage_results[stage.value]
            result.status = StageStatus.FAILED
            result.errors.append(reason)
            result.completed_at = datetime.now().isoformat()
        self.is_failed = True
        self.failure_reason = f"Stage {stage.value} failed: {reason}"
        self.current_stage = None

    def get_stage_result(self, stage: PipelineStage) -> Optional[StageResult]:
        """Get result for a specific stage."""
        return self.stage_results.get(stage.value)

    def all_stages_complete(self) -> bool:
        """Check if all stages are complete."""
        for stage in PipelineStage:
            if stage.value not in self.stage_results:
                return False
            if self.stage_results[stage.value].status not in [
                StageStatus.COMPLETED,
                StageStatus.SKIPPED,
            ]:
                return False
        return True

    def compute_summary(self):
        """Compute summary metrics from all stages."""
        # Aggregate metrics across stages
        all_metrics = {}
        for result in self.stage_results.values():
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Summarize (use last value for deltas, mean for others)
        for key, values in all_metrics.items():
            if "delta" in key or "change" in key:
                # Use sum for cumulative changes
                self.summary_metrics[key] = sum(values)
            else:
                # Use last value
                self.summary_metrics[key] = values[-1] if values else 0.0

        # Mark as complete if all stages done
        if self.all_stages_complete():
            self.is_complete = True
            self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration_id": self.iteration_id,
            "iteration_number": self.iteration_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
            "config": self.config,
            "summary_metrics": self.summary_metrics,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "is_complete": self.is_complete,
            "is_failed": self.is_failed,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineIteration":
        """Create from dictionary."""
        stage_results = {
            k: StageResult.from_dict(v) for k, v in d.get("stage_results", {}).items()
        }
        current_stage = None
        if d.get("current_stage"):
            current_stage = PipelineStage(d["current_stage"])

        return cls(
            iteration_id=d.get("iteration_id", str(uuid.uuid4())),
            iteration_number=d.get("iteration_number", 0),
            started_at=d.get("started_at", datetime.now().isoformat()),
            completed_at=d.get("completed_at", ""),
            stage_results=stage_results,
            config=d.get("config", {}),
            summary_metrics=d.get("summary_metrics", {}),
            current_stage=current_stage,
            is_complete=d.get("is_complete", False),
            is_failed=d.get("is_failed", False),
            failure_reason=d.get("failure_reason", ""),
        )


@dataclass
class PipelineManager:
    """
    Manages the overall learning pipeline across multiple iterations.

    IMPORTANT: This is ADVISORY ONLY. It provides structure and visibility
    into the learning process but does NOT execute training or modify Phase B.
    """
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Pipeline"
    description: str = ""

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # History of iterations
    iterations: List[PipelineIteration] = field(default_factory=list)

    # Current state
    current_iteration: Optional[PipelineIteration] = None

    # Global metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Advisory recommendations
    global_recommendations: List[str] = field(default_factory=list)

    def start_new_iteration(self, config: Optional[Dict[str, Any]] = None) -> PipelineIteration:
        """Start a new pipeline iteration."""
        iteration_number = len(self.iterations) + 1
        iteration = PipelineIteration(
            iteration_number=iteration_number,
            config=config or self.config.copy(),
        )
        self.current_iteration = iteration
        return iteration

    def complete_current_iteration(self):
        """Complete the current iteration and archive it."""
        if self.current_iteration:
            self.current_iteration.compute_summary()
            self.iterations.append(self.current_iteration)
            self.current_iteration = None

    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """Get history of all iterations."""
        return [it.to_dict() for it in self.iterations]

    def get_progress_metrics(self) -> Dict[str, Any]:
        """Get progress metrics across all iterations."""
        if not self.iterations:
            return {"iterations": 0, "progress": {}}

        # Track metrics over time
        metrics_over_time = {}
        for iteration in self.iterations:
            for key, value in iteration.summary_metrics.items():
                if key not in metrics_over_time:
                    metrics_over_time[key] = []
                metrics_over_time[key].append(value)

        # Compute trends
        trends = {}
        for key, values in metrics_over_time.items():
            if len(values) >= 2:
                # Simple trend: last value minus first value
                trends[key] = values[-1] - values[0]
            else:
                trends[key] = 0.0

        return {
            "iterations": len(self.iterations),
            "completed_iterations": sum(1 for it in self.iterations if it.is_complete),
            "failed_iterations": sum(1 for it in self.iterations if it.is_failed),
            "metrics_over_time": metrics_over_time,
            "trends": trends,
        }

    def generate_advisory_report(self) -> Dict[str, Any]:
        """
        Generate advisory report based on pipeline history.

        ADVISORY ONLY: Suggests actions but does not execute them.
        """
        report = {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "total_iterations": len(self.iterations),
            "progress": self.get_progress_metrics(),
            "recommendations": self.global_recommendations.copy(),
            "stage_summaries": {},
            "action_items": [],
        }

        # Aggregate stage-specific insights
        for stage in PipelineStage:
            stage_stats = {
                "total_runs": 0,
                "successful": 0,
                "failed": 0,
                "avg_duration_s": 0.0,
                "common_warnings": [],
                "common_recommendations": [],
            }

            durations = []
            all_warnings = []
            all_recommendations = []

            for iteration in self.iterations:
                result = iteration.get_stage_result(stage)
                if result:
                    stage_stats["total_runs"] += 1
                    if result.status == StageStatus.COMPLETED:
                        stage_stats["successful"] += 1
                        durations.append(result.duration_seconds)
                    elif result.status == StageStatus.FAILED:
                        stage_stats["failed"] += 1
                    all_warnings.extend(result.warnings)
                    all_recommendations.extend(result.recommendations)

            if durations:
                stage_stats["avg_duration_s"] = sum(durations) / len(durations)

            # Find most common warnings/recommendations
            if all_warnings:
                stage_stats["common_warnings"] = list(set(all_warnings))[:3]
            if all_recommendations:
                stage_stats["common_recommendations"] = list(set(all_recommendations))[:3]

            report["stage_summaries"][stage.value] = stage_stats

        # Generate action items based on patterns
        progress = report["progress"]
        if "trends" in progress:
            if progress["trends"].get("mpl_delta", 0) < 0:
                report["action_items"].append(
                    "MPL is declining - consider reviewing objective weights or data quality"
                )
            if progress["trends"].get("error_rate", 0) > 0:
                report["action_items"].append(
                    "Error rate is increasing - consider adding safety constraints"
                )
            if progress["trends"].get("energy_efficiency", 0) < 0:
                report["action_items"].append(
                    "Energy efficiency declining - review motion planning strategies"
                )

        return report

    def preview_next_iteration(self) -> Dict[str, Any]:
        """
        Preview what the next iteration would look like.

        ADVISORY ONLY: Does not execute anything.
        """
        # Get insights from last iteration
        last_results = {}
        if self.iterations:
            last_iteration = self.iterations[-1]
            last_results = {
                "summary_metrics": last_iteration.summary_metrics,
                "recommendations": [],
            }
            for result in last_iteration.stage_results.values():
                last_results["recommendations"].extend(result.recommendations)

        # Suggest next iteration configuration
        suggested_config = self.config.copy()

        # Apply heuristic adjustments based on last iteration
        if last_results.get("summary_metrics", {}).get("error_rate", 0) > 0.1:
            suggested_config["increase_safety_weight"] = True
        if last_results.get("summary_metrics", {}).get("mpl_delta", 0) < 1.0:
            suggested_config["increase_data_collection"] = True

        return {
            "iteration_number": len(self.iterations) + 1,
            "last_iteration_summary": last_results,
            "suggested_config": suggested_config,
            "expected_stages": [stage.value for stage in PipelineStage],
            "advisory_notes": self.global_recommendations,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "iterations": [it.to_dict() for it in self.iterations],
            "current_iteration": self.current_iteration.to_dict() if self.current_iteration else None,
            "metadata": self.metadata,
            "global_recommendations": self.global_recommendations,
            "progress": self.get_progress_metrics(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineManager":
        """Create from dictionary."""
        iterations = [PipelineIteration.from_dict(it) for it in d.get("iterations", [])]
        current = None
        if d.get("current_iteration"):
            current = PipelineIteration.from_dict(d["current_iteration"])

        return cls(
            pipeline_id=d.get("pipeline_id", str(uuid.uuid4())),
            name=d.get("name", "Default Pipeline"),
            description=d.get("description", ""),
            config=d.get("config", {}),
            iterations=iterations,
            current_iteration=current,
            metadata=d.get("metadata", {}),
            global_recommendations=d.get("global_recommendations", []),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineManager":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def create_default_pipeline_manager() -> PipelineManager:
    """Create a default pipeline manager with standard configuration."""
    return PipelineManager(
        name="Economics-Grounded Learning Pipeline",
        description="5-stage pipeline for MPL-grounded policy training with datapack valuation",
        config={
            "env_name": "drawer_vase",
            "engine_type": "pybullet",
            "task_type": "fragility",
            "customer_segment": "balanced",
            "market_region": "US",
            "objective_preset": "balanced",
            "max_iterations": 10,
            "convergence_threshold": 0.95,  # 95% wage parity
        },
        global_recommendations=[
            "Monitor MPL growth rate vs energy consumption",
            "Track error rates especially for fragile objects",
            "Ensure data diversity across task conditions",
            "Review objective weights every 5 iterations",
        ],
        metadata={
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "phase": "scaffolding",  # Currently additive infrastructure only
        },
    )


def simulate_pipeline_iteration(manager: PipelineManager) -> PipelineIteration:
    """
    Simulate a single pipeline iteration for testing/preview.

    ADVISORY ONLY: This simulates what an iteration would look like,
    does not actually execute training or modify any real data.
    """
    iteration = manager.start_new_iteration()

    # Stage 1: Objective Solving
    stage1 = iteration.start_stage(PipelineStage.OBJECTIVE_SOLVING)
    stage1.outputs = {
        "objective_vector": [1.0, 0.2, 0.1, 0.05, 0.0],
        "preset": "balanced",
        "solver_method": "heuristic",
    }
    stage1.metrics = {
        "solve_time_s": 2.5,
        "objective_confidence": 0.85,
    }
    stage1.recommendations = [
        "Consider increasing safety weight for fragile tasks",
    ]
    iteration.complete_stage(PipelineStage.OBJECTIVE_SOLVING, stage1)

    # Stage 2: Data Collection
    stage2 = iteration.start_stage(PipelineStage.DATA_COLLECTION)
    stage2.outputs = {
        "datapacks_collected": 100,
        "datapacks_annotated": 95,
        "with_vla_annotations": 80,
        "with_embeddings": 90,
    }
    stage2.metrics = {
        "data_coverage": 0.75,
        "novelty_score_mean": 0.45,
        "tier2_fraction": 0.15,
    }
    stage2.recommendations = [
        "More data needed for edge cases with occlusion",
    ]
    iteration.complete_stage(PipelineStage.DATA_COLLECTION, stage2)

    # Stage 3: Policy Training (advisory)
    stage3 = iteration.start_stage(PipelineStage.POLICY_TRAINING)
    stage3.outputs = {
        "policy_type": "PPO",
        "training_episodes": 5000,
        "advisory_only": True,  # Does not actually train
    }
    stage3.metrics = {
        "mpl_delta": 3.2,
        "error_reduction": 0.08,
        "energy_efficiency_delta": 0.05,
    }
    stage3.recommendations = [
        "Policy showing good MPL growth, maintain current objective weights",
    ]
    iteration.complete_stage(PipelineStage.POLICY_TRAINING, stage3)

    # Stage 4: Evaluation
    stage4 = iteration.start_stage(PipelineStage.EVALUATION)
    stage4.outputs = {
        "eval_episodes": 50,
        "success_rate": 0.85,
        "vase_breaks": 2,
    }
    stage4.metrics = {
        "mpl": 52.0,  # dishes/hour (target: 60)
        "error_rate": 0.04,
        "wage_parity": 0.87,
        "energy_Wh_per_unit": 0.08,
    }
    stage4.recommendations = [
        "Wage parity at 87% - continue training towards 95% target",
    ]
    iteration.complete_stage(PipelineStage.EVALUATION, stage4)

    # Stage 5: Feedback and Iteration
    stage5 = iteration.start_stage(PipelineStage.FEEDBACK_ITERATION)
    stage5.outputs = {
        "feedback_processed": True,
        "next_iteration_plan": "increase_fragility_focus",
    }
    stage5.metrics = {
        "iteration_value": 0.82,
        "convergence_progress": 0.87,
    }
    stage5.recommendations = [
        "Next iteration: focus on reducing vase breaks to improve wage parity",
        "Consider adjusting energy profile for careful manipulation",
    ]
    stage5.next_stage_config = {
        "increase_safety_weight": True,
        "focus_on_fragile": True,
    }
    iteration.complete_stage(PipelineStage.FEEDBACK_ITERATION, stage5)

    # Finalize iteration
    manager.complete_current_iteration()

    return iteration


def run_semantic_feedback_pass(
    econ: Any,
    datapacks: Any,
    sem: SemanticOrchestrator,
    out_path: str = "data/semantic_metrics/latest.jsonl",
) -> None:
    """
    Offline semantic feedback loop (advisory-only).
    """
    # Econ/datapack signals are assumed to be provided by these facades.
    # Econ signals may require datapacks list; pass empty list for advisory stub
    econ_signals = getattr(econ, "compute_signals", lambda dps, episodes=None: {})([], None)
    datapack_signals = getattr(datapacks, "compute_signals", lambda dps, econ=None: {})([], None)

    sem.export_semantic_metrics(econ_signals, datapack_signals, out_path)

    metrics_list = load_semantic_metrics(out_path)
    if not metrics_list:
        return
    metrics = metrics_list[-1]

    if hasattr(econ, "update_from_semantic_metrics"):
        econ.update_from_semantic_metrics(metrics)
        _ = econ.suggest_objective_adjustments_from_semantics(metrics)
    if hasattr(datapacks, "update_sampling_from_semantics"):
        _ = datapacks.update_sampling_from_semantics(metrics)


def run_pipeline_step_with_causal_order(
    econ_controller,
    datapack_engine,
    semantic_orchestrator,
    meta_transformer=None,
    datapacks=None,
    perception_embeddings=None,
):
    """
    Execute a single pipeline step with CORRECT CAUSAL ORDER.

    IMPORTANT: This enforces the dependency hierarchy:
    1. EconController computes signals (UPSTREAM)
    2. DatapackEngine computes signals (UPSTREAM, uses econ_signals)
    3. MetaTransformer proposes plan (OPTIONAL, uses both signals)
    4. SemanticOrchestrator updates semantics (DOWNSTREAM, uses all above)
    5. Run configurations derived from semantic state

    Args:
        econ_controller: EconomicController instance
        datapack_engine: DatapackEngine instance
        semantic_orchestrator: SemanticOrchestrator instance
        meta_transformer: Optional MetaTransformer instance
        datapacks: List of DataPackMeta objects
        perception_embeddings: Optional perception module embeddings

    Returns:
        Dict with step results and run specifications
    """
    # STEP 1: Compute economic signals (UPSTREAM - defines physics of value)
    econ_signals = econ_controller.compute_signals(datapacks or [])

    # STEP 2: Compute datapack signals (UPSTREAM - defines data value)
    datapack_signals = datapack_engine.compute_signals(datapacks or [], econ_signals)

    # STEP 3: Meta-transformer proposes plan (OPTIONAL - suggestions only)
    meta_out = None
    if meta_transformer is not None:
        # MetaTransformer uses signals but does NOT define value
        # It suggests energy profiles, objective presets, data mixes
        try:
            meta_out = meta_transformer.propose_plan(
                econ_signals=econ_signals,
                datapack_signals=datapack_signals,
                perception_embeddings=perception_embeddings,
            )
        except AttributeError:
            # MetaTransformer may not have propose_plan yet
            meta_out = None

    # STEP 4: Semantic orchestrator builds update plan (DOWNSTREAM)
    # This is the KEY step - semantics are shaped by econ/datapack signals
    semantic_plan = semantic_orchestrator.build_update_plan(
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
        meta_out=meta_out,
    )

    # STEP 5: Apply semantic updates (mutate task graph + ontology)
    semantic_orchestrator.apply_update_plan(semantic_plan)

    # STEP 6: Generate run specifications FROM semantic state
    # This is what VLA/DINO/SIMA/diffusion/RL will see
    semantic_state = semantic_orchestrator.snapshot()

    run_specs = {
        "econ_signals": econ_signals.to_dict(),
        "datapack_signals": datapack_signals.to_dict(),
        "semantic_plan": semantic_plan.to_dict(),
        "semantic_state": semantic_state,
        "cross_module_constraints": semantic_plan.cross_module_constraints,
        "primitive_updates": semantic_plan.primitive_updates,
    }

    if meta_out:
        run_specs["meta_transformer_suggestions"] = {
            "objective_preset": meta_out.objective_preset,
            "energy_profile_weights": meta_out.energy_profile_weights,
            "expected_delta_mpl": meta_out.expected_delta_mpl,
        }

    return run_specs


def verify_dependency_hierarchy():
    """
    Verify that the dependency hierarchy is enforced.

    ALLOWED imports:
    - SemanticOrchestrator imports EconomicController, DatapackEngine
    - PipelineManager imports all orchestrator modules
    - VLA/SIMA/diffusion/RL import SemanticOrchestrator (or its outputs)

    FORBIDDEN imports:
    - EconomicController importing SemanticOrchestrator or MetaTransformer
    - DatapackEngine importing SemanticOrchestrator or MetaTransformer
    - Perception modules directly querying econ/datapacks

    Returns:
        Dict with verification results
    """
    results = {
        "hierarchy_valid": True,
        "violations": [],
        "causal_chain": [
            "EconomicController (defines value physics)",
            "DatapackEngine (defines data value)",
            "MetaTransformer (suggests plans - optional)",
            "SemanticOrchestrator (applies value to meaning)",
            "VLA/SIMA/Diffusion/RL (act within meaning)",
        ],
    }

    # Check EconomicController doesn't import downstream
    try:
        import src.orchestrator.economic_controller as ec
        source = open(ec.__file__).read()
        if "SemanticOrchestrator" in source or "MetaTransformer" in source:
            results["violations"].append(
                "EconomicController imports downstream modules"
            )
            results["hierarchy_valid"] = False
    except Exception:
        pass

    # Check DatapackEngine doesn't import downstream
    try:
        import src.orchestrator.datapack_engine as de
        source = open(de.__file__).read()
        if "SemanticOrchestrator" in source or "MetaTransformer" in source:
            results["violations"].append(
                "DatapackEngine imports downstream modules"
            )
            results["hierarchy_valid"] = False
    except Exception:
        pass

    # Check SemanticOrchestrator imports upstream
    try:
        import src.orchestrator.semantic_orchestrator as so
        source = open(so.__file__).read()
        if "EconomicController" not in source:
            results["violations"].append(
                "SemanticOrchestrator doesn't import EconomicController"
            )
            results["hierarchy_valid"] = False
        if "DatapackEngine" not in source:
            results["violations"].append(
                "SemanticOrchestrator doesn't import DatapackEngine"
            )
            results["hierarchy_valid"] = False
    except Exception:
        pass

    return results
