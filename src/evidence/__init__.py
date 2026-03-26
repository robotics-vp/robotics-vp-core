"""Evidence-layer scaffolding for economic-world-model readiness."""

from src.evidence.benchmark_gating import (
    build_benchmark_gate_report,
    collect_benchmark_gating_signals,
)
from src.evidence.belief_state import BeliefState, belief_state_from_evidence_bus
from src.evidence.bus import EvidenceBus, EvidenceRecord
from src.evidence.preconditions import (
    ExecutionPreconditionsReport,
    ExecutionWorkOrder,
    PreconditionCheck,
    build_execution_preconditions,
    build_execution_work_order,
    summarize_execution_preconditions,
)
from src.evidence.scene_tracks_truth import normalize_scene_tracks_truth
from src.evidence.teacher_trace import (
    TeacherStep,
    TeacherTrace,
    load_teacher_trace_json,
    save_teacher_trace_json,
)

__all__ = [
    "BeliefState",
    "EvidenceBus",
    "EvidenceRecord",
    "ExecutionPreconditionsReport",
    "ExecutionWorkOrder",
    "PreconditionCheck",
    "TeacherStep",
    "TeacherTrace",
    "belief_state_from_evidence_bus",
    "build_benchmark_gate_report",
    "build_execution_preconditions",
    "build_execution_work_order",
    "collect_benchmark_gating_signals",
    "load_teacher_trace_json",
    "normalize_scene_tracks_truth",
    "save_teacher_trace_json",
    "summarize_execution_preconditions",
]
