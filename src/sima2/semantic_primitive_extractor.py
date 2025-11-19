"""
Rule-based semantic primitive extractor for stubbed SIMA-2 rollouts.

This stays dependency-light and only emits advisory ontology update proposals;
it does not mutate any global ontology state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from src.sima2.config import extract_provenance


@dataclass
class SemanticPrimitive:
    """Represents a discovered semantic primitive from a SIMA-2 rollout."""

    primitive_id: str
    task_type: str
    tags: List[str]
    risk_level: str
    energy_intensity: float
    success_rate: float
    avg_steps: float
    source: str = "sima2"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """Lightweight validation for primitives."""
        errors: List[str] = []
        if not self.primitive_id:
            errors.append("primitive_id is required")
        if not self.task_type:
            errors.append("task_type is required")
        if not self.tags:
            errors.append("tags must be non-empty")
        if self.risk_level not in {"low", "medium", "high"}:
            errors.append(f"risk_level '{self.risk_level}' must be one of low/medium/high")
        if self.energy_intensity is None:
            errors.append("energy_intensity is required")
        if not 0.0 <= float(self.success_rate) <= 1.0:
            errors.append("success_rate must be within [0,1]")
        if self.avg_steps <= 0:
            errors.append("avg_steps must be > 0")
        return (len(errors) == 0, errors)


def _split_to_tags(value: str) -> List[str]:
    """Split a string into lightweight tags."""
    tokens: List[str] = []
    for chunk in value.replace("-", "_").replace("/", "_").split("_"):
        part = chunk.strip().lower()
        if part:
            tokens.append(part)
    return tokens


def _collect_event_tags(event: Dict[str, Any]) -> Set[str]:
    """Extract tag-like fields from an event dict."""
    tags: Set[str] = set()
    for key in ("tags", "labels"):
        for tag in event.get(key, []) or []:
            tags.update(_split_to_tags(str(tag)))
    for key in ("object", "target", "tool", "type", "name", "action"):
        if key in event and event[key]:
            tags.update(_split_to_tags(str(event[key])))
    return tags


def _events_from_segments(segments: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert segments into primitive-like events for downstream extraction."""
    events: List[Dict[str, Any]] = []
    for seg in segments or []:
        seg_dict = seg.to_dict() if hasattr(seg, "to_dict") else seg
        if not isinstance(seg_dict, dict):
            continue
        events.append(
            {
                "primitive_id": seg_dict.get("segment_id") or seg_dict.get("id"),
                "action": seg_dict.get("label"),
                "object": seg_dict.get("object") or seg_dict.get("label"),
                "timestep": seg_dict.get("start_t", 0),
                "end_timestep": seg_dict.get("end_t"),
                "success": str(seg_dict.get("outcome", "")).lower() in {"success", "recovered"},
                "status": seg_dict.get("outcome"),
                "tags": [seg_dict.get("label"), seg_dict.get("outcome")],
                "risk": seg_dict.get("metadata", {}).get("risk_level") if isinstance(seg_dict.get("metadata"), dict) else None,
                "metadata": seg_dict.get("metadata", {}),
                "segment_id": seg_dict.get("segment_id") or seg_dict.get("id"),
            }
        )
    return events


def _infer_risk_level(tags: Sequence[str], metrics: Dict[str, Any]) -> str:
    """Derive a simple categorical risk level."""
    tag_set = {t.lower() for t in tags}
    if "fragile" in tag_set or metrics.get("fragile_interaction"):
        return "high"
    if metrics.get("collisions", 0) > 0 or "collision" in tag_set:
        return "medium"
    energy = metrics.get("energy_intensity") or metrics.get("energy_used", 0.0)
    if energy and energy > 5.0:
        return "medium"
    return "low"


def _infer_energy_intensity(event: Dict[str, Any], metrics: Dict[str, Any], steps: float) -> float:
    if "energy_intensity" in event:
        return float(event["energy_intensity"])
    if "energy_intensity" in metrics:
        return float(metrics["energy_intensity"])
    energy_used = metrics.get("energy_used")
    if energy_used is not None and steps:
        return float(energy_used) / float(steps)
    return 0.0


def _infer_success_rate(event: Dict[str, Any], metrics: Dict[str, Any]) -> float:
    if "success_rate" in event:
        return float(event["success_rate"])
    if "success" in event:
        return 1.0 if event.get("success") else 0.0
    if "success_rate" in metrics:
        return float(metrics["success_rate"])
    if "success" in metrics:
        return 1.0 if metrics.get("success") else 0.0
    return 1.0


def _infer_avg_steps(event: Dict[str, Any], metrics: Dict[str, Any], total_events: int) -> float:
    if "steps" in event:
        return float(event["steps"])
    if "avg_steps" in metrics:
        return float(metrics["avg_steps"])
    if "steps" in metrics:
        return float(metrics["steps"])
    if total_events:
        return float(total_events)
    return 1.0


def extract_primitives_from_rollout(rollout: Dict[str, Any]) -> List[SemanticPrimitive]:
    """
    Extract semantic primitives from a stubbed SIMA-2 rollout.

    Uses simple deterministic rules so tests remain stable without SIMA-2 deps.
    """
    task_type = str(rollout.get("task_type") or rollout.get("task") or "unknown")
    events = rollout.get("events") or _events_from_segments(rollout.get("segments", []))
    metrics = rollout.get("metrics") or {}
    base_tags: Set[str] = set(_split_to_tags(task_type))
    for tag in rollout.get("tags", []) or []:
        base_tags.update(_split_to_tags(str(tag)))
    provenance = extract_provenance(rollout)

    primitives: List[SemanticPrimitive] = []
    for idx, event in enumerate(events):
        event_tags = base_tags | _collect_event_tags(event)
        event_tags = event_tags or {"generic"}
        avg_steps = _infer_avg_steps(event, metrics, len(events))
        primitives.append(
            SemanticPrimitive(
                primitive_id=str(
                    event.get("primitive_id")
                    or event.get("name")
                    or event.get("action")
                    or f"{task_type}_evt_{idx}"
                ),
                task_type=task_type,
                tags=sorted(event_tags),
                risk_level=_infer_risk_level(event_tags, metrics),
                energy_intensity=_infer_energy_intensity(event, metrics, avg_steps),
                success_rate=_infer_success_rate(event, metrics),
                avg_steps=avg_steps,
                source=str(provenance.get("sima2_backend_id") or rollout.get("source", "sima2")),
                provenance=provenance,
            )
        )

    if primitives:
        return _sort_primitives_deterministically(primitives)

    fallback_tags = base_tags or {"generic"}
    primitives.append(
        SemanticPrimitive(
            primitive_id=f"{task_type}_primitive_0",
            task_type=task_type,
            tags=sorted(fallback_tags),
            risk_level=_infer_risk_level(fallback_tags, metrics),
            energy_intensity=_infer_energy_intensity({}, metrics, len(events)),
            success_rate=_infer_success_rate({}, metrics),
            avg_steps=_infer_avg_steps({}, metrics, len(events)),
            source=str(provenance.get("sima2_backend_id") or rollout.get("source", "sima2")),
            provenance=provenance,
        )
    )
    return _sort_primitives_deterministically(primitives)


def primitive_to_ontology_update(primitive: SemanticPrimitive) -> Dict[str, Any]:
    """
    Convert a semantic primitive into a canonical ontology update proposal.

    The update is advisory-only; the orchestrator can merge or reject it.
    """
    update = {
        "action": "add_or_update_skill",
        "skill_name": primitive.task_type or primitive.primitive_id,
        "primitive_id": primitive.primitive_id,
        "risk_level": primitive.risk_level,
        "tags": primitive.tags,
        "source": primitive.source,
        "estimates": {
            "energy_intensity": primitive.energy_intensity,
            "success_rate": primitive.success_rate,
            "avg_steps": primitive.avg_steps,
        },
    }
    if primitive.provenance:
        update["provenance"] = primitive.provenance
    return update


def _sort_primitives_deterministically(primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
    """Return primitives sorted for deterministic ordering."""
    return sorted(
        primitives,
        key=lambda p: (
            p.task_type or "",
            p.primitive_id or "",
        ),
    )


class SemanticPrimitiveExtractor:
    """Lightweight wrapper to extract primitives from multiple rollouts."""

    def extract(self, rollouts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        primitives: List[Dict[str, Any]] = []
        for rollout in rollouts:
            prims = extract_primitives_from_rollout(rollout)
            for prim in prims:
                prov = prim.provenance or extract_provenance(rollout)
                primitives.append(
                    {
                        "episode_id": rollout.get("episode_id"),
                        "task": rollout.get("task"),
                        "primitive_id": prim.primitive_id,
                        "task_type": prim.task_type,
                        "tags": prim.tags,
                        "risk_level": prim.risk_level,
                        "energy_intensity": prim.energy_intensity,
                        "success_rate": prim.success_rate,
                        "avg_steps": prim.avg_steps,
                        "source": prim.source,
                        "provenance": prov,
                        "sima2_backend_id": prov.get("sima2_backend_id"),
                        "sima2_model_version": prov.get("sima2_model_version"),
                        "sima2_task_spec": prov.get("sima2_task_spec"),
                        "sima2_backend_mode": prov.get("sima2_backend_mode"),
                    }
                )
        return sorted(primitives, key=lambda p: (p.get("episode_id", ""), p.get("primitive_id", "")))
