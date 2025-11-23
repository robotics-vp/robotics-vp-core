"""
ConditionVectorBuilder: single source of truth for constructing ConditionVector.

Combines episode/task metadata, econ state, curriculum phase, SIMA-2 trust,
and datapack tags. Reads inputs, never mutates them, and falls back to
deterministic defaults when fields are missing.
"""
from typing import Any, Dict, Optional, Sequence

from src.observation.condition_vector import ConditionVector, _flatten_sequence
from src.rl.skill_mode_resolver import SkillModeResolver, resolve_skill_mode


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Graceful attribute/dict lookup."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ConditionVectorBuilder:
    """
    Builds a ConditionVector per episode/rollout.

    This is the only fusion point for task/env metadata, econ state, and
    semantic curriculum signals.
    """

    DEFAULT_SKILL_MODE = "efficiency_throughput"
    DEFAULT_OBJECTIVE = "balanced"
    DEFAULT_RECAP_BUCKET = "bronze"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.skill_mode_order = self.config.get("skill_mode_order") or [
            "frontier_exploration",
            "safety_critical",
            "efficiency_throughput",
            "recovery_heavy",
            "default",
        ]
        self.skill_resolver = SkillModeResolver(
            default_mode=self.DEFAULT_SKILL_MODE,
            mode_order=self.skill_mode_order,
        )

    def build(
        self,
        *,
        episode_config: Any,
        econ_state: Any,
        curriculum_phase: str,
        sima2_trust: Optional[Any],
        datapack_metadata: Optional[Dict[str, Any]] = None,
        episode_step: int = 0,
        overrides: Optional[Dict[str, Any]] = None,
        econ_slice: Optional[Any] = None,
        semantic_tags: Optional[Dict[str, float]] = None,
        recap_scores: Optional[Any] = None,
        trust_summary: Optional[Dict[str, Any]] = None,
        episode_metadata: Optional[Dict[str, Any]] = None,
        advisory_context: Optional[Dict[str, Any]] = None,
        tfd_instruction: Optional[Dict[str, Any]] = None,
        enable_tfd_integration: bool = False,
    ) -> ConditionVector:
        """
        Construct a ConditionVector with deterministic fallbacks.

        Flag-gated TFD integration:
        - enable_tfd_integration: if True, TFD fields influence condition vector
        - Default False preserves existing behavior
        """
        overrides = overrides or {}
        meta = datapack_metadata or {}
        semantic_tags = semantic_tags or {}
        episode_metadata = episode_metadata or {}
        advisory_context = advisory_context or {}

        # Store TFD instruction in metadata for logging
        if tfd_instruction:
            episode_metadata = dict(episode_metadata)
            episode_metadata["tfd_instruction"] = tfd_instruction

        # Extract TFD condition vector if present and integration enabled
        tfd_cv = None
        if enable_tfd_integration and tfd_instruction:
            tfd_cv = self._extract_tfd_condition_vector(tfd_instruction)

        phase = str(overrides.get("curriculum_phase", curriculum_phase or "warmup"))
        tags = self._merge_tags(meta.get("tags"), semantic_tags)
        trust_score = self._summarize_trust(trust_summary, sima2_trust)

        # Extract SIMA-2 OOD/Recovery signals from tags and metadata
        ood_signals = self._extract_ood_signals(tags, meta, episode_metadata)
        recovery_signals = self._extract_recovery_signals(tags, meta, episode_metadata)

        # Apply TFD phase override if present
        if tfd_cv and tfd_cv.get("curriculum_phase"):
            phase = str(tfd_cv.get("curriculum_phase"))

        recap_bucket = overrides.get("recap_goodness_bucket")
        if recap_bucket is None:
            recap_bucket = self._bucketize_recap(recap_scores)

        sampler_strategy = episode_metadata.get("sampler_strategy") or meta.get("sampler_strategy")

        skill_mode = overrides.get("skill_mode") or advisory_context.get("skill_mode")
        # TFD skill_mode takes precedence if integration enabled
        if tfd_cv and tfd_cv.get("skill_mode"):
            skill_mode = str(tfd_cv.get("skill_mode"))
        elif not skill_mode:
            skill_mode = self.skill_resolver.resolve(
                tags=tags,
                trust_matrix=trust_summary,
                curriculum_phase=phase,
                advisory=advisory_context,
                econ_slice=econ_slice if isinstance(econ_slice, dict) else None,
                recap_bucket=recap_bucket,
                strategy=sampler_strategy,
            )

        objective_vector = overrides.get("objective_vector") or _get(episode_config, "objective_vector")
        # TFD objective_vector takes precedence if available
        if tfd_cv and tfd_cv.get("objective_vector"):
            obj_vec_dict = tfd_cv.get("objective_vector") or {}
            if isinstance(obj_vec_dict, dict) and obj_vec_dict:
                objective_vector = list(obj_vec_dict.values())
        objective_vector = _flatten_sequence(objective_vector) if objective_vector is not None else None

        novelty_tier = overrides.get("novelty_tier")
        # TFD novelty_tier takes precedence if available
        if tfd_cv and tfd_cv.get("novelty_tier") is not None:
            novelty_tier = int(tfd_cv.get("novelty_tier"))
        elif novelty_tier is None:
            novelty_tier = self._novelty_tier_from_context(advisory_context, tags, recap_scores)

        episode_step_val = overrides.get("episode_step")
        if episode_step_val is None:
            episode_step_val = episode_metadata.get("timestep", episode_metadata.get("step", episode_step))

        wage_parity = overrides.get("current_wage_parity", _get(econ_state, "current_wage_parity", 0.0))
        wage_parity = self._safe_float(wage_parity, self._safe_float(_get(econ_slice, "wage_parity", None)))

        # Apply TFD economic fields if available
        target_mpl_val = overrides.get("target_mpl", _get(econ_state, "target_mpl", _get(econ_slice, "mpl", 0.0)))
        if tfd_cv and tfd_cv.get("target_mpl_uplift") is not None:
            current_mpl = self._safe_float(_get(econ_slice, "mpl", 0.0))
            target_mpl_val = current_mpl + self._safe_float(tfd_cv.get("target_mpl_uplift"))

        energy_budget_val = overrides.get("energy_budget_wh", _get(econ_state, "energy_budget_wh", _get(econ_slice, "energy_wh", 0.0)))
        if tfd_cv and tfd_cv.get("energy_budget_wh") is not None:
            energy_budget_val = self._safe_float(tfd_cv.get("energy_budget_wh"))

        return ConditionVector(
            task_id=str(overrides.get("task_id") or _get(episode_config, "task_id", "")),
            env_id=str(overrides.get("env_id") or _get(episode_config, "env_id", "")),
            backend_id=str(overrides.get("backend_id") or _get(episode_config, "backend_id", _get(episode_config, "backend", ""))),
            target_mpl=self._safe_float(target_mpl_val),
            current_wage_parity=self._safe_float(wage_parity),
            energy_budget_wh=self._safe_float(energy_budget_val),
            skill_mode=str(skill_mode or self.DEFAULT_SKILL_MODE),
            ood_risk_level=self._compute_ood_risk_level(
                overrides=overrides,
                meta=meta,
                episode_metadata=episode_metadata,
                ood_signals=ood_signals,
            ),
            recovery_priority=self._compute_recovery_priority(
                overrides=overrides,
                meta=meta,
                episode_metadata=episode_metadata,
                recovery_signals=recovery_signals,
            ),
            novelty_tier=int(novelty_tier or 0),
            sima2_trust_score=self._safe_float(overrides.get("sima2_trust_score", trust_score)),
            recap_goodness_bucket=str(recap_bucket or self.DEFAULT_RECAP_BUCKET),
            objective_preset=str(overrides.get("objective_preset", _get(episode_config, "objective_preset", self.DEFAULT_OBJECTIVE))),
            objective_vector=objective_vector,
            episode_step=int(episode_step_val or 0),
            curriculum_phase=str(phase),
            metadata=self._build_metadata(
                datapack_metadata=meta,
                episode_metadata=episode_metadata,
                econ_slice=econ_slice,
                semantic_tags=tags,
                recap_scores=recap_scores,
                advisory_context=advisory_context,
                sampler_strategy=sampler_strategy,
            ),
        )

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            try:
                return float(default)
            except Exception:
                return 0.0

    def _get_trust(self, sima2_trust: Optional[Any]) -> float:
        if sima2_trust is None:
            return 0.0
        if isinstance(sima2_trust, (int, float)):
            return float(sima2_trust)
        if isinstance(sima2_trust, dict) and "trust_score" in sima2_trust:
            return float(sima2_trust.get("trust_score", 0.0))
        return float(_get(sima2_trust, "trust_score", 0.0))

    def _build_metadata(
        self,
        *,
        datapack_metadata: Dict[str, Any],
        episode_metadata: Dict[str, Any],
        econ_slice: Optional[Any],
        semantic_tags: Dict[str, float],
        recap_scores: Optional[Any],
        advisory_context: Dict[str, Any],
        sampler_strategy: Optional[str],
    ) -> Dict[str, Any]:
        # Keep only JSON-safe, low-risk fields
        allowed_keys = ["tags", "datapack_id", "backend_id", "phase", "pack_tier", "pack_id", "tfd_instruction"]
        meta: Dict[str, Any] = {k: v for k, v in (datapack_metadata or {}).items() if k in allowed_keys}
        if episode_metadata.get("episode_id"):
            meta["episode_id"] = episode_metadata["episode_id"]
        if episode_metadata.get("tfd_instruction") is not None:
            # Store compact TFD metadata for logging
            tfd_inst = episode_metadata.get("tfd_instruction")
            tfd_metadata = self._build_tfd_metadata(tfd_inst)
            if tfd_metadata:
                meta["tfd_metadata"] = tfd_metadata
        if sampler_strategy:
            meta["sampler_strategy"] = sampler_strategy
        if advisory_context:
            meta["advisory"] = {
                "frontier_score": advisory_context.get("frontier_score"),
                "priority": advisory_context.get("priority"),
                "skill_mode": advisory_context.get("skill_mode"),
            }
        if semantic_tags:
            meta["semantic_tags"] = {k: float(v) for k, v in sorted(semantic_tags.items(), key=lambda kv: kv[0])}
        if recap_scores is not None:
            try:
                recap_score = _get(recap_scores, "recap_goodness_score", recap_scores if isinstance(recap_scores, (int, float)) else None)
                meta["recap"] = {"recap_goodness_score": float(recap_score or 0.0)}
            except Exception:
                pass
        if econ_slice is not None:
            econ_payload = {
                "mpl": self._safe_float(_get(econ_slice, "mpl", None)),
                "energy_wh": self._safe_float(_get(econ_slice, "energy_wh", None)),
                "damage_cost": self._safe_float(_get(econ_slice, "damage_cost", None)),
                "wage_parity": self._safe_float(_get(econ_slice, "wage_parity", None)),
            }
            meta["econ_slice"] = econ_payload
        return meta

    def _summarize_trust(self, trust_summary: Optional[Dict[str, Any]], sima2_trust: Optional[Any]) -> float:
        if trust_summary and isinstance(trust_summary, dict) and trust_summary:
            try:
                vals = [float(v) for v in trust_summary.values() if isinstance(v, (int, float))]
                if vals:
                    return float(sum(vals) / len(vals))
            except Exception:
                pass
        return self._get_trust(sima2_trust)

    def _merge_tags(self, datapack_tags: Optional[Sequence[Any]], semantic_tags: Dict[str, float]) -> Dict[str, float]:
        tags: Dict[str, float] = {}
        for tag in datapack_tags or []:
            tags[str(tag)] = tags.get(str(tag), 0.0) + 1.0
        for key, val in semantic_tags.items():
            try:
                tags[str(key)] = tags.get(str(key), 0.0) + float(val)
            except Exception:
                continue
        return tags

    def _bucketize_recap(self, recap_scores: Optional[Any]) -> str:
        score = None
        if recap_scores is None:
            return self.DEFAULT_RECAP_BUCKET
        if isinstance(recap_scores, (int, float)):
            score = float(recap_scores)
        elif isinstance(recap_scores, dict):
            score = recap_scores.get("recap_goodness_score")
        else:
            score = _get(recap_scores, "recap_goodness_score")
        if score is None:
            return self.DEFAULT_RECAP_BUCKET
        try:
            score_f = float(score)
        except Exception:
            return self.DEFAULT_RECAP_BUCKET
        if score_f >= 0.85:
            return "platinum"
        if score_f >= 0.65:
            return "gold"
        if score_f >= 0.45:
            return "silver"
        return self.DEFAULT_RECAP_BUCKET

    def _novelty_tier_from_context(
        self,
        advisory_context: Dict[str, Any],
        semantic_tags: Dict[str, float],
        recap_scores: Optional[Any],
    ) -> int:
        frontier_score = advisory_context.get("frontier_score")
        if frontier_score is not None:
            try:
                return int(min(3, max(0, round(float(frontier_score)))))
            except Exception:
                pass
        if recap_scores and isinstance(recap_scores, dict):
            try:
                adv_bins = recap_scores.get("advantage_bin_probs") or []
                if adv_bins:
                    return int(min(len(adv_bins), max(0, adv_bins.index(max(adv_bins))) + 1))
            except Exception:
                pass
        if semantic_tags:
            try:
                return int(min(3, max(0, round(max(semantic_tags.values())))))
            except Exception:
                return 0
        return 0

    def _extract_tfd_condition_vector(self, tfd_instruction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract TFDConditionVector from TFDInstruction or TFDSession and convert to dict.

        Returns a dict with TFD advisory fields that can override ConditionVector fields.

        Handles three cases:
        1. TFDSession object (get canonical instruction)
        2. TFDInstruction object or dict
        3. Raw dict with condition_vector field
        """
        if not tfd_instruction:
            return None

        # Case 1: Handle TFDSession (get canonical instruction)
        if hasattr(tfd_instruction, "get_canonical_condition_vector"):
            try:
                tfd_cv = tfd_instruction.get_canonical_condition_vector()
                if tfd_cv is not None:
                    if hasattr(tfd_cv, "to_dict"):
                        return tfd_cv.to_dict()
                    elif isinstance(tfd_cv, dict):
                        return tfd_cv
            except Exception:
                pass

        # Case 2 & 3: Handle TFDInstruction or raw dict
        if isinstance(tfd_instruction, dict):
            condition_vector = tfd_instruction.get("condition_vector")
            if condition_vector is None:
                return None
            # If it's already a dict, return it
            if isinstance(condition_vector, dict):
                return condition_vector
            # If it's an object, try to call to_dict()
            if hasattr(condition_vector, "to_dict"):
                try:
                    return condition_vector.to_dict()
                except Exception:
                    return None
        return None

    def _build_tfd_metadata(self, tfd_instruction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build compact TFD metadata for logging in ConditionVector.metadata and episode logs.

        Returns JSON-safe dict with intent_type, status, and key parsed parameters.

        Handles both TFDSession and TFDInstruction.
        """
        if not tfd_instruction:
            return None

        metadata = {}

        # Handle TFDSession (get canonical instruction and session summary)
        if hasattr(tfd_instruction, "get_session_summary"):
            try:
                session_summary = tfd_instruction.get_session_summary()
                if session_summary:
                    metadata.update(session_summary)
                    # Get canonical instruction for detailed metadata
                    canonical_inst = tfd_instruction.get_canonical_instruction()
                    if canonical_inst:
                        tfd_instruction = canonical_inst
                    else:
                        return metadata if metadata else None
            except Exception:
                pass

        # Extract from TFDInstruction (handles both object and dict)
        if hasattr(tfd_instruction, "to_dict"):
            try:
                inst_dict = tfd_instruction.to_dict()
                if inst_dict:
                    tfd_instruction = inst_dict
            except Exception:
                pass

        # Extract status
        status = tfd_instruction.get("status") if isinstance(tfd_instruction, dict) else None
        if status:
            metadata["status"] = str(status)

        # Extract raw text
        raw_text = tfd_instruction.get("raw_text") if isinstance(tfd_instruction, dict) else None
        if raw_text:
            metadata["raw_text"] = str(raw_text)

        # Extract parsed intent if present
        parsed_intent = tfd_instruction.get("parsed_intent") if isinstance(tfd_instruction, dict) else None
        if parsed_intent:
            if isinstance(parsed_intent, dict):
                intent_type = parsed_intent.get("intent_type")
                if intent_type:
                    metadata["intent_type"] = str(intent_type)
                parameters = parsed_intent.get("parameters")
                if parameters and isinstance(parameters, dict):
                    metadata["parameters"] = dict(parameters)
            elif hasattr(parsed_intent, "intent_type"):
                # Handle ParsedIntent object
                metadata["intent_type"] = str(parsed_intent.intent_type) if parsed_intent.intent_type else None
                if hasattr(parsed_intent, "parameters"):
                    metadata["parameters"] = dict(parsed_intent.parameters or {})

        # Extract advisory fields from condition_vector
        tfd_cv = self._extract_tfd_condition_vector(tfd_instruction)
        if tfd_cv:
            advisory_fields = {}
            for field in ["novelty_bias", "safety_emphasis", "exploration_priority", "efficiency_preference", "fragility_avoidance"]:
                val = tfd_cv.get(field)
                if val is not None:
                    try:
                        advisory_fields[field] = float(val)
                    except Exception:
                        pass
            if advisory_fields:
                metadata["advisory_fields"] = advisory_fields

        return metadata if metadata else None

    def _extract_ood_signals(
        self,
        tags: Dict[str, float],
        datapack_metadata: Dict[str, Any],
        episode_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract OOD (Out-Of-Distribution) signals from SIMA-2 tags and metadata.

        Returns dict with:
        - ood_tag_count: number of OOD tags
        - avg_severity: average severity across OOD tags
        - max_severity: maximum severity
        - sources: list of OOD sources (visual, kinematic, temporal)
        """
        signals = {
            "ood_tag_count": 0,
            "avg_severity": 0.0,
            "max_severity": 0.0,
            "sources": [],
        }

        # Look for OOD tags in semantic tags
        ood_tags = []
        for tag_name, tag_value in tags.items():
            if "ood" in tag_name.lower() or "out_of_distribution" in tag_name.lower():
                ood_tags.append({"name": tag_name, "value": tag_value})

        # Also check metadata for ood_tags list
        metadata_ood_tags = datapack_metadata.get("ood_tags") or episode_metadata.get("ood_tags") or []
        for ood_tag in metadata_ood_tags:
            if isinstance(ood_tag, dict):
                ood_tags.append(ood_tag)

        if not ood_tags:
            return signals

        signals["ood_tag_count"] = len(ood_tags)

        severities = []
        sources = []
        for tag in ood_tags:
            if isinstance(tag, dict):
                severity = self._safe_float(tag.get("severity", tag.get("value", 0.0)))
                severities.append(severity)
                source = tag.get("source")
                if source and source not in sources:
                    sources.append(str(source))
            else:
                severities.append(1.0)  # Default severity for boolean tags

        if severities:
            signals["avg_severity"] = sum(severities) / len(severities)
            signals["max_severity"] = max(severities)
        signals["sources"] = sources

        return signals

    def _extract_recovery_signals(
        self,
        tags: Dict[str, float],
        datapack_metadata: Dict[str, Any],
        episode_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract Recovery signals from SIMA-2 tags and metadata.

        Returns dict with:
        - recovery_tag_count: number of recovery tags
        - high_value_count: count of high-value recoveries
        - avg_cost_wh: average energy cost of recoveries
        - correction_types: list of correction types seen
        """
        signals = {
            "recovery_tag_count": 0,
            "high_value_count": 0,
            "avg_cost_wh": 0.0,
            "correction_types": [],
        }

        # Look for recovery tags in semantic tags
        recovery_tags = []
        for tag_name, tag_value in tags.items():
            if "recovery" in tag_name.lower():
                recovery_tags.append({"name": tag_name, "value": tag_value})

        # Also check metadata for recovery_tags list
        metadata_recovery_tags = datapack_metadata.get("recovery_tags") or episode_metadata.get("recovery_tags") or []
        for recovery_tag in metadata_recovery_tags:
            if isinstance(recovery_tag, dict):
                recovery_tags.append(recovery_tag)

        if not recovery_tags:
            return signals

        signals["recovery_tag_count"] = len(recovery_tags)

        costs = []
        correction_types = []
        high_value_count = 0

        for tag in recovery_tags:
            if isinstance(tag, dict):
                value_add = tag.get("value_add")
                if value_add == "high":
                    high_value_count += 1

                cost_wh = self._safe_float(tag.get("cost_wh", 0.0))
                if cost_wh > 0:
                    costs.append(cost_wh)

                correction_type = tag.get("correction_type")
                if correction_type and correction_type not in correction_types:
                    correction_types.append(str(correction_type))

        signals["high_value_count"] = high_value_count
        if costs:
            signals["avg_cost_wh"] = sum(costs) / len(costs)
        signals["correction_types"] = correction_types

        return signals

    def _compute_ood_risk_level(
        self,
        overrides: Dict[str, Any],
        meta: Dict[str, Any],
        episode_metadata: Dict[str, Any],
        ood_signals: Dict[str, Any],
    ) -> float:
        """
        Compute ood_risk_level from OOD signals with trust-gating.

        Logic:
        - Use override if present
        - Otherwise use metadata ood_score if present
        - Otherwise compute from ood_signals (trust-gated)
        """
        # Check for override first
        override_val = overrides.get("ood_risk_level")
        if override_val is not None:
            return self._safe_float(override_val)

        # Check for direct metadata values
        metadata_ood = meta.get("ood_risk_level") or episode_metadata.get("ood_score")
        if metadata_ood is not None:
            return self._safe_float(metadata_ood)

        # Compute from OOD signals if present
        if ood_signals["ood_tag_count"] == 0:
            return 0.0

        # Use max severity as base, bounded to [0, 1]
        base_risk = min(1.0, max(0.0, ood_signals["max_severity"]))

        # Amplify based on tag count (more OOD tags → higher risk)
        # Cap contribution to avoid explosion
        count_factor = min(1.5, 1.0 + (ood_signals["ood_tag_count"] - 1) * 0.1)
        risk_level = min(1.0, base_risk * count_factor)

        return risk_level

    def _compute_recovery_priority(
        self,
        overrides: Dict[str, Any],
        meta: Dict[str, Any],
        episode_metadata: Dict[str, Any],
        recovery_signals: Dict[str, Any],
    ) -> float:
        """
        Compute recovery_priority from Recovery signals.

        Logic:
        - Use override if present
        - Otherwise use metadata recovery_priority/recovery_score if present
        - Otherwise compute from recovery_signals
        """
        # Check for override first
        override_val = overrides.get("recovery_priority")
        if override_val is not None:
            return self._safe_float(override_val)

        # Check for direct metadata values
        metadata_recovery = meta.get("recovery_priority") or episode_metadata.get("recovery_score")
        if metadata_recovery is not None:
            return self._safe_float(metadata_recovery)

        # Compute from recovery signals if present
        if recovery_signals["recovery_tag_count"] == 0:
            return 0.0

        # Base priority from high-value recovery ratio
        total_count = recovery_signals["recovery_tag_count"]
        high_value_ratio = recovery_signals["high_value_count"] / total_count

        # Priority scales with high-value ratio: [0.3, 1.0]
        priority = 0.3 + (high_value_ratio * 0.7)

        return min(1.0, max(0.0, priority))


def select_skill_mode(
    *,
    tags: Optional[Dict[str, float]],
    trust_matrix: Optional[Dict[str, Any]],
    curriculum_phase: str,
    default: str = ConditionVectorBuilder.DEFAULT_SKILL_MODE,
    advisory: Optional[Dict[str, Any]] = None,
    skill_mode_order: Optional[Sequence[str]] = None,
    econ_slice: Optional[Dict[str, Any]] = None,
    recap_bucket: Optional[str] = None,
    strategy: Optional[str] = None,
    use_condition_vector: bool = True,
) -> str:
    """
    Deterministic skill_mode resolver shared by samplers and ConditionVector.
    """
    default_order = skill_mode_order or [
        "frontier_exploration",
        "safety_critical",
        "efficiency_throughput",
        "recovery_heavy",
        "default",
    ]
    return resolve_skill_mode(
        tags=tags,
        trust_matrix=trust_matrix,
        curriculum_phase=curriculum_phase,
        advisory=advisory,
        default_mode=default,
        mode_order=default_order,
        econ_slice=econ_slice,
        recap_bucket=recap_bucket,
        strategy=strategy,
        use_condition_vector=use_condition_vector,
    )
