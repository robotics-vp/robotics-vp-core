"""
DataPack Repository with Query API.

Provides storage and retrieval of datapacks with filtering capabilities.
Backs to JSONL files for simplicity and portability.
"""

import os
import json
import numpy as np
from typing import List, Optional, Dict, Any, Iterator
from .datapack_schema import DataPackMeta


class DataPackRepo:
    """
    Repository for storing and querying DataPacks.

    Features:
    - JSONL backing store for each task
    - Append-only writes
    - In-memory filtering queries
    - Support for positive/negative bucket filtering
    - Skill-based queries
    - Condition-based filtering
    """

    def __init__(self, base_dir="data/datapacks"):
        """
        Initialize repository.

        Args:
            base_dir: Base directory for JSONL files
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # In-memory cache (optional, for faster queries)
        self._cache = {}
        self._cache_dirty = {}

    def _get_file_path(self, task_name):
        """Get JSONL file path for task."""
        return os.path.join(self.base_dir, f"{task_name}_datapacks.jsonl")

    def append(self, datapack: DataPackMeta):
        """
        Append a datapack to the repository.

        Args:
            datapack: DataPackMeta to store
        """
        file_path = self._get_file_path(datapack.task_name)

        # Append to JSONL
        with open(file_path, 'a') as f:
            f.write(datapack.to_json() + '\n')

        # Invalidate cache
        if datapack.task_name in self._cache:
            self._cache_dirty[datapack.task_name] = True

    def append_batch(self, datapacks: List[DataPackMeta]):
        """
        Append multiple datapacks efficiently.

        Args:
            datapacks: List of DataPackMeta objects
        """
        # Group by task_name
        by_task = {}
        for dp in datapacks:
            if dp.task_name not in by_task:
                by_task[dp.task_name] = []
            by_task[dp.task_name].append(dp)

        # Write each task's datapacks
        for task_name, task_datapacks in by_task.items():
            file_path = self._get_file_path(task_name)

            with open(file_path, 'a') as f:
                for dp in task_datapacks:
                    f.write(dp.to_json() + '\n')

            # Invalidate cache
            if task_name in self._cache:
                self._cache_dirty[task_name] = True

    def iter_all(self, task_name: str) -> Iterator[DataPackMeta]:
        """
        Iterate over all datapacks for a task.

        Args:
            task_name: Task identifier

        Yields:
            DataPackMeta objects
        """
        file_path = self._get_file_path(task_name)

        if not os.path.exists(file_path):
            return

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield DataPackMeta.from_json(line)

    def load_all(self, task_name: str) -> List[DataPackMeta]:
        """
        Load all datapacks for a task into memory.

        Args:
            task_name: Task identifier

        Returns:
            List of DataPackMeta objects
        """
        # Check cache
        if task_name in self._cache and not self._cache_dirty.get(task_name, False):
            return self._cache[task_name]

        # Load from file
        datapacks = list(self.iter_all(task_name))

        # Update cache
        self._cache[task_name] = datapacks
        self._cache_dirty[task_name] = False

        return datapacks

    def query(
        self,
        task_name: str,
        bucket: Optional[str] = None,
        skill_id: Optional[int] = None,
        engine_type: Optional[str] = None,
        objective_vector: Optional[np.ndarray] = None,
        condition_filters: Optional[Dict[str, Any]] = None,
        min_trust: float = 0.0,
        min_delta_j: Optional[float] = None,
        max_delta_j: Optional[float] = None,
        min_mvd_score: Optional[float] = None,
        source_type: Optional[str] = None,
        limit: int = 100,
        include_counterfactual: bool = False,
        include_sima: bool = False,
        sort_by: Optional[str] = None,
        sort_descending: bool = True
    ) -> List[DataPackMeta]:
        """
        Query datapacks with filters.

        Args:
            task_name: Task identifier
            bucket: "positive" or "negative" (optional)
            skill_id: Filter by skill ID used (optional)
            engine_type: "pybullet", "isaac", "ue5" (optional)
            objective_vector: Match objective vector with tolerance (optional)
            condition_filters: Dict of condition filters (optional)
            min_trust: Minimum trust score (default 0.0)
            min_delta_j: Minimum ΔJ value (optional)
            max_delta_j: Maximum ΔJ value (optional)
            min_mvd_score: Minimum MVD score (optional)
            source_type: "real", "synthetic", "hybrid" (optional)
            limit: Maximum results to return
            include_counterfactual: Include counterfactual plans (for negative)
            include_sima: Include SIMA annotations
            sort_by: Sort field ("delta_j", "trust_score", "mvd_score")
            sort_descending: Sort order

        Returns:
            List of matching DataPackMeta objects
        """
        # Load all datapacks
        all_datapacks = self.load_all(task_name)

        # Filter
        results = []

        for dp in all_datapacks:
            # Bucket filter
            if bucket is not None and dp.bucket != bucket:
                continue

            # Engine type filter
            if engine_type is not None and dp.condition.engine_type != engine_type:
                continue

            # Skill ID filter
            if skill_id is not None and not dp.has_skill(skill_id):
                continue

            # Trust filter
            if dp.attribution.trust_score < min_trust:
                continue

            # Delta J filters
            if min_delta_j is not None and dp.attribution.delta_J < min_delta_j:
                continue
            if max_delta_j is not None and dp.attribution.delta_J > max_delta_j:
                continue

            # MVD score filter
            if min_mvd_score is not None:
                if dp.attribution.mvd_score is None or dp.attribution.mvd_score < min_mvd_score:
                    continue

            # Source type filter
            if source_type is not None:
                if dp.attribution.source_type != source_type:
                    continue

            # Objective vector similarity (if provided)
            if objective_vector is not None:
                dp_vector = np.array(dp.condition.objective_vector)
                if len(dp_vector) == len(objective_vector):
                    # Cosine similarity or distance threshold
                    similarity = np.dot(dp_vector, objective_vector) / (
                        np.linalg.norm(dp_vector) * np.linalg.norm(objective_vector) + 1e-8
                    )
                    if similarity < 0.9:  # 90% similarity threshold
                        continue

            # Condition filters
            if condition_filters is not None:
                if not dp.matches_condition_filters(condition_filters):
                    continue

            # SIMA filter
            if include_sima and dp.sima_annotation is None:
                continue

            # Counterfactual filter
            if include_counterfactual and dp.counterfactual_plan is None:
                if dp.bucket == "negative":
                    continue  # Skip negative without counterfactual

            results.append(dp)

        # Sort
        if sort_by:
            if sort_by == "delta_j":
                results.sort(key=lambda x: x.attribution.delta_J, reverse=sort_descending)
            elif sort_by == "trust_score":
                results.sort(key=lambda x: x.attribution.trust_score, reverse=sort_descending)
            elif sort_by == "mvd_score":
                results.sort(
                    key=lambda x: x.attribution.mvd_score or 0,
                    reverse=sort_descending
                )
            elif sort_by == "created_at":
                results.sort(key=lambda x: x.created_at, reverse=sort_descending)

        # Limit
        return results[:limit]

    def get_positive_for_skill(
        self,
        task_name: str,
        skill_id: int,
        objective_vector: Optional[np.ndarray] = None,
        condition_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[DataPackMeta]:
        """
        Get top positive datapacks for a specific skill.

        Args:
            task_name: Task identifier
            skill_id: Skill ID to filter
            objective_vector: Optional objective vector match
            condition_filters: Optional condition filters
            top_k: Number of results

        Returns:
            List of positive datapacks sorted by ΔJ
        """
        return self.query(
            task_name=task_name,
            bucket="positive",
            skill_id=skill_id,
            objective_vector=objective_vector,
            condition_filters=condition_filters,
            min_trust=0.9,
            limit=top_k,
            sort_by="delta_j",
            sort_descending=True
        )

    def get_negative_for_skill(
        self,
        task_name: str,
        skill_id: int,
        objective_vector: Optional[np.ndarray] = None,
        condition_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        require_counterfactual: bool = True
    ) -> List[DataPackMeta]:
        """
        Get top negative datapacks for a specific skill.

        Args:
            task_name: Task identifier
            skill_id: Skill ID to filter
            objective_vector: Optional objective vector match
            condition_filters: Optional condition filters
            top_k: Number of results
            require_counterfactual: Only include packs with counterfactual

        Returns:
            List of negative datapacks sorted by ΔJ (most negative first)
        """
        return self.query(
            task_name=task_name,
            bucket="negative",
            skill_id=skill_id,
            objective_vector=objective_vector,
            condition_filters=condition_filters,
            min_trust=0.9,
            limit=top_k,
            include_counterfactual=require_counterfactual,
            sort_by="delta_j",
            sort_descending=False  # Most negative first
        )

    def get_statistics(self, task_name: str) -> Dict[str, Any]:
        """
        Get statistics for a task's datapacks.

        Args:
            task_name: Task identifier

        Returns:
            Dict with statistics
        """
        datapacks = self.load_all(task_name)

        if not datapacks:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
            }

        positive = [dp for dp in datapacks if dp.bucket == "positive"]
        negative = [dp for dp in datapacks if dp.bucket == "negative"]

        # Delta J statistics
        delta_js = [dp.attribution.delta_J for dp in datapacks]
        trust_scores = [dp.attribution.trust_score for dp in datapacks]

        # Skills used
        all_skills = set()
        for dp in datapacks:
            all_skills.update(dp.get_skill_ids())

        # Engine types
        engine_types = {}
        for dp in datapacks:
            et = dp.condition.engine_type
            engine_types[et] = engine_types.get(et, 0) + 1

        # Source types
        source_types = {}
        for dp in datapacks:
            st = dp.attribution.source_type
            source_types[st] = source_types.get(st, 0) + 1

        return {
            'total': len(datapacks),
            'positive': len(positive),
            'negative': len(negative),
            'positive_ratio': len(positive) / len(datapacks),
            'delta_j_mean': np.mean(delta_js),
            'delta_j_std': np.std(delta_js),
            'delta_j_min': np.min(delta_js),
            'delta_j_max': np.max(delta_js),
            'trust_mean': np.mean(trust_scores),
            'trust_std': np.std(trust_scores),
            'unique_skills': list(all_skills),
            'engine_types': engine_types,
            'source_types': source_types,
            'with_sima': sum(1 for dp in datapacks if dp.sima_annotation is not None),
            'with_counterfactual': sum(1 for dp in negative if dp.counterfactual_plan is not None),
        }

    def clear(self, task_name: str):
        """
        Clear all datapacks for a task.

        Args:
            task_name: Task identifier
        """
        file_path = self._get_file_path(task_name)

        if os.path.exists(file_path):
            os.remove(file_path)

        # Clear cache
        if task_name in self._cache:
            del self._cache[task_name]
        if task_name in self._cache_dirty:
            del self._cache_dirty[task_name]

    def export_to_json(self, task_name: str, output_path: str):
        """
        Export all datapacks for a task to a single JSON file.

        Args:
            task_name: Task identifier
            output_path: Output JSON file path
        """
        datapacks = self.load_all(task_name)

        with open(output_path, 'w') as f:
            json.dump([dp.to_dict() for dp in datapacks], f, indent=2)

    def import_from_json(self, input_path: str):
        """
        Import datapacks from a JSON file.

        Args:
            input_path: Input JSON file path
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        datapacks = [DataPackMeta.from_dict(d) for d in data]
        self.append_batch(datapacks)

    def list_tasks(self) -> List[str]:
        """
        List all tasks with datapacks.

        Returns:
            List of task names
        """
        tasks = []
        for filename in os.listdir(self.base_dir):
            if filename.endswith('_datapacks.jsonl'):
                task_name = filename.replace('_datapacks.jsonl', '')
                tasks.append(task_name)
        return tasks
