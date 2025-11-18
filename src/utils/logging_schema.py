"""
Canonical logging field names used across samplers/curricula/ontology logs.
"""
LOG_FIELDS = {
    "task_id": "task_id",
    "episode_id": "episode_id",
    "sampler_strategy": "sampler_strategy",
    "curriculum_phase": "curriculum_phase",
    "objective_preset": "objective_preset",
    "objective_vector": "objective_vector",
    "pack_id": "pack_id",
    "backend": "backend",
}

POLICY_LOG_FIELDS = {
    "policy_name": "policy",
    "input_features": "features",
    "output": "target",
    "meta": "meta",
    "timestamp": "timestamp",
    "task_id": "task_id",
    "episode_id": "episode_id",
    "datapack_id": "datapack_id",
}
