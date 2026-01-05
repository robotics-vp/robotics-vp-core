from __future__ import annotations

from typing import Any, Optional

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.base import MotorBackend
from src.motor_backend.datapacks import DatapackProvider
from src.ontology.store import OntologyStore


def make_motor_backend(
    name: str,
    econ_meter: EconomicMeter,
    store: OntologyStore,
    backend_config: Optional[Any] = None,
) -> MotorBackend | None:
    if name == "holosoma":
        from src.motor_backend.holosoma_backend import HolosomaBackend

        return HolosomaBackend(econ_meter=econ_meter, datapack_provider=DatapackProvider(store))
    if name == "synthetic":
        from src.motor_backend.synthetic_backend import SyntheticBackend

        return SyntheticBackend(econ_meter=econ_meter)
    if name == "lsd_vector_scene":
        from src.motor_backend.lsd_vector_scene_backend import LSDVectorSceneBackend
        from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

        lsd_config = None
        if backend_config is not None:
            if isinstance(backend_config, LSDVectorSceneConfig):
                lsd_config = backend_config
            elif isinstance(backend_config, dict):
                from src.config.lsd_vector_scene_config import load_lsd_vector_scene_config
                lsd_config = load_lsd_vector_scene_config(backend_config)

        return LSDVectorSceneBackend(econ_meter=econ_meter, default_config=lsd_config)
    if name == "workcell_env":
        from src.motor_backend.workcell_env_backend import WorkcellEnvBackend
        from src.config.workcell_env_config import WorkcellEnvConfig, load_workcell_env_config

        workcell_config = None
        if backend_config is not None:
            if isinstance(backend_config, WorkcellEnvConfig):
                workcell_config = backend_config
            elif isinstance(backend_config, dict):
                workcell_config = load_workcell_env_config(backend_config)

        return WorkcellEnvBackend(
            econ_meter=econ_meter,
            datapack_provider=DatapackProvider(store),
            default_config=workcell_config,
        )
    if name == "dummy":
        return None
    raise ValueError(f"Unknown motor backend: {name}")
