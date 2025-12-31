from __future__ import annotations

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.base import MotorBackend
from src.motor_backend.datapacks import DatapackProvider
from src.ontology.store import OntologyStore


def make_motor_backend(name: str, econ_meter: EconomicMeter, store: OntologyStore) -> MotorBackend | None:
    if name == "holosoma":
        from src.motor_backend.holosoma_backend import HolosomaBackend

        return HolosomaBackend(econ_meter=econ_meter, datapack_provider=DatapackProvider(store))
    if name == "synthetic":
        from src.motor_backend.synthetic_backend import SyntheticBackend

        return SyntheticBackend(econ_meter=econ_meter)
    if name == "dummy":
        return None
    raise ValueError(f"Unknown motor backend: {name}")
