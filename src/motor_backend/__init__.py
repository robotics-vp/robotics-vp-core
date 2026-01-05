"""Motor backend interfaces and implementations."""

from src.motor_backend.base import (  # noqa: F401
    MotorBackend,
    MotorTrainingResult,
    MotorEvalResult,
)
from src.motor_backend.factory import make_motor_backend  # noqa: F401

# WorkcellEnvBackend is imported lazily via factory to avoid circular imports
# Use: from src.motor_backend.workcell_env_backend import WorkcellEnvBackend
