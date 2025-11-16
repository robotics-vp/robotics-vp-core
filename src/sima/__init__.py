"""
SIMA-2 Co-Agent for Drawer+Vase Task.

Provides:
- SIMACoAgent: Co-agent that generates demonstrations with narrations
- TrajectoryGenerator: Generates full task trajectories
- Narrator: Produces step-level language annotations
"""

from .co_agent import SIMACoAgent
from .trajectory_generator import TrajectoryGenerator
from .narrator import Narrator

__all__ = [
    'SIMACoAgent',
    'TrajectoryGenerator',
    'Narrator',
]
