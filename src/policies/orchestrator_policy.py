"""
Heuristic OrchestratorPolicy that delegates to SemanticOrchestratorV2.
"""
from typing import Any, Dict, Optional

from src.orchestrator.semantic_orchestrator_v2 import OrchestratorAdvisory, SemanticOrchestratorV2
from src.policies.interfaces import OrchestratorPolicy


class HeuristicOrchestratorPolicy(OrchestratorPolicy):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._impl = SemanticOrchestratorV2(config or {})

    def advise(self, snapshot: Any) -> OrchestratorAdvisory:
        return self._impl.propose(snapshot)
