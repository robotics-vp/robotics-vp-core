"""
HydraNet Loss Components - Scaffolding for Multi-Head Actor/Critic Training

This module provides loss computation for HydraNet architecture with:
- Per-head actor losses
- Per-metric critic losses (MPL, energy, damage, novelty)
- Head-isolation rules
- Skill-mode routing for which heads contribute to each loss
- Gradient clipping & detachment invariants
- Bounded contributions (±20%) exactly like Phase H rules

IMPORTANT: This is SCAFFOLDING ONLY. Neural network implementations will be
provided by Codex. This module defines the loss computation interface, routing
logic, and invariants.

All functionality is:
- Flag-gated
- Advisory-only (does not modify reward math)
- Deterministic
- JSON-safe
- Backward-compatible
- Bounded (±20%)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# Constants and Configuration
# =============================================================================

# Version stamp to track scaffolding updates
HYDRA_LOSS_SCAFFOLD_VERSION = "stage5_scaffolding_v1"

# Head isolation modes
HEAD_ISOLATION_MODES = {
    'full_isolation': 'Each head trained independently',
    'partial_sharing': 'Shared backbone, isolated heads',
    'full_sharing': 'All heads share gradients'
}

# Skill modes for head routing
SKILL_MODES = {
    'manipulation': ['grasp', 'place', 'precision'],
    'navigation': ['locomotion', 'pathfinding', 'obstacle_avoidance'],
    'coordination': ['bimanual', 'tool_use', 'sequencing'],
    'economic': ['mpl_optimization', 'energy_efficiency', 'damage_minimization']
}

# Bounded contribution limits (Phase H compliance)
CONTRIBUTION_BOUNDS = {
    'min': -0.20,  # -20%
    'max': 0.20    # +20%
}

# Gradient clipping thresholds
GRADIENT_CLIP_NORM = 1.0
GRADIENT_CLIP_VALUE = 0.5


# =============================================================================
# Per-Head Actor Losses
# =============================================================================

class PerHeadActorLoss:
    """
    Computes actor losses for individual HydraNet heads.

    Supports:
    - Policy gradient losses (PPO, SAC, etc.)
    - Head-specific skill routing
    - Gradient isolation rules
    - Bounded contribution enforcement
    """

    def __init__(
        self,
        head_name: str,
        skill_modes: List[str],
        isolation_mode: str = 'partial_sharing',
        enable_clipping: bool = True,
        enable_bounds: bool = True
    ):
        """
        Initialize per-head actor loss.

        Args:
            head_name: Name of this actor head
            skill_modes: List of skill modes this head handles
            isolation_mode: Head isolation strategy
            enable_clipping: Whether to clip gradients
            enable_bounds: Whether to enforce bounded contributions
        """
        self.head_name = head_name
        self.skill_modes = skill_modes
        self.isolation_mode = isolation_mode
        self.enable_clipping = enable_clipping
        self.enable_bounds = enable_bounds

        # Validate isolation mode
        if isolation_mode not in HEAD_ISOLATION_MODES:
            raise ValueError(f"Invalid isolation mode: {isolation_mode}")

    def compute_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        skill_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute actor loss for this head.

        Args:
            predictions: Model predictions (to be provided by Codex neural impl)
            targets: Target values
            skill_context: Current skill mode context

        Returns:
            Dictionary containing:
            - 'loss': Scalar loss value
            - 'components': Breakdown of loss components
            - 'metadata': Diagnostic information
            - 'gradients': Gradient statistics (if applicable)
        """
        # Check if this head should contribute based on skill context
        should_contribute = self._should_contribute(skill_context)

        if not should_contribute:
            return {
                'loss': 0.0,
                'components': {},
                'metadata': {
                    'head_name': self.head_name,
                    'skill_context': skill_context,
                    'contributed': False,
                    'reason': 'skill_context_mismatch'
                },
                'gradients': {}
            }

        # PLACEHOLDER: Actual loss computation will be implemented by Codex
        # This is the interface that Codex's neural implementation will use
        loss_value = self._compute_policy_gradient_loss(predictions, targets)

        # Apply bounds if enabled
        if self.enable_bounds:
            loss_value = self._apply_bounds(loss_value)

        # Compute gradient statistics (for monitoring)
        grad_stats = self._compute_gradient_stats(predictions)

        return {
            'loss': float(loss_value),
            'components': {
                'policy_gradient': float(loss_value),
                'entropy_bonus': 0.0,  # PLACEHOLDER
                'regularization': 0.0   # PLACEHOLDER
            },
            'metadata': {
                'head_name': self.head_name,
                'skill_context': skill_context,
                'contributed': True,
                'isolation_mode': self.isolation_mode
            },
            'gradients': grad_stats
        }

    def _should_contribute(self, skill_context: Optional[str]) -> bool:
        """Determine if this head should contribute to loss for given skill context."""
        if skill_context is None:
            return True  # Contribute to all if no context specified

        # Check if skill context matches any of this head's skill modes
        for mode in self.skill_modes:
            if skill_context.startswith(mode) or mode in skill_context:
                return True

        return False

    def _compute_policy_gradient_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> float:
        """
        PLACEHOLDER: Compute policy gradient loss.

        This will be implemented by Codex with actual neural network logic.
        For now, returns a deterministic placeholder value.
        """
        # TODO: Replace with PPO/SAC actor loss once neural heads are wired.
        # Deterministic placeholder
        return 0.0

    def _apply_bounds(self, loss_value: float) -> float:
        """Apply bounded contribution limits (±20%)."""
        return np.clip(
            loss_value,
            CONTRIBUTION_BOUNDS['min'],
            CONTRIBUTION_BOUNDS['max']
        )

    def _compute_gradient_stats(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute gradient statistics for monitoring.

        Returns:
            Dictionary with gradient norm, max value, etc.
        """
        # PLACEHOLDER: Will be implemented with actual gradients by Codex
        return {
            'grad_norm': 0.0,
            'grad_max': 0.0,
            'grad_min': 0.0,
            'grad_mean': 0.0
        }


# =============================================================================
# Per-Metric Critic Losses
# =============================================================================

class PerMetricCriticLoss:
    """
    Computes critic losses for individual metrics.

    Metrics:
    - MPL (marginal product of labor)
    - Energy efficiency
    - Damage/error rate
    - Novelty/exploration

    Each metric gets its own value head with isolated training.
    """

    def __init__(
        self,
        metric_name: str,
        metric_type: str,
        enable_clipping: bool = True,
        enable_bounds: bool = True
    ):
        """
        Initialize per-metric critic loss.

        Args:
            metric_name: Name of this metric
            metric_type: Type ('mpl', 'energy', 'damage', 'novelty')
            enable_clipping: Whether to clip gradients
            enable_bounds: Whether to enforce bounded contributions
        """
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.enable_clipping = enable_clipping
        self.enable_bounds = enable_bounds

        # Validate metric type
        valid_types = ['mpl', 'energy', 'damage', 'novelty']
        if metric_type not in valid_types:
            raise ValueError(f"Invalid metric type: {metric_type}")

    def compute_loss(
        self,
        value_predictions: Dict[str, Any],
        value_targets: Dict[str, Any],
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute critic loss for this metric.

        Args:
            value_predictions: Value function predictions
            value_targets: Target values
            returns: Optional return estimates

        Returns:
            Dictionary containing:
            - 'loss': Scalar loss value
            - 'components': Breakdown of loss components
            - 'metadata': Diagnostic information
            - 'predictions': Value predictions for analysis
        """
        # PLACEHOLDER: Actual loss computation will be implemented by Codex
        loss_value = self._compute_td_loss(value_predictions, value_targets, returns)

        # Apply bounds if enabled
        if self.enable_bounds:
            loss_value = self._apply_bounds(loss_value)

        return {
            'loss': float(loss_value),
            'components': {
                'td_error': float(loss_value),
                'regularization': 0.0  # PLACEHOLDER
            },
            'metadata': {
                'metric_name': self.metric_name,
                'metric_type': self.metric_type,
                'n_samples': len(value_targets) if isinstance(value_targets, (list, np.ndarray)) else 1
            },
            'predictions': {
                'mean': 0.0,  # PLACEHOLDER
                'std': 0.0,   # PLACEHOLDER
                'min': 0.0,   # PLACEHOLDER
                'max': 0.0    # PLACEHOLDER
            }
        }

    def _compute_td_loss(
        self,
        value_predictions: Dict[str, Any],
        value_targets: Dict[str, Any],
        returns: Optional[np.ndarray]
    ) -> float:
        """
        PLACEHOLDER: Compute temporal difference loss.

        This will be implemented by Codex with actual neural network logic.
        For now, returns a deterministic placeholder value.
        """
        # TODO: Replace with TD(\lambda) or GAE style critic loss.
        # Deterministic placeholder
        return 0.0

    def _apply_bounds(self, loss_value: float) -> float:
        """Apply bounded contribution limits (±20%)."""
        return np.clip(
            loss_value,
            CONTRIBUTION_BOUNDS['min'],
            CONTRIBUTION_BOUNDS['max']
        )


# =============================================================================
# Head Isolation Manager
# =============================================================================

class HeadIsolationManager:
    """
    Manages gradient isolation between HydraNet heads.

    Enforces:
    - Gradient detachment rules
    - Head-specific backpropagation
    - Shared backbone updates
    - Isolation mode compliance
    """

    def __init__(self, isolation_mode: str = 'partial_sharing'):
        """
        Initialize head isolation manager.

        Args:
            isolation_mode: Isolation strategy ('full_isolation', 'partial_sharing', 'full_sharing')
        """
        self.isolation_mode = isolation_mode

        if isolation_mode not in HEAD_ISOLATION_MODES:
            raise ValueError(f"Invalid isolation mode: {isolation_mode}")

    def apply_isolation(
        self,
        head_losses: Dict[str, float],
        shared_backbone: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Apply gradient isolation rules to head losses.

        Args:
            head_losses: Dictionary of {head_name: loss_value}
            shared_backbone: Optional shared backbone (for partial_sharing mode)

        Returns:
            Dictionary with isolation metadata and gradient routing
        """
        if self.isolation_mode == 'full_isolation':
            return self._apply_full_isolation(head_losses)
        elif self.isolation_mode == 'partial_sharing':
            return self._apply_partial_sharing(head_losses, shared_backbone)
        else:  # full_sharing
            return self._apply_full_sharing(head_losses)

    def _apply_full_isolation(self, head_losses: Dict[str, float]) -> Dict[str, Any]:
        """Apply full isolation (no gradient sharing between heads)."""
        return {
            'isolation_mode': 'full_isolation',
            'head_losses': head_losses,
            'shared_gradients': {},
            'metadata': {
                'n_heads': len(head_losses),
                'gradient_sharing': False
            }
        }

    def _apply_partial_sharing(
        self,
        head_losses: Dict[str, float],
        shared_backbone: Optional[Any]
    ) -> Dict[str, Any]:
        """Apply partial sharing (shared backbone, isolated heads)."""
        return {
            'isolation_mode': 'partial_sharing',
            'head_losses': head_losses,
            'shared_gradients': {
                'backbone': sum(head_losses.values()) / len(head_losses) if head_losses else 0.0
            },
            'metadata': {
                'n_heads': len(head_losses),
                'gradient_sharing': True,
                'sharing_scope': 'backbone_only'
            }
        }

    def _apply_full_sharing(self, head_losses: Dict[str, float]) -> Dict[str, Any]:
        """Apply full sharing (all heads share gradients)."""
        return {
            'isolation_mode': 'full_sharing',
            'head_losses': head_losses,
            'shared_gradients': {
                'all': sum(head_losses.values()) / len(head_losses) if head_losses else 0.0
            },
            'metadata': {
                'n_heads': len(head_losses),
                'gradient_sharing': True,
                'sharing_scope': 'full_network'
            }
        }


# =============================================================================
# Skill-Mode Router
# =============================================================================

class SkillModeRouter:
    """
    Routes skill contexts to appropriate HydraNet heads.

    Determines which heads should contribute to loss computation
    based on current skill mode and task context.
    """

    def __init__(self, skill_modes: Optional[Dict[str, List[str]]] = None):
        """
        Initialize skill mode router.

        Args:
            skill_modes: Optional custom skill mode definitions
        """
        self.skill_modes = skill_modes or SKILL_MODES

    def route(
        self,
        skill_context: str,
        available_heads: List[str]
    ) -> Dict[str, bool]:
        """
        Route skill context to appropriate heads.

        Args:
            skill_context: Current skill mode/context
            available_heads: List of available head names

        Returns:
            Dictionary mapping head names to whether they should contribute
        """
        routing = {}

        for head_name in available_heads:
            # Determine if this head should be active for this skill context
            should_activate = self._should_activate_head(head_name, skill_context)
            routing[head_name] = should_activate

        return routing

    def _should_activate_head(self, head_name: str, skill_context: str) -> bool:
        """Determine if a head should be activated for given skill context."""
        # Check if head name matches any skill mode
        for mode_category, mode_list in self.skill_modes.items():
            for mode in mode_list:
                if mode in head_name.lower() and mode in skill_context.lower():
                    return True
                if mode_category in head_name.lower() and mode in skill_context.lower():
                    return True

        # Default: activate if no specific match (allow head to self-select)
        return True


# =============================================================================
# Gradient Utilities
# =============================================================================

def clip_gradients(
    gradients: Dict[str, np.ndarray],
    clip_norm: float = GRADIENT_CLIP_NORM,
    clip_value: float = GRADIENT_CLIP_VALUE
) -> Dict[str, np.ndarray]:
    """
    Clip gradients by norm and value.

    Args:
        gradients: Dictionary of gradient arrays
        clip_norm: Maximum gradient norm
        clip_value: Maximum absolute gradient value

    Returns:
        Clipped gradients
    """
    clipped = {}

    for name, grad in gradients.items():
        if grad is None:
            continue
        # Clip by norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm > clip_norm:
            grad = grad * (clip_norm / grad_norm)

        # Clip by value
        grad = np.clip(grad, -clip_value, clip_value)

        clipped[name] = grad

    return clipped


def compute_gradient_stats(gradients: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute gradient statistics for monitoring.

    Args:
        gradients: Dictionary of gradient arrays

    Returns:
        Statistics dictionary
    """
    grad_list = [g.flatten() for g in gradients.values() if g is not None]
    if not grad_list:
        return {
            'grad_norm': 0.0,
            'grad_max': 0.0,
            'grad_min': 0.0,
            'grad_mean': 0.0,
            'grad_std': 0.0,
            'n_parameters': 0
        }

    all_grads = np.concatenate(grad_list)

    return {
        'grad_norm': float(np.linalg.norm(all_grads)),
        'grad_max': float(np.max(np.abs(all_grads))),
        'grad_min': float(np.min(np.abs(all_grads))),
        'grad_mean': float(np.mean(np.abs(all_grads))),
        'grad_std': float(np.std(all_grads)),
        'n_parameters': len(all_grads)
    }


# =============================================================================
# Bounded Contribution Enforcer
# =============================================================================

def enforce_bounded_contributions(
    contributions: Dict[str, float],
    bounds: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Enforce bounded contribution limits (±20%).

    Args:
        contributions: Dictionary of contribution values
        bounds: Optional custom bounds (default: ±20%)

    Returns:
        Bounded contributions
    """
    if bounds is None:
        bounds = CONTRIBUTION_BOUNDS

    bounded = {}
    for name, value in contributions.items():
        bounded[name] = np.clip(value, bounds['min'], bounds['max'])

    return bounded


# =============================================================================
# JSON-Safe Serialization
# =============================================================================

def make_json_safe(obj: Any) -> Any:
    """
    Convert object to JSON-safe representation.

    Args:
        obj: Object to convert

    Returns:
        JSON-safe version
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        return float(obj) if obj.size == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    else:
        return obj


# =============================================================================
# Hydra Loss Orchestration Helpers
# =============================================================================

@dataclass(frozen=True)
class HydraLossResult:
    """Canonical, JSON-safe loss summary used by smoke tests and trainers."""
    actor: Dict[str, Any]
    critic: Dict[str, Any]
    isolation: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "actor": make_json_safe(self.actor),
            "critic": make_json_safe(self.critic),
            "isolation": make_json_safe(self.isolation),
            "metadata": make_json_safe(self.metadata),
        }
        return payload


def compute_hydra_losses(
    actor_losses: Dict[str, PerHeadActorLoss],
    critic_losses: Dict[str, PerMetricCriticLoss],
    head_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
    head_targets: Optional[Dict[str, Dict[str, Any]]] = None,
    value_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
    value_targets: Optional[Dict[str, Dict[str, Any]]] = None,
    returns: Optional[np.ndarray] = None,
    skill_context: Optional[str] = None,
    isolation_mode: str = 'partial_sharing',
    custom_skill_router: Optional[SkillModeRouter] = None
) -> HydraLossResult:
    """
    Orchestrate per-head actor and per-metric critic loss computation.

    This is intentionally lightweight and matches the HydraActor/HydraCritic
    routing semantics: skill_context should match ConditionVector.skill_mode.
    """
    router = custom_skill_router or SkillModeRouter()
    available_heads = sorted(actor_losses.keys())
    routing = router.route(skill_context or "", available_heads)

    actor_results: Dict[str, Any] = {}
    head_scalar_losses: Dict[str, float] = {}
    for head_name in available_heads:
        loss_impl = actor_losses[head_name]
        preds = (head_predictions or {}).get(head_name, {})
        targs = (head_targets or {}).get(head_name, {})
        result = loss_impl.compute_loss(preds, targs, skill_context=skill_context)
        # Respect routing decision while keeping deterministic ordering
        result["metadata"]["routed_active"] = bool(routing.get(head_name, True))
        actor_results[head_name] = result
        if routing.get(head_name, True):
            head_scalar_losses[head_name] = float(result.get("loss", 0.0))

    isolation_manager = HeadIsolationManager(isolation_mode=isolation_mode)
    isolation_info = isolation_manager.apply_isolation(head_scalar_losses, shared_backbone=None)

    critic_results: Dict[str, Any] = {}
    for metric_name in sorted(critic_losses.keys()):
        loss_impl = critic_losses[metric_name]
        preds = (value_predictions or {}).get(metric_name, {})
        targs = (value_targets or {}).get(metric_name, {})
        critic_results[metric_name] = loss_impl.compute_loss(preds, targs, returns=returns)

    metadata = {
        "skill_context": skill_context,
        "available_heads": available_heads,
        "hydra_loss_version": HYDRA_LOSS_SCAFFOLD_VERSION,
    }

    return HydraLossResult(
        actor=actor_results,
        critic=critic_results,
        isolation=isolation_info,
        metadata=metadata,
    )
