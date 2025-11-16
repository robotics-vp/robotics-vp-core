#!/usr/bin/env python3
"""
Scripted Baseline Policy: Open Drawer While Avoiding Vase

A simple scripted policy that:
1. Locates the drawer handle
2. Plans a safe approach path (avoiding vase)
3. Opens drawer slowly while maximizing clearance
4. Retreats after success

This is for debugging and baseline comparison only.
"""

import numpy as np


class DrawerOpenAvoidVasePolicy:
    """
    Scripted policy for safe drawer opening.

    State observation format (13-dim):
        [ee_pos(3), ee_vel(3), drawer_frac(1), vase_pos(3),
         vase_upright(1), min_clearance(1), grasp_state(1)]
    """

    def __init__(
        self,
        handle_pos=np.array([0.0, -0.42, 0.65]),  # Estimated handle position
        vase_pos=np.array([0.3, 0.0, 0.8]),
        safe_offset=0.15,  # Safety margin from vase
        approach_speed=0.8,
        pull_speed=0.6,
        retreat_speed=0.5,
    ):
        self.handle_pos = handle_pos
        self.vase_pos = vase_pos
        self.safe_offset = safe_offset
        self.approach_speed = approach_speed
        self.pull_speed = pull_speed
        self.retreat_speed = retreat_speed

        # State machine
        self.phase = "approach"  # approach, grasp, pull, retreat
        self.step_count = 0

    def reset(self):
        """Reset policy state."""
        self.phase = "approach"
        self.step_count = 0

    def predict(self, obs):
        """
        Predict action given observation.

        Args:
            obs: numpy array of shape (13,) or dict with 'state' key

        Returns:
            action: numpy array of shape (3,) - velocity command
        """
        if isinstance(obs, dict):
            obs = obs.get('state', obs)

        # Parse observation
        ee_pos = obs[0:3]
        ee_vel = obs[3:6]
        drawer_frac = obs[6]
        vase_pos = obs[7:10]
        vase_upright = obs[10]
        min_clearance = obs[11]
        grasp_state = obs[12]

        self.step_count += 1

        # Update vase position from observation
        self.vase_pos = vase_pos

        # State machine
        if self.phase == "approach":
            action = self._approach_handle(ee_pos)
            if self._near_handle(ee_pos):
                self.phase = "grasp"

        elif self.phase == "grasp":
            action = self._align_with_handle(ee_pos)
            if grasp_state > 0.5 or self._near_handle(ee_pos, threshold=0.03):
                self.phase = "pull"

        elif self.phase == "pull":
            action = self._pull_drawer(ee_pos, drawer_frac)
            if drawer_frac >= 0.9:
                self.phase = "retreat"

        elif self.phase == "retreat":
            action = self._retreat_safe(ee_pos)

        else:
            action = np.zeros(3)

        # Safety check: ensure clearance from vase
        action = self._apply_vase_avoidance(ee_pos, action)

        return action.astype(np.float32)

    def _approach_handle(self, ee_pos):
        """Move towards drawer handle while avoiding vase."""
        # Compute safe approach waypoint (offset from vase)
        approach_pos = self.handle_pos.copy()

        # If vase is between EE and handle, go around
        vase_xy = self.vase_pos[:2]
        handle_xy = self.handle_pos[:2]
        ee_xy = ee_pos[:2]

        # Simple: approach from the side opposite to vase
        if np.linalg.norm(ee_xy - vase_xy) < self.safe_offset * 2:
            # Go around
            offset_dir = (handle_xy - vase_xy)
            offset_dir = offset_dir / (np.linalg.norm(offset_dir) + 1e-6)
            approach_pos[:2] = handle_xy - offset_dir * self.safe_offset

        direction = approach_pos - ee_pos
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            return np.zeros(3)

        direction = direction / distance
        speed = min(self.approach_speed, distance * 2)

        return direction * speed

    def _align_with_handle(self, ee_pos):
        """Fine-tune position to grasp handle."""
        direction = self.handle_pos - ee_pos
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            return np.zeros(3)

        direction = direction / distance
        speed = min(0.3, distance)

        return direction * speed

    def _pull_drawer(self, ee_pos, drawer_frac):
        """Pull drawer open (move in -Y direction)."""
        # Pull direction: negative Y (towards robot)
        pull_direction = np.array([0.0, -1.0, 0.0])

        # Modulate speed based on progress
        speed = self.pull_speed * (1.0 - drawer_frac * 0.3)

        return pull_direction * speed

    def _retreat_safe(self, ee_pos):
        """Retreat to safe position after success."""
        safe_pos = np.array([-0.3, 0.0, 0.8])  # Back to start
        direction = safe_pos - ee_pos
        distance = np.linalg.norm(direction)

        if distance < 0.05:
            return np.zeros(3)

        direction = direction / distance
        speed = min(self.retreat_speed, distance)

        return direction * speed

    def _near_handle(self, ee_pos, threshold=0.05):
        """Check if EE is near handle."""
        return np.linalg.norm(ee_pos - self.handle_pos) < threshold

    def _apply_vase_avoidance(self, ee_pos, action):
        """
        Modify action to maintain clearance from vase.

        Uses a repulsive potential field approach.
        """
        ee_to_vase = ee_pos - self.vase_pos
        distance = np.linalg.norm(ee_to_vase)

        if distance < self.safe_offset:
            # Apply repulsive force
            repulsion_strength = (self.safe_offset - distance) / self.safe_offset
            repulsion_direction = ee_to_vase / (distance + 1e-6)

            # Stronger repulsion when closer
            repulsion_force = repulsion_direction * repulsion_strength * 0.5

            # Add to action
            action = action + repulsion_force

            # Limit magnitude
            action_norm = np.linalg.norm(action)
            if action_norm > 1.0:
                action = action / action_norm

        return action

    def get_phase(self):
        """Get current policy phase."""
        return self.phase


def create_scripted_policy():
    """Factory function to create scripted policy."""
    return DrawerOpenAvoidVasePolicy()
