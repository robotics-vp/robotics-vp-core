"""
Skill Termination Detection for HRL.

Determines when a skill has completed (success or failure).
"""

import numpy as np
from .skills import SkillID


class SkillTerminationDetector:
    """
    Detects when a skill has completed.

    Each skill has specific termination conditions based on:
    - Task completion (success)
    - Timeout (failure)
    - Safety violation (failure)
    """

    def __init__(self):
        self.handle_pos = np.array([0.0, -0.42, 0.65])
        self.safe_pos = np.array([-0.3, 0.0, 0.8])

    def is_done(self, skill_id, obs, info, step_count=0, max_steps=100):
        """
        Check if skill is done.

        Args:
            skill_id: Current skill ID
            obs: Current observation (13,)
            info: Info dict from environment
            step_count: Steps executed in this skill
            max_steps: Maximum allowed steps

        Returns:
            done: bool - whether skill is done
            success: bool - whether skill completed successfully
            reason: str - reason for termination
        """
        # Check timeout
        if step_count >= max_steps:
            return True, False, "timeout"

        # Check safety (vase broken)
        if not info.get('vase_intact', True):
            return True, False, "vase_broken"

        # Skill-specific conditions
        if skill_id == SkillID.LOCATE_DRAWER:
            return self._check_locate_drawer(obs, info, step_count)

        elif skill_id == SkillID.LOCATE_VASE:
            return self._check_locate_vase(obs, info, step_count)

        elif skill_id == SkillID.PLAN_SAFE_APPROACH:
            return self._check_plan_safe_approach(obs, info, step_count)

        elif skill_id == SkillID.GRASP_HANDLE:
            return self._check_grasp_handle(obs, info, step_count)

        elif skill_id == SkillID.OPEN_WITH_CLEARANCE:
            return self._check_open_with_clearance(obs, info, step_count)

        elif skill_id == SkillID.RETRACT_SAFE:
            return self._check_retract_safe(obs, info, step_count)

        return False, False, "unknown_skill"

    def _check_locate_drawer(self, obs, info, step_count):
        """Check if drawer has been located."""
        # For now, assume drawer is always visible (no occlusion model)
        # In full implementation, check info['drawer_handle_visible']
        if step_count >= 5:  # Minimal observation time
            return True, True, "drawer_located"
        return False, False, ""

    def _check_locate_vase(self, obs, info, step_count):
        """Check if vase has been located."""
        vase_pos = obs[7:10]
        vase_detected = np.linalg.norm(vase_pos) > 0.1

        if vase_detected and step_count >= 3:
            return True, True, "vase_located"

        if step_count >= 10:
            # Timeout with partial success (vase may not exist)
            return True, True, "observation_complete"

        return False, False, ""

    def _check_plan_safe_approach(self, obs, info, step_count):
        """Check if safe approach has been planned."""
        # Planning is instantaneous (single step)
        # In full implementation, this would compute actual waypoints
        if step_count >= 1:
            return True, True, "plan_computed"
        return False, False, ""

    def _check_grasp_handle(self, obs, info, step_count):
        """Check if handle has been grasped."""
        ee_pos = obs[0:3]
        grasp_state = obs[12]

        distance = np.linalg.norm(ee_pos - self.handle_pos)

        # Success: close to handle or grasp confirmed
        if distance < 0.05 or grasp_state > 0.5:
            return True, True, "handle_grasped"

        # Check for safety violations
        min_clearance = obs[11]
        if min_clearance < 0.03:  # Too close to vase
            return True, False, "clearance_violation"

        return False, False, ""

    def _check_open_with_clearance(self, obs, info, step_count):
        """Check if drawer has been opened safely."""
        drawer_frac = obs[6]
        min_clearance = obs[11]

        # Success: drawer >= 90% open and vase intact
        if drawer_frac >= 0.9:
            return True, True, "drawer_opened"

        # Failure: clearance violation
        if min_clearance < 0.02:
            return True, False, "clearance_violation"

        # Failure: too many high-risk contacts
        n_high_risk = info.get('n_high_risk_contacts', 0)
        if n_high_risk >= 5:
            return True, False, "sla_violation"

        return False, False, ""

    def _check_retract_safe(self, obs, info, step_count):
        """Check if EE has retracted to safe position."""
        ee_pos = obs[0:3]
        distance = np.linalg.norm(ee_pos - self.safe_pos)

        if distance < 0.1:
            return True, True, "retracted_safely"

        return False, False, ""

    def get_skill_reward(self, skill_id, obs, obs_next, info, done, success):
        """
        Compute dense reward for skill execution.

        Args:
            skill_id: Current skill
            obs: Previous observation
            obs_next: Current observation
            info: Info dict
            done: Whether skill is done
            success: Whether skill succeeded

        Returns:
            reward: float
        """
        reward = 0.0

        # Base step penalty to encourage efficiency
        reward -= 0.01

        # Safety penalty
        min_clearance = obs_next[11]
        if min_clearance < 0.1:
            reward -= 0.1 * (0.1 - min_clearance)

        # Energy penalty
        ee_vel = obs_next[3:6]
        energy = np.linalg.norm(ee_vel)
        reward -= 0.01 * energy

        # Skill-specific rewards
        if skill_id == SkillID.GRASP_HANDLE:
            # Reward for approaching handle
            ee_pos = obs_next[0:3]
            dist_to_handle = np.linalg.norm(ee_pos - self.handle_pos)
            prev_dist = np.linalg.norm(obs[0:3] - self.handle_pos)
            reward += 0.5 * (prev_dist - dist_to_handle)  # Positive when getting closer

        elif skill_id == SkillID.OPEN_WITH_CLEARANCE:
            # Reward for opening drawer
            drawer_frac = obs_next[6]
            prev_drawer_frac = obs[6]
            reward += 2.0 * (drawer_frac - prev_drawer_frac)

            # Bonus for maintaining clearance
            if min_clearance > 0.15:
                reward += 0.05

        # Terminal rewards
        if done:
            if success:
                reward += 10.0  # Success bonus
            else:
                reward -= 5.0   # Failure penalty

        return reward

    def get_progress_estimate(self, skill_id, obs, info):
        """
        Estimate progress within current skill (0 to 1).

        Useful for high-level controller to decide skill switching.
        """
        if skill_id == SkillID.LOCATE_DRAWER:
            return 1.0  # Instantaneous

        elif skill_id == SkillID.LOCATE_VASE:
            vase_pos = obs[7:10]
            return 1.0 if np.linalg.norm(vase_pos) > 0.1 else 0.0

        elif skill_id == SkillID.PLAN_SAFE_APPROACH:
            return 1.0  # Instantaneous

        elif skill_id == SkillID.GRASP_HANDLE:
            ee_pos = obs[0:3]
            distance = np.linalg.norm(ee_pos - self.handle_pos)
            # Progress is inverse of distance (0 at far, 1 at handle)
            return max(0, min(1, 1 - distance / 0.5))

        elif skill_id == SkillID.OPEN_WITH_CLEARANCE:
            drawer_frac = obs[6]
            return drawer_frac / 0.9  # 0 to 1 as drawer opens

        elif skill_id == SkillID.RETRACT_SAFE:
            ee_pos = obs[0:3]
            distance = np.linalg.norm(ee_pos - self.safe_pos)
            return max(0, min(1, 1 - distance / 0.5))

        return 0.0


class SkillRewardShaper:
    """
    Provides shaped rewards for skill learning.

    Combines:
    - Progress rewards (dense)
    - Safety rewards (constraints)
    - Efficiency rewards (energy, jerk)
    - Terminal bonuses
    """

    def __init__(self):
        self.termination_detector = SkillTerminationDetector()

    def compute_reward(
        self,
        skill_id,
        obs,
        obs_next,
        action,
        info,
        skill_params=None,
        step_in_skill=0
    ):
        """
        Compute comprehensive reward for skill step.

        Returns:
            reward: float
            reward_components: dict
        """
        components = {}

        # 1. Step penalty (encourage efficiency)
        step_penalty = -0.01
        components['step_penalty'] = step_penalty

        # 2. Safety reward (clearance from vase)
        min_clearance = obs_next[11]
        target_clearance = 0.15 if skill_params is None else skill_params.target_clearance

        if min_clearance < target_clearance:
            safety_penalty = -0.5 * (target_clearance - min_clearance) / target_clearance
        else:
            safety_bonus = 0.05
            safety_penalty = safety_bonus
        components['safety'] = safety_penalty

        # 3. Energy penalty
        ee_vel = obs_next[3:6]
        energy = np.linalg.norm(ee_vel) ** 2
        energy_penalty = -0.01 * energy
        components['energy'] = energy_penalty

        # 4. Jerk penalty (smooth motion)
        if step_in_skill > 0:
            prev_vel = obs[3:6]
            jerk = np.linalg.norm(ee_vel - prev_vel)
            jerk_penalty = -0.01 * jerk
        else:
            jerk_penalty = 0.0
        components['jerk'] = jerk_penalty

        # 5. Progress reward (skill-specific)
        progress_reward = self._compute_progress_reward(
            skill_id, obs, obs_next, action
        )
        components['progress'] = progress_reward

        # Total reward
        total_reward = sum(components.values())
        components['total'] = total_reward

        return total_reward, components

    def _compute_progress_reward(self, skill_id, obs, obs_next, action):
        """Compute skill-specific progress reward."""
        if skill_id == SkillID.GRASP_HANDLE:
            # Reward for getting closer to handle
            ee_pos = obs_next[0:3]
            prev_ee_pos = obs[0:3]
            handle_pos = np.array([0.0, -0.42, 0.65])

            prev_dist = np.linalg.norm(prev_ee_pos - handle_pos)
            curr_dist = np.linalg.norm(ee_pos - handle_pos)

            return 1.0 * (prev_dist - curr_dist)

        elif skill_id == SkillID.OPEN_WITH_CLEARANCE:
            # Reward for opening drawer
            prev_frac = obs[6]
            curr_frac = obs_next[6]

            return 5.0 * (curr_frac - prev_frac)

        elif skill_id == SkillID.RETRACT_SAFE:
            # Reward for getting to safe position
            ee_pos = obs_next[0:3]
            prev_ee_pos = obs[0:3]
            safe_pos = np.array([-0.3, 0.0, 0.8])

            prev_dist = np.linalg.norm(prev_ee_pos - safe_pos)
            curr_dist = np.linalg.norm(ee_pos - safe_pos)

            return 0.5 * (prev_dist - curr_dist)

        # For observation skills, reward is minimal
        return 0.0
