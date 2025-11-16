"""
SIMA-2 Co-Agent.

Generates demonstration trajectories with natural language annotations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.hrl.skills import SkillID, SkillParams
from src.hrl.low_level_policy import ScriptedSkillPolicy
from src.hrl.skill_termination import SkillTerminationDetector
from .narrator import Narrator


@dataclass
class SIMATrajectory:
    """
    Complete trajectory from SIMA co-agent.

    Contains:
    - High-level instruction
    - Skill sequence plan
    - Frame-by-frame data
    - Step-level narrations
    """
    instruction: str = ""
    plan: List[Dict[str, Any]] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    narrations: List[str] = field(default_factory=list)
    skill_ids: List[int] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    frames: List[np.ndarray] = field(default_factory=list)
    success: bool = False
    total_steps: int = 0

    def to_dict(self):
        """Convert to dictionary for saving."""
        return {
            'instruction': self.instruction,
            'plan': self.plan,
            'states': [s.tolist() for s in self.states],
            'actions': [a.tolist() for a in self.actions],
            'narrations': self.narrations,
            'skill_ids': self.skill_ids,
            'success': self.success,
            'total_steps': self.total_steps,
        }

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(
            instruction=d['instruction'],
            plan=d['plan'],
            states=[np.array(s) for s in d['states']],
            actions=[np.array(a) for a in d['actions']],
            narrations=d['narrations'],
            skill_ids=d['skill_ids'],
            success=d['success'],
            total_steps=d['total_steps'],
        )

    def get_skill_sequence(self):
        """Get unique skill sequence from trajectory."""
        sequence = []
        prev_skill = None
        for sid in self.skill_ids:
            if sid != prev_skill:
                sequence.append(sid)
                prev_skill = sid
        return sequence

    def get_vla_training_sample(self):
        """Convert to VLA training sample."""
        skill_sequence = self.get_skill_sequence()

        # Extract skill parameters (from plan)
        skill_params = []
        for skill_info in self.plan:
            params = skill_info.get('params', SkillParams().to_array())
            if isinstance(params, SkillParams):
                params = params.to_array()
            skill_params.append(np.array(params, dtype=np.float32))

        # Pad if needed
        while len(skill_params) < len(skill_sequence):
            skill_params.append(SkillParams().to_array())

        return {
            'instruction': self.instruction,
            'skill_sequence': skill_sequence,
            'skill_params': skill_params[:len(skill_sequence)],
        }


class SIMACoAgent:
    """
    SIMA-style co-agent for demonstration generation.

    Receives natural language instructions, generates plans,
    executes actions, and provides step-level narrations.
    """

    def __init__(
        self,
        policy=None,
        narrator=None,
        termination_detector=None,
        planner_type='scripted'
    ):
        """
        Args:
            policy: Low-level action policy
            narrator: Narration generator
            termination_detector: Skill termination detector
            planner_type: 'scripted' or 'learned'
        """
        self.policy = policy or ScriptedSkillPolicy()
        self.narrator = narrator or Narrator()
        self.termination_detector = termination_detector or SkillTerminationDetector()
        self.planner_type = planner_type

        # Current task state
        self.instruction = ""
        self.plan = []
        self.current_skill_idx = 0

        # Trajectory recording
        self.trajectory = SIMATrajectory()

    def reset(self, instruction):
        """
        Reset agent with new instruction.

        Args:
            instruction: Natural language task instruction
        """
        self.instruction = instruction
        self.plan = self.generate_plan(instruction)
        self.current_skill_idx = 0

        # Reset trajectory
        self.trajectory = SIMATrajectory(
            instruction=instruction,
            plan=self.plan
        )

    def generate_plan(self, instruction):
        """
        Generate high-level plan from instruction.

        Args:
            instruction: Natural language instruction

        Returns:
            plan: List of (skill_name, params, narration) dicts
        """
        instruction_lower = instruction.lower()

        # Analyze instruction for task modifications
        careful = 'careful' in instruction_lower or 'safe' in instruction_lower
        quick = 'quick' in instruction_lower or 'fast' in instruction_lower
        avoid_vase = 'vase' in instruction_lower or 'fragile' in instruction_lower

        # Determine parameters
        if careful:
            clearance = 0.2
            speed = 0.4
        elif quick:
            clearance = 0.12
            speed = 0.8
        else:
            clearance = 0.15
            speed = 0.6

        # Build plan
        plan = []

        # Skill 0: Locate drawer
        plan.append({
            'skill_id': SkillID.LOCATE_DRAWER,
            'skill_name': 'LOCATE_DRAWER',
            'params': SkillParams.default_for_skill(SkillID.LOCATE_DRAWER),
            'narration': self.narrator.narrate_skill_start(SkillID.LOCATE_DRAWER),
            'priority': 1
        })

        # Skill 1: Locate vase (if mentioned)
        if avoid_vase or 'vase' in instruction_lower:
            plan.append({
                'skill_id': SkillID.LOCATE_VASE,
                'skill_name': 'LOCATE_VASE',
                'params': SkillParams.default_for_skill(SkillID.LOCATE_VASE),
                'narration': self.narrator.narrate_skill_start(SkillID.LOCATE_VASE),
                'priority': 1
            })

        # Skill 2: Plan safe approach
        safe_params = SkillParams(target_clearance=clearance)
        plan.append({
            'skill_id': SkillID.PLAN_SAFE_APPROACH,
            'skill_name': 'PLAN_SAFE_APPROACH',
            'params': safe_params,
            'narration': self.narrator.narrate_skill_start(SkillID.PLAN_SAFE_APPROACH),
            'priority': 2
        })

        # Skill 3: Grasp handle
        grasp_params = SkillParams(approach_speed=speed, grasp_force=0.5)
        plan.append({
            'skill_id': SkillID.GRASP_HANDLE,
            'skill_name': 'GRASP_HANDLE',
            'params': grasp_params,
            'narration': self.narrator.narrate_skill_start(SkillID.GRASP_HANDLE),
            'priority': 3
        })

        # Skill 4: Open with clearance
        open_params = SkillParams(
            target_clearance=clearance,
            pull_speed=speed
        )
        plan.append({
            'skill_id': SkillID.OPEN_WITH_CLEARANCE,
            'skill_name': 'OPEN_WITH_CLEARANCE',
            'params': open_params,
            'narration': self.narrator.narrate_skill_start(SkillID.OPEN_WITH_CLEARANCE),
            'priority': 4
        })

        # Skill 5: Retract safely
        plan.append({
            'skill_id': SkillID.RETRACT_SAFE,
            'skill_name': 'RETRACT_SAFE',
            'params': SkillParams.default_for_skill(SkillID.RETRACT_SAFE),
            'narration': self.narrator.narrate_skill_start(SkillID.RETRACT_SAFE),
            'priority': 5
        })

        return plan

    def step(self, obs, info):
        """
        Generate action and narration for current step.

        Args:
            obs: (13,) current observation
            info: Environment info dict

        Returns:
            action: (3,) action vector
            narration: str describing action
            skill_done: bool
            task_done: bool
        """
        if self.current_skill_idx >= len(self.plan):
            return np.zeros(3), "Task complete", True, True

        # Get current skill
        skill_info = self.plan[self.current_skill_idx]
        skill_id = skill_info['skill_id']
        skill_params = skill_info['params']

        # Generate action
        action = self.policy.act(obs, skill_id, skill_params)

        # Generate step narration
        narration = self.narrator.narrate_step(obs, action, info, skill_id)

        # Check for safety warnings
        min_clearance = obs[11]
        safety_warning = self.narrator.narrate_safety_warning(min_clearance)
        if safety_warning:
            narration = f"{narration}. {safety_warning}"

        # Record trajectory
        self.trajectory.states.append(obs)
        self.trajectory.actions.append(action)
        self.trajectory.narrations.append(narration)
        self.trajectory.skill_ids.append(skill_id)
        self.trajectory.infos.append(info)
        self.trajectory.total_steps += 1

        # Check skill termination
        skill_done, skill_success, reason = self.termination_detector.is_done(
            skill_id,
            obs,
            info,
            self._count_skill_steps(skill_id),
            skill_params.timeout_steps if isinstance(skill_params, SkillParams) else 100
        )

        task_done = False

        if skill_done:
            completion_narration = self.narrator.narrate_skill_completion(
                skill_id, skill_success, reason
            )
            self.trajectory.narrations[-1] += f". {completion_narration}"

            self.current_skill_idx += 1

            if self.current_skill_idx >= len(self.plan):
                task_done = True
            elif self.current_skill_idx < len(self.plan):
                # Announce next skill
                next_skill = self.plan[self.current_skill_idx]
                next_narration = next_skill['narration']
                self.trajectory.narrations[-1] += f". Next: {next_narration}"

        # Check for task completion conditions
        if info.get('success', False):
            task_done = True
            self.trajectory.success = True

        if not info.get('vase_intact', True):
            task_done = True
            self.trajectory.success = False

        return action, narration, skill_done, task_done

    def _count_skill_steps(self, skill_id):
        """Count how many steps have been executed for current skill."""
        count = 0
        for sid in reversed(self.trajectory.skill_ids):
            if sid == skill_id:
                count += 1
            else:
                break
        return count

    def get_trajectory(self):
        """
        Get complete trajectory.

        Returns:
            trajectory: SIMATrajectory object
        """
        return self.trajectory

    def generate_full_demonstration(self, env, instruction, max_steps=500):
        """
        Generate complete demonstration trajectory.

        Args:
            env: Environment to execute in
            instruction: Task instruction
            max_steps: Maximum steps

        Returns:
            trajectory: SIMATrajectory
        """
        self.reset(instruction)

        obs, info = env.reset()
        start_narration = self.narrator.narrate_task_start(instruction)
        self.trajectory.narrations.append(start_narration)

        for step in range(max_steps):
            action, narration, skill_done, task_done = self.step(obs, info)

            # Execute action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs

            if task_done or done:
                # Final narration
                final_stats = {
                    'drawer_frac': obs[6],
                    'vase_intact': info.get('vase_intact', True),
                    'total_steps': self.trajectory.total_steps,
                    'reason': info.get('terminated_reason', '')
                }
                final_narration = self.narrator.narrate_task_completion(
                    info.get('success', False), final_stats
                )
                self.trajectory.narrations.append(final_narration)
                self.trajectory.success = info.get('success', False)
                break

        return self.trajectory

    def get_instruction_to_plan_mapping(self):
        """
        Get mapping from instruction to plan for training.

        Returns:
            mapping: dict with instruction and plan details
        """
        return {
            'instruction': self.instruction,
            'skill_sequence': [s['skill_id'] for s in self.plan],
            'skill_names': [s['skill_name'] for s in self.plan],
            'skill_params': [
                s['params'].to_array() if isinstance(s['params'], SkillParams)
                else s['params']
                for s in self.plan
            ],
            'plan_narrations': [s['narration'] for s in self.plan]
        }


class SIMADataCollector:
    """
    Collects SIMA demonstration trajectories at scale.
    """

    def __init__(self, env, co_agent=None):
        """
        Args:
            env: Environment
            co_agent: SIMACoAgent instance
        """
        self.env = env
        self.co_agent = co_agent or SIMACoAgent()
        self.trajectories = []

    def collect_trajectory(self, instruction):
        """
        Collect single demonstration.

        Args:
            instruction: Task instruction

        Returns:
            trajectory: SIMATrajectory
        """
        trajectory = self.co_agent.generate_full_demonstration(self.env, instruction)
        self.trajectories.append(trajectory)
        return trajectory

    def collect_batch(self, instructions, verbose=True):
        """
        Collect batch of demonstrations.

        Args:
            instructions: List of instructions
            verbose: Print progress

        Returns:
            trajectories: List of SIMATrajectory
        """
        for i, instruction in enumerate(instructions):
            if verbose:
                print(f"Collecting trajectory {i+1}/{len(instructions)}: {instruction[:50]}...")

            traj = self.collect_trajectory(instruction)

            if verbose:
                print(f"  Success: {traj.success}, Steps: {traj.total_steps}")

        return self.trajectories

    def generate_instruction_batch(self, n_instructions=100):
        """
        Generate batch of varied instructions.

        Args:
            n_instructions: Number of instructions

        Returns:
            instructions: List of instructions
        """
        base_instructions = [
            "open the drawer without hitting the vase",
            "carefully open the top drawer",
            "pull the drawer open while avoiding the fragile vase",
            "open the drawer safely",
            "open the cabinet drawer without touching the vase",
        ]

        instructions = []

        for _ in range(n_instructions):
            base = np.random.choice(base_instructions)
            variations = self.co_agent.narrator.generate_instruction_variations(base)
            instructions.append(np.random.choice(variations))

        return instructions

    def get_vla_training_data(self):
        """
        Convert collected trajectories to VLA training format.

        Returns:
            training_data: List of VLA training samples
        """
        return [traj.get_vla_training_sample() for traj in self.trajectories]

    def get_statistics(self):
        """
        Get collection statistics.

        Returns:
            stats: dict with statistics
        """
        if len(self.trajectories) == 0:
            return {'n_trajectories': 0}

        successes = sum(t.success for t in self.trajectories)
        steps = [t.total_steps for t in self.trajectories]
        narrations = [len(t.narrations) for t in self.trajectories]

        return {
            'n_trajectories': len(self.trajectories),
            'success_rate': successes / len(self.trajectories),
            'mean_steps': np.mean(steps),
            'std_steps': np.std(steps),
            'mean_narrations': np.mean(narrations),
            'unique_instructions': len(set(t.instruction for t in self.trajectories)),
        }

    def save_trajectories(self, path):
        """Save collected trajectories."""
        import json
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            'trajectories': [t.to_dict() for t in self.trajectories],
            'statistics': self.get_statistics()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_trajectories(cls, path):
        """Load saved trajectories."""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        trajectories = [SIMATrajectory.from_dict(d) for d in data['trajectories']]
        return trajectories
