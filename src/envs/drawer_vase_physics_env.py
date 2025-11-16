#!/usr/bin/env python3
"""
Drawer+Vase Physics Environment (PyBullet)

Fei-Fei benchmark: Open a drawer while avoiding collision with a fragile vase.

This environment uses PyBullet for realistic physics simulation including:
- Cabinet with two sliding drawers (top + bottom)
- Fragile vase positioned near drawer pull path
- Robot end-effector for drawer manipulation
- Collision detection, clearance metrics, energy accounting

Compatible with Phase B synthetic flywheel (trust + econ + λ weighting).
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Try gymnasium first, fallback to gym, fallback to stub
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("WARNING: Neither gymnasium nor gym available. Using stub base class.")

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("WARNING: PyBullet not available. Using mock physics.")

from src.config.econ_params import EconParams


# Stub for when gym is not available
if not GYM_AVAILABLE:
    class spaces:
        @staticmethod
        def Box(low, high, shape, dtype):
            return {'type': 'Box', 'low': low, 'high': high, 'shape': shape, 'dtype': dtype}

    class gym:
        class Env:
            def __init__(self):
                self.observation_space = None
                self.action_space = None
                self.metadata = {}

            def reset(self, **kwargs):
                raise NotImplementedError

            def step(self, action):
                raise NotImplementedError


@dataclass
class DrawerVaseConfig:
    """Configuration for Drawer+Vase environment."""
    # Cabinet geometry
    cabinet_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    cabinet_size: Tuple[float, float, float] = (0.5, 0.4, 0.6)  # width, depth, height
    drawer_mass: float = 1.0
    drawer_friction: float = 0.3
    drawer_damping: float = 0.8
    drawer_max_extension: float = 0.35  # How far drawer can open

    # Vase geometry and fragility
    vase_pos: Tuple[float, float, float] = (0.3, 0.0, 0.8)  # Near drawer pull path
    vase_radius: float = 0.04
    vase_height: float = 0.12
    vase_mass: float = 0.3
    vase_fragility: float = 10.0  # Impulse threshold for breaking
    vase_tip_angle: float = 30.0  # Degrees before tipping = broken

    # Robot end-effector
    ee_start_pos: Tuple[float, float, float] = (-0.3, 0.0, 0.8)
    ee_mass: float = 0.5
    ee_max_force: float = 50.0
    ee_max_velocity: float = 0.3

    # Camera config
    camera_width: int = 128
    camera_height: int = 128
    camera_fov: float = 60.0
    camera_near: float = 0.1
    camera_far: float = 3.0
    camera_pos: Tuple[float, float, float] = (0.0, -1.0, 1.0)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.7)

    # Simulation
    physics_dt: float = 1.0 / 240.0
    control_dt: float = 1.0 / 30.0  # Control frequency
    max_steps: int = 300
    gravity: float = -9.81

    # Success/failure thresholds
    drawer_open_threshold: float = 0.9  # 90% open = success
    vase_collision_threshold: float = 5.0  # Max impulse before break
    sla_violation_contacts: int = 5  # Repeated high-risk contacts


class DrawerVasePhysicsEnv(gym.Env):
    """
    PyBullet-based Drawer+Vase environment.

    Task: Open the top drawer without hitting the fragile vase.

    Observation Modes:
        - 'state': Low-dim vector (joint angles, positions, clearances)
        - 'vision': RGB frames from PyBullet camera

    Action Space: 3D continuous (dx, dy, dz) end-effector velocity command
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        config: Optional[DrawerVaseConfig] = None,
        econ_params: Optional[EconParams] = None,
        obs_mode: str = 'state',
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.config = config or DrawerVaseConfig()
        self.econ_params = econ_params
        self.obs_mode = obs_mode
        self.render_mode = render_mode

        # Physics client
        self.physics_client = None
        self._setup_physics()

        # Action space: end-effector velocity (dx, dy, dz)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation space depends on mode
        if self.obs_mode == 'state':
            # State: [ee_pos(3), ee_vel(3), drawer_frac(1), vase_pos(3),
            #         vase_upright(1), min_clearance(1), grasp_state(1)]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
            )
        elif self.obs_mode == 'vision':
            # RGB image
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.config.camera_height, self.config.camera_width, 3),
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")

        # Episode state
        self.steps = 0
        self.t = 0.0
        self.energy_Wh = 0.0
        self.vase_intact = True
        self.drawer_opened_fraction = 0.0
        self.min_clearance = float('inf')
        self.total_impulse = 0.0
        self.n_high_risk_contacts = 0
        self.terminated_reason = None
        self.grasp_state = 0  # 0=not grasping, 1=grasping handle

        # PyBullet body IDs
        self.cabinet_id = None
        self.top_drawer_id = None
        self.bottom_drawer_id = None
        self.vase_id = None
        self.ee_id = None
        self.drawer_constraint = None

    def _setup_physics(self):
        """Initialize PyBullet physics simulation."""
        if not PYBULLET_AVAILABLE:
            return

        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.config.physics_dt, physicsClientId=self.physics_client)

    def _create_scene(self):
        """Create cabinet, drawers, vase, and end-effector."""
        if not PYBULLET_AVAILABLE:
            return

        # Ground plane
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # Cabinet body (static box)
        cabinet_half = [s / 2 for s in self.config.cabinet_size]
        cabinet_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=cabinet_half)
        cabinet_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=cabinet_half, rgbaColor=[0.6, 0.4, 0.2, 1.0])
        self.cabinet_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=cabinet_col,
            baseVisualShapeIndex=cabinet_vis,
            basePosition=self.config.cabinet_pos,
            physicsClientId=self.physics_client
        )

        # Top drawer (dynamic, constrained to slide)
        drawer_size = [self.config.cabinet_size[0] * 0.9, self.config.cabinet_size[1] * 0.8, 0.08]
        drawer_half = [s / 2 for s in drawer_size]
        drawer_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=drawer_half)
        drawer_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=drawer_half, rgbaColor=[0.7, 0.5, 0.3, 1.0])

        top_drawer_pos = [
            self.config.cabinet_pos[0],
            self.config.cabinet_pos[1] - self.config.cabinet_size[1] / 2 - 0.01,
            self.config.cabinet_pos[2] + self.config.cabinet_size[2] / 4
        ]
        self.top_drawer_id = p.createMultiBody(
            baseMass=self.config.drawer_mass,
            baseCollisionShapeIndex=drawer_col,
            baseVisualShapeIndex=drawer_vis,
            basePosition=top_drawer_pos,
            physicsClientId=self.physics_client
        )

        # Constrain drawer to slide along Y-axis (forward/backward)
        self.drawer_constraint = p.createConstraint(
            self.cabinet_id, -1, self.top_drawer_id, -1,
            p.JOINT_PRISMATIC,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, -self.config.cabinet_size[1] / 2, self.config.cabinet_size[2] / 4],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        p.changeConstraint(
            self.drawer_constraint,
            maxForce=100,
            physicsClientId=self.physics_client
        )

        # Apply friction and damping
        p.changeDynamics(
            self.top_drawer_id, -1,
            lateralFriction=self.config.drawer_friction,
            linearDamping=self.config.drawer_damping,
            physicsClientId=self.physics_client
        )

        # Bottom drawer (static for now, just visual)
        bottom_drawer_pos = [
            self.config.cabinet_pos[0],
            self.config.cabinet_pos[1] - self.config.cabinet_size[1] / 2 - 0.01,
            self.config.cabinet_pos[2] - self.config.cabinet_size[2] / 4
        ]
        self.bottom_drawer_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=drawer_col,
            baseVisualShapeIndex=drawer_vis,
            basePosition=bottom_drawer_pos,
            physicsClientId=self.physics_client
        )

        # Vase (cylinder)
        vase_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.config.vase_radius,
            height=self.config.vase_height
        )
        vase_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.config.vase_radius,
            length=self.config.vase_height,
            rgbaColor=[0.8, 0.2, 0.2, 1.0]
        )
        self.vase_id = p.createMultiBody(
            baseMass=self.config.vase_mass,
            baseCollisionShapeIndex=vase_col,
            baseVisualShapeIndex=vase_vis,
            basePosition=self.config.vase_pos,
            physicsClientId=self.physics_client
        )

        # End-effector (sphere)
        ee_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
        ee_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0.2, 0.6, 0.9, 1.0])
        self.ee_id = p.createMultiBody(
            baseMass=self.config.ee_mass,
            baseCollisionShapeIndex=ee_col,
            baseVisualShapeIndex=ee_vis,
            basePosition=self.config.ee_start_pos,
            physicsClientId=self.physics_client
        )

        # Disable gravity on EE (it's controlled)
        p.changeDynamics(self.ee_id, -1, linearDamping=0.9, physicsClientId=self.physics_client)

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if GYM_AVAILABLE:
            try:
                super().reset(seed=seed)
            except TypeError:
                # Old gym version without seed parameter
                super().reset()

        # Reset episode state
        self.steps = 0
        self.t = 0.0
        self.energy_Wh = 0.0
        self.vase_intact = True
        self.drawer_opened_fraction = 0.0
        self.min_clearance = float('inf')
        self.total_impulse = 0.0
        self.n_high_risk_contacts = 0
        self.terminated_reason = None
        self.grasp_state = 0

        if PYBULLET_AVAILABLE:
            # Reset simulation
            p.resetSimulation(physicsClientId=self.physics_client)
            p.setGravity(0, 0, self.config.gravity, physicsClientId=self.physics_client)
            p.setTimeStep(self.config.physics_dt, physicsClientId=self.physics_client)

            # Recreate scene
            self._create_scene()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one control step."""
        # Parse action (end-effector velocity command)
        action = np.clip(action, -1.0, 1.0)
        ee_vel = action * self.config.ee_max_velocity

        # Apply action and simulate
        prev_energy = self.energy_Wh

        if PYBULLET_AVAILABLE:
            # Set EE velocity
            current_ee_pos, _ = p.getBasePositionAndOrientation(
                self.ee_id, physicsClientId=self.physics_client
            )
            target_pos = np.array(current_ee_pos) + ee_vel * self.config.control_dt

            # Use position control with velocity
            p.resetBaseVelocity(
                self.ee_id,
                linearVelocity=ee_vel.tolist(),
                angularVelocity=[0, 0, 0],
                physicsClientId=self.physics_client
            )

            # Grasp logic: if EE is close to drawer handle, apply force
            drawer_pos, _ = p.getBasePositionAndOrientation(
                self.top_drawer_id, physicsClientId=self.physics_client
            )
            ee_pos = np.array(current_ee_pos)
            handle_pos = np.array(drawer_pos) + np.array([0, -0.02, 0])
            dist_to_handle = np.linalg.norm(ee_pos - handle_pos)

            if dist_to_handle < 0.05:
                self.grasp_state = 1
                # Pull drawer with EE motion
                pull_force = ee_vel[1] * self.config.ee_max_force
                p.applyExternalForce(
                    self.top_drawer_id, -1,
                    [0, pull_force, 0],
                    drawer_pos,
                    p.WORLD_FRAME,
                    physicsClientId=self.physics_client
                )
            else:
                self.grasp_state = 0

            # Simulate physics
            n_substeps = int(self.config.control_dt / self.config.physics_dt)
            for _ in range(n_substeps):
                p.stepSimulation(physicsClientId=self.physics_client)

            # Check collisions
            self._check_collisions()

            # Update drawer fraction
            drawer_pos_new, _ = p.getBasePositionAndOrientation(
                self.top_drawer_id, physicsClientId=self.physics_client
            )
            initial_y = self.config.cabinet_pos[1] - self.config.cabinet_size[1] / 2 - 0.01
            drawer_extension = drawer_pos_new[1] - initial_y
            self.drawer_opened_fraction = np.clip(
                drawer_extension / self.config.drawer_max_extension, 0.0, 1.0
            )

            # Update clearance
            ee_pos_new, _ = p.getBasePositionAndOrientation(
                self.ee_id, physicsClientId=self.physics_client
            )
            vase_pos, _ = p.getBasePositionAndOrientation(
                self.vase_id, physicsClientId=self.physics_client
            )
            clearance = np.linalg.norm(np.array(ee_pos_new) - np.array(vase_pos))
            self.min_clearance = min(self.min_clearance, clearance)

            # Check vase tipping
            vase_orn = p.getBasePositionAndOrientation(self.vase_id, physicsClientId=self.physics_client)[1]
            vase_euler = p.getEulerFromQuaternion(vase_orn)
            tip_angle = np.degrees(np.sqrt(vase_euler[0]**2 + vase_euler[1]**2))
            if tip_angle > self.config.vase_tip_angle:
                self.vase_intact = False
                self.terminated_reason = "vase_tipped"

        # Energy accounting
        delta_energy_Wh = np.linalg.norm(ee_vel) * self.config.control_dt * 0.01  # Simplified
        self.energy_Wh += delta_energy_Wh

        # Time and step counter
        self.t += self.config.control_dt
        self.steps += 1

        # Check termination
        done = False

        if not self.vase_intact:
            done = True
            if self.terminated_reason is None:
                self.terminated_reason = "vase_broken"

        if self.drawer_opened_fraction >= self.config.drawer_open_threshold:
            done = True
            if self.terminated_reason is None:
                self.terminated_reason = "success"

        if self.n_high_risk_contacts >= self.config.sla_violation_contacts:
            done = True
            if self.terminated_reason is None:
                self.terminated_reason = "sla_violation"

        if self.steps >= self.config.max_steps:
            done = True
            if self.terminated_reason is None:
                self.terminated_reason = "max_steps"

        # Compute observation and info
        obs = self._get_observation()
        info = self._get_info()

        # Reward is 0; actual reward comes from trainer (compute_econ_reward)
        reward = 0.0

        return obs, reward, done, False, info

    def _check_collisions(self):
        """Check for collisions between EE/drawer and vase."""
        if not PYBULLET_AVAILABLE:
            return

        # EE-Vase collision
        contacts = p.getContactPoints(
            self.ee_id, self.vase_id, physicsClientId=self.physics_client
        )
        for contact in contacts:
            impulse = contact[9]  # Normal force
            self.total_impulse += impulse
            if impulse > self.config.vase_collision_threshold:
                self.vase_intact = False
                self.terminated_reason = "vase_collision"
            if impulse > 1.0:
                self.n_high_risk_contacts += 1

        # Drawer-Vase collision
        contacts = p.getContactPoints(
            self.top_drawer_id, self.vase_id, physicsClientId=self.physics_client
        )
        for contact in contacts:
            impulse = contact[9]
            self.total_impulse += impulse
            if impulse > self.config.vase_collision_threshold:
                self.vase_intact = False
                self.terminated_reason = "vase_collision"

    def _get_observation(self):
        """Get observation based on mode."""
        if self.obs_mode == 'state':
            return self._get_state_observation()
        elif self.obs_mode == 'vision':
            return self._get_vision_observation()
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

    def _get_state_observation(self):
        """Get low-dimensional state observation."""
        if PYBULLET_AVAILABLE:
            ee_pos, _ = p.getBasePositionAndOrientation(
                self.ee_id, physicsClientId=self.physics_client
            )
            ee_vel, _ = p.getBaseVelocity(self.ee_id, physicsClientId=self.physics_client)
            vase_pos, vase_orn = p.getBasePositionAndOrientation(
                self.vase_id, physicsClientId=self.physics_client
            )
            vase_euler = p.getEulerFromQuaternion(vase_orn)
            vase_upright = 1.0 - np.sqrt(vase_euler[0]**2 + vase_euler[1]**2) / np.pi
        else:
            ee_pos = self.config.ee_start_pos
            ee_vel = [0, 0, 0]
            vase_pos = self.config.vase_pos
            vase_upright = 1.0

        obs = np.array([
            ee_pos[0], ee_pos[1], ee_pos[2],
            ee_vel[0], ee_vel[1], ee_vel[2],
            self.drawer_opened_fraction,
            vase_pos[0], vase_pos[1], vase_pos[2],
            vase_upright,
            self.min_clearance if self.min_clearance != float('inf') else 1.0,
            float(self.grasp_state)
        ], dtype=np.float32)

        return obs

    def _get_vision_observation(self):
        """Get RGB image from PyBullet camera."""
        if not PYBULLET_AVAILABLE:
            return np.zeros(
                (self.config.camera_height, self.config.camera_width, 3),
                dtype=np.uint8
            )

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.config.camera_pos,
            cameraTargetPosition=self.config.camera_target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.physics_client
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=self.config.camera_width / self.config.camera_height,
            nearVal=self.config.camera_near,
            farVal=self.config.camera_far,
            physicsClientId=self.physics_client
        )

        # Render
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.config.camera_width,
            height=self.config.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.render_mode == 'human' else p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client
        )

        # Convert to uint8 RGB
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            self.config.camera_height, self.config.camera_width, 4
        )[:, :, :3]

        return rgb_array

    def _get_info(self):
        """Get info dict compatible with EpisodeInfoSummary."""
        dt_hours = self.t / 3600.0
        mpl_t = 0.0  # Drawer open is binary, not units/hour
        ep_t = 0.0  # Energy productivity N/A for drawer task

        # Compute "units" as drawer fraction (for compatibility)
        units_done = self.drawer_opened_fraction

        # Error rate based on vase contacts
        error_rate = self.total_impulse / (self.config.vase_fragility * 10 + 1e-6)

        # Economic metrics (if econ_params provided)
        if self.econ_params and hasattr(self.econ_params, 'value_per_unit'):
            revenue_step = self.econ_params.value_per_unit * (1.0 if self.terminated_reason == "success" else 0.0)
            error_cost_step = self.econ_params.damage_cost * (0.0 if self.vase_intact else 1.0)
            profit_step = revenue_step - error_cost_step
        else:
            revenue_step = 0.0
            error_cost_step = 0.0
            profit_step = 0.0

        info = {
            # Step-level metrics
            "drawer_fraction": self.drawer_opened_fraction,
            "vase_intact": self.vase_intact,
            "min_clearance": self.min_clearance if self.min_clearance != float('inf') else 0.0,
            "total_impulse": self.total_impulse,
            "n_high_risk_contacts": self.n_high_risk_contacts,
            "grasp_state": self.grasp_state,

            # Episode-level (cumulative)
            "t": self.t,
            "energy_Wh": self.energy_Wh,
            "steps": self.steps,
            "units_done": units_done,
            "errors": 0 if self.vase_intact else 1,
            "error_rate_t": error_rate,
            "mpl_t": mpl_t,
            "ep_t": ep_t,

            # Economics
            "profit_step": profit_step,
            "revenue_step": revenue_step,
            "error_cost_step": error_cost_step,

            # Termination
            "terminated_reason": self.terminated_reason,
            "success": self.terminated_reason == "success",
        }

        return info

    def render(self):
        """Render environment."""
        if self.render_mode == 'rgb_array':
            return self._get_vision_observation()
        elif self.render_mode == 'human':
            # PyBullet GUI handles rendering
            pass
        return None

    def close(self):
        """Clean up PyBullet."""
        if PYBULLET_AVAILABLE and self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


def summarize_drawer_vase_episode(info_history):
    """
    Aggregate per-step info dicts into EpisodeInfoSummary-compatible format.

    Compatible with src/envs/dishwashing_env.py summarize_episode_info.
    """
    from src.envs.dishwashing_env import EpisodeInfoSummary

    if not info_history:
        return EpisodeInfoSummary(
            termination_reason="unknown",
            mpl_episode=0.0,
            ep_episode=0.0,
            error_rate_episode=0.0,
            throughput_units_per_hour=0.0,
            energy_Wh=0.0,
            profit=0.0,
            wage_parity=None,
        )

    last = info_history[-1]
    total_energy = last.get("energy_Wh", 0.0)
    time_hours = last.get("t", 0.0) / 3600.0 if "t" in last else 0.0

    # For drawer task: "units" = successful drawer opens (0 or 1)
    success = last.get("success", False)
    units_done = 1.0 if success else 0.0

    mpl_episode = (units_done / time_hours) if time_hours > 0 else 0.0
    ep_episode = (units_done / total_energy) if total_energy > 0 else 0.0

    # Error rate: vase broken = 1.0 error rate
    vase_intact = last.get("vase_intact", True)
    error_rate_episode = 0.0 if vase_intact else 1.0

    throughput_units_per_hour = mpl_episode
    profit = sum(step.get("profit_step", 0.0) for step in info_history)

    return EpisodeInfoSummary(
        termination_reason=last.get("terminated_reason", "unknown") or "unknown",
        mpl_episode=mpl_episode,
        ep_episode=ep_episode,
        error_rate_episode=error_rate_episode,
        throughput_units_per_hour=throughput_units_per_hour,
        energy_Wh=total_energy,
        profit=profit,
        wage_parity=None,
    )
