"""
PyBullet-based Dishwashing Physics Environment (Phase A: Calibrated Realism)

Phase A improvements:
1. Stochastic realism: random dish mass, poses, camera jitter, lighting
2. Better error model: slip, gripper failure, misalignment, wet friction
3. Human-ish throughput: capped speed/torque (no anime-speed bullshit)
4. Realistic error detection: not just z < -0.1

Returns RGB video observations from simulated dishwashing task.
"""
import numpy as np
import pybullet as p
import pybullet_data
from collections import deque
from typing import Tuple, Dict, Any
import time

class DishwashingPhysicsEnv:
    """
    Physics-based dishwashing environment with PyBullet backend.

    Returns video observations (T, C, H, W) from fixed camera.
    Economics layer unchanged - same info dict as DishwashingEnv.
    """

    def __init__(
        self,
        frames=8,
        image_size=(64, 64),
        max_steps=60,
        headless=True,
        camera_config=None,
        # Phase A: Stochastic realism
        randomize_dishes=True,
        camera_jitter=0.02,
        lighting_variation=0.1,
        # Phase A: Better error model
        slip_probability=0.05,
        gripper_failure_rate=0.02,
        # Phase A: Human-ish throughput caps
        max_speed_multiplier=2.0,  # Cap at 2x human speed
        max_acceleration=1.0  # Limit sudden movements
    ):
        """
        Args:
            frames: Number of frames in temporal stack
            image_size: (height, width) of rendered images
            max_steps: Maximum steps per episode
            headless: If True, run without GUI
            camera_config: Dict with camera params (position, target, fov)

            # Phase A: Stochastic realism
            randomize_dishes: Random dish mass & starting poses per episode
            camera_jitter: Camera position noise (meters)
            lighting_variation: Light intensity variation (fraction)

            # Phase A: Better error model
            slip_probability: Probability of dish slipping when grasped
            gripper_failure_rate: Probability of gripper malfunction

            # Phase A: Human-ish caps
            max_speed_multiplier: Maximum speed relative to baseline
            max_acceleration: Maximum acceleration (m/s^2)
        """
        self.frames = frames
        self.image_height, self.image_width = image_size
        self.max_steps = max_steps
        self.headless = headless

        # Phase A: Stochastic realism parameters
        self.randomize_dishes = randomize_dishes
        self.camera_jitter = camera_jitter
        self.lighting_variation = lighting_variation

        # Phase A: Better error model
        self.slip_probability = slip_probability
        self.gripper_failure_rate = gripper_failure_rate

        # Phase A: Human-ish throughput caps
        self.max_speed_multiplier = max_speed_multiplier
        self.max_acceleration = max_acceleration

        # Camera configuration (will be jittered per episode if enabled)
        if camera_config is None:
            camera_config = {
                'position': [0.0, 0.5, 0.8],
                'target': [0.0, 0.0, 0.0],
                'fov': 60
            }
        self.base_camera_config = camera_config
        self.camera_config = camera_config.copy()

        # Episode state
        self.t = 0.0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
        self.step_count = 0

        # Frame buffer for temporal stacking
        self.frame_buffer = deque(maxlen=frames)

        # Physics parameters (tuned from speed/care)
        self.current_speed = 0.5
        self.current_care = 0.5
        self.previous_velocity = np.zeros(3)  # For acceleration limiting

        # PyBullet client (will be initialized in reset)
        self.physics_client = None
        self.robot_id = None
        self.controlled_joint_ids = []
        self.controlled_joint_ids = []
        self.dishes = []
        self.dish_properties = []  # Store per-episode dish properties

        # Observation space shape
        self.observation_space_shape = (frames, 3, self.image_height, self.image_width)

    def reset(self) -> np.ndarray:
        """Reset environment and return initial video observation."""
        # Disconnect previous client if exists
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        # Connect to PyBullet
        if self.headless:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)

        # Set up simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        # Load ground plane
        p.loadURDF("plane.urdf")

        # Create simple "sink" (box)
        sink_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.05])
        sink_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.05], rgbaColor=[0.7, 0.7, 0.7, 1])
        self.sink_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sink_collision,
            baseVisualShapeIndex=sink_visual,
            basePosition=[0, 0, -0.05]
        )

        # Create "robot arm" (simple gripper simulation)
        # Load a lightweight articulated arm (KUKA-IIWA)
        kuka_urdf = pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf"
        self.robot_id = p.loadURDF(
            kuka_urdf,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        # Controlled joints: all revolute joints except last flange
        self.controlled_joint_ids = []
        for j in range(p.getNumJoints(self.robot_id)):
            ji = p.getJointInfo(self.robot_id, j)
            joint_type = ji[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.controlled_joint_ids.append(j)

        # Disable default motors and set damping
        p.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.controlled_joint_ids)
        )

        # Phase A: Randomize camera position (jitter)
        if self.camera_jitter > 0:
            jitter = np.random.uniform(-self.camera_jitter, self.camera_jitter, size=3)
            self.camera_config['position'] = [
                self.base_camera_config['position'][i] + jitter[i]
                for i in range(3)
            ]
        else:
            self.camera_config = self.base_camera_config.copy()

        # Create dish stack (cylinders representing dishes)
        # Phase A: Randomize dish properties per episode
        self.dishes = []
        self.dish_properties = []

        for i in range(10):  # Stack of 10 dishes
            # Phase A: Random dish mass (0.05 to 0.15 kg)
            if self.randomize_dishes:
                dish_mass = np.random.uniform(0.05, 0.15)
                # Random starting position (slight lateral offset)
                pos_x = 0.15 + np.random.uniform(-0.02, 0.02)
                pos_y = np.random.uniform(-0.02, 0.02)
                # Random friction (wet vs dry dishes)
                lateral_friction = np.random.uniform(0.3, 0.7)
            else:
                dish_mass = 0.1
                pos_x, pos_y = 0.15, 0.0
                lateral_friction = 0.5

            dish_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=0.01)
            dish_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.08,
                length=0.01,
                rgbaColor=[0.9, 0.9, 0.9, 1]
            )
            dish_id = p.createMultiBody(
                baseMass=dish_mass,
                baseCollisionShapeIndex=dish_collision,
                baseVisualShapeIndex=dish_visual,
                basePosition=[pos_x, pos_y, 0.05 + i * 0.015]
            )

            # Set friction
            p.changeDynamics(dish_id, -1, lateralFriction=lateral_friction)

            self.dishes.append(dish_id)
            self.dish_properties.append({
                'mass': dish_mass,
                'friction': lateral_friction,
                'is_wet': lateral_friction < 0.4  # Low friction = wet/slippery
            })

        # Reset episode state
        self.t = 0.0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
        self.step_count = 0

        # Render initial frames
        initial_frame = self._render_camera()
        self.frame_buffer.clear()
        for _ in range(self.frames):
            self.frame_buffer.append(initial_frame.copy())

        return self._get_video_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """
        Step environment with action.

        Args:
            action: [speed, care] in [0, 1]^2

        Returns:
            video_obs: (T, C, H, W) video observation
            info: Dict with task metrics (for economics)
            done: Episode termination flag
        """
        step_start = time.time()

        speed, care = action
        self.current_speed = np.clip(speed, 0, 1)
        self.current_care = np.clip(care, 0, 1)

        # Phase A: Cap speed to human-ish throughput (no anime-speed)
        # Baseline speed = 0.5, max allowed = max_speed_multiplier * baseline
        baseline_speed = 0.5
        effective_speed = min(self.current_speed, self.max_speed_multiplier * baseline_speed)

        # Update robot position based on capped speed
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)

        # Simple movement: oscillate horizontally (simulating washing motion)
        time_phase = self.step_count * 0.1
        target_x = 0.15 * np.sin(time_phase * effective_speed * 5)
        target_y = 0.0
        target_z = 0.1 + 0.05 * np.sin(time_phase * 2)  # Up-down motion

        # Add jitter based on care (low care = more jitter)
        jitter_scale = (1.0 - self.current_care) * 0.05
        target_x += np.random.randn() * jitter_scale
        target_y += np.random.randn() * jitter_scale

        # Phase A: Acceleration smoothing (low-pass filter)
        # Compute desired velocity
        dt = 1.0 / 240.0 * 10  # Time for 10 sub-steps
        desired_velocity = np.array([
            (target_x - robot_pos[0]) / dt,
            (target_y - robot_pos[1]) / dt,
            (target_z - robot_pos[2]) / dt
        ])

        # Acceleration limiting: smooth transition from previous velocity
        # Use exponential moving average (alpha controls smoothing)
        alpha = 0.3  # Lower = more smoothing, higher = more responsive
        smoothed_velocity = alpha * desired_velocity + (1 - alpha) * self.previous_velocity

        # Clip acceleration to max_acceleration
        velocity_delta = smoothed_velocity - self.previous_velocity
        accel_magnitude = np.linalg.norm(velocity_delta) / dt
        if accel_magnitude > self.max_acceleration:
            # Scale down velocity change to respect acceleration limit
            velocity_delta = velocity_delta * (self.max_acceleration * dt / np.linalg.norm(velocity_delta))
            smoothed_velocity = self.previous_velocity + velocity_delta

        # Update previous velocity for next step
        self.previous_velocity = smoothed_velocity

        # Apply smoothed velocity control
        # Compute target position from smoothed velocity
        target_pos = robot_pos + smoothed_velocity * dt
        p.resetBasePositionAndOrientation(
            self.robot_id,
            target_pos,
            robot_orn
        )

        # Simulate physics
        for _ in range(10):  # Sub-steps for smoother simulation
            p.stepSimulation()

        # Phase A: Better error model (4 channels)
        # Check for dish interactions (attempt)
        # Use time-based triggering instead of strict spatial condition
        # This ensures attempts actually happen so errors can occur
        attempt_probability = 0.1 * self.current_speed  # Speed-dependent (0-0.1 per step)
        if np.random.rand() < attempt_probability:
            self.attempts += 1

            # Initialize error flag and type
            is_error = False
            error_type = None

            # Channel 1: Slip probability (wet dishes)
            # If grasping a wet/low-friction dish, higher chance of slip
            for i, dish_id in enumerate(self.dishes):
                dish_pos, _ = p.getBasePositionAndOrientation(dish_id)

                # Check if robot is near this dish
                dist_to_dish = np.linalg.norm(np.array([target_x, target_y, target_z]) - np.array(dish_pos))
                if dist_to_dish < 0.15:  # Within grasp range
                    # Wet dishes have higher slip probability
                    if self.dish_properties[i]['is_wet']:
                        slip_chance = self.slip_probability * 2.0  # Double for wet
                    else:
                        slip_chance = self.slip_probability

                    # Low care increases slip chance
                    slip_chance *= (1.5 - self.current_care)  # 1.5x at care=0, 0.5x at care=1

                    if np.random.rand() < slip_chance:
                        is_error = True
                        error_type = 'slip'
                        break

            # Channel 2: Random gripper failure
            if not is_error and np.random.rand() < self.gripper_failure_rate:
                is_error = True
                error_type = 'gripper_failure'

            # Channel 3: Contact force-based breaks
            if not is_error:
                # Check contact points between robot and dishes
                for i, dish_id in enumerate(self.dishes):
                    contact_points = p.getContactPoints(self.robot_id, dish_id)
                    if len(contact_points) > 0:
                        # Get maximum normal force
                        max_force = max([pt[9] for pt in contact_points])

                        # Break threshold depends on care (low care = harsh, high break chance)
                        break_threshold = 5.0 * self.current_care  # 0-5 Newtons

                        if max_force > break_threshold:
                            is_error = True
                            error_type = 'harsh_contact'
                            break

            # Channel 4: Misalignment detection (position fell too far or rotated)
            if not is_error:
                for i, dish_id in enumerate(self.dishes):
                    dish_pos, dish_orn = p.getBasePositionAndOrientation(dish_id)

                    # Check if dish fell below threshold
                    if dish_pos[2] < -0.1:
                        is_error = True
                        error_type = 'fell'
                        break

                    # Check lateral misalignment (dishes stacked too far from center)
                    lateral_offset = np.sqrt(dish_pos[0]**2 + dish_pos[1]**2)
                    if lateral_offset > 0.5:  # More than 50cm from center
                        is_error = True
                        error_type = 'misalignment'
                        break

                    # Check tilt (using quaternion to detect if dish is too tilted)
                    # Simplified: check if z-component of up vector is too small
                    # (proper check would compute rotation matrix, but this is close enough)
                    euler = p.getEulerFromQuaternion(dish_orn)
                    tilt_angle = np.sqrt(euler[0]**2 + euler[1]**2)  # Roll and pitch
                    if tilt_angle > np.pi / 6:  # More than 30 degrees tilt
                        is_error = True
                        error_type = 'tilt'
                        break

            # Record outcome
            if is_error:
                self.errors += 1
            else:
                self.completed += 1

        # Update time (seconds)
        self.t += 1.0  # Each step = 1 second
        self.step_count += 1

        # Render new frame
        cam_start = time.time()
        frame = self._render_camera()
        cam_time = time.time() - cam_start
        self.frame_buffer.append(frame)

        # Build info dict (same structure as DishwashingEnv)
        info = {
            't': self.t,
            'completed': self.completed,
            'attempts': self.attempts,
            'errors': self.errors,
            'speed': self.current_speed,
            'care': self.current_care,
            'rate_per_min': (self.completed / max(1, self.t / 60.0))
        }

        # Episode termination
        done = self.step_count >= self.max_steps

        # Timing debug (log ~1% of steps)
        step_time = time.time() - step_start
        if np.random.rand() < 0.01:
            print(f"[DEBUG] step total={step_time:.4f}s, camera={cam_time:.4f}s")

        return self._get_video_obs(), info, done

    def _render_camera(self) -> np.ndarray:
        """
        Render RGB frame from fixed camera.

        Returns:
            frame: (H, W, 3) uint8 RGB image
        """
        # Camera parameters
        cam_pos = self.camera_config['position']
        cam_target = self.camera_config['target']
        cam_fov = self.camera_config['fov']

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=[0, 0, 1]
        )

        aspect = self.image_width / self.image_height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=cam_fov,
            aspect=aspect,
            nearVal=0.1,
            farVal=5.0
        )

        # Render
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER  # Fast CPU renderer
        )

        # Extract RGB (remove alpha channel)
        rgb = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

        return rgb

    def _get_video_obs(self) -> np.ndarray:
        """
        Get stacked video observation.

        Returns:
            video: (T, C, H, W) float32 in [0, 1]
        """
        # Stack frames: (T, H, W, 3)
        stacked = np.stack(list(self.frame_buffer), axis=0)

        # Transpose to (T, C, H, W)
        video = stacked.transpose(0, 3, 1, 2)

        # Normalize to [0, 1]
        video = video.astype(np.float32) / 255.0

        return video

    def get_state(self) -> Dict[str, Any]:
        """Get underlying state dict (for economics calculations)."""
        return {
            't': self.t,
            'completed': self.completed,
            'attempts': self.attempts,
            'errors': self.errors
        }

    def close(self):
        """Clean up PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


def create_physics_env(config: dict):
    """
    Factory function to create physics environment from config.

    Args:
        config: Dict with keys:
            - frames: int
            - image_size: [height, width]
            - max_steps: int
            - camera: dict with position, target, fov
            - headless: bool

    Returns:
        DishwashingPhysicsEnv instance
    """
    return DishwashingPhysicsEnv(
        frames=config.get('frames', 8),
        image_size=tuple(config.get('image_size', [64, 64])),
        max_steps=config.get('max_steps', 60),
        headless=config.get('headless', True),
        camera_config=config.get('camera', None)
    )
