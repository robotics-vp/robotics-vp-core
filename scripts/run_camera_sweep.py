#!/usr/bin/env python3
"""
Run a deterministic camera sweep in the simulator.
Useful for generating demo videos or verifying rendering consistency.
"""
import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pybullet
    import pybullet_data
except ImportError:
    print("PyBullet not installed. Skipping camera sweep.")
    sys.exit(0)

def run_sweep(args):
    print(f"[CameraSweep] Starting sweep for task: {args.task_id}")
    
    # Setup PyBullet
    client = pybullet.connect(pybullet.GUI if args.render else pybullet.DIRECT)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setGravity(0, 0, -9.8)
    
    # Load plane and robot (stub)
    plane = pybullet.loadURDF("plane.urdf")
    # In a real scenario, we'd load the robot and task environment here
    # For this script, we just visualize a simple scene
    cube = pybullet.loadURDF("cube.urdf", [0, 0, 0.5])
    
    # Camera settings
    width, height = 640, 480
    fov = 60
    aspect = width / height
    near = 0.1
    far = 100
    
    # Sweep parameters
    radius = 2.0
    height_offset = 1.0
    num_frames = args.frames
    
    print(f"[CameraSweep] Generating {num_frames} frames...")
    
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = height_offset
        
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[0, 0, 0.5],
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = pybullet.computeProjectionMatrixFOV(
            fov, aspect, near, far
        )
        
        # Render
        width, height, rgb, depth, seg = pybullet.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        
        # In a real script, we would save 'rgb' to disk
        # time.sleep(0.01)
        
    print("[CameraSweep] Sweep complete.")
    pybullet.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, default="drawer_open")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="Show GUI")
    args = parser.parse_args()
    run_sweep(args)
