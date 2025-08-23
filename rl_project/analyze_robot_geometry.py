#!/usr/bin/env python3
"""
Analyze the robot's geometry to understand its natural pushing axis.
This will help us position the cube optimally.
"""

import numpy as np
import mujoco
from envs.finger_env import FingerPushEnv

def analyze_robot_geometry():
    """Analyze robot fingertip position and orientation to determine natural pushing direction"""
    print("=== Robot Geometry Analysis ===")
    
    # Create environment
    env = FingerPushEnv(render_mode=None)
    
    # Reset to get consistent starting position
    env.reset()
    
    # Get robot joint positions at default configuration
    print(f"Default joint positions: {env.default_joint_pos}")
    
    # Set robot to default position
    for i, jid in enumerate(env.actuated_joint_ids):
        env.data.qpos[int(env.jnt_qposadr[jid])] = env.default_joint_pos[i]
    
    # Step simulation to update positions
    mujoco.mj_forward(env.model, env.data)
    
    # Get fingertip position
    fingertip_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
    if fingertip_site_id != -1:
        fingertip_pos = env.data.site_xpos[fingertip_site_id]
        print(f"Fingertip position (default config): {fingertip_pos}")
        print(f"Fingertip XY: {fingertip_pos[:2]}")
    
    # Test different joint configurations to see fingertip movement patterns
    print("\n=== Joint Configuration Analysis ===")
    
    # Test variations in each joint
    joint_variations = [
        [0.0, 0.5, -0.75],  # Default
        [0.2, 0.5, -0.75],  # Joint 1 positive
        [-0.2, 0.5, -0.75], # Joint 1 negative
        [0.0, 0.7, -0.75],  # Joint 2 positive
        [0.0, 0.3, -0.75],  # Joint 2 negative
        [0.0, 0.5, -0.5],   # Joint 3 positive
        [0.0, 0.5, -1.0],   # Joint 3 negative
    ]
    
    names = ["Default", "J1+", "J1-", "J2+", "J2-", "J3+", "J3-"]
    
    fingertip_positions = []
    
    for i, joint_config in enumerate(joint_variations):
        # Set joints
        for j, jid in enumerate(env.actuated_joint_ids):
            if j < len(joint_config):
                env.data.qpos[int(env.jnt_qposadr[jid])] = joint_config[j]
        
        # Update simulation
        mujoco.mj_forward(env.model, env.data)
        
        # Get fingertip position
        if fingertip_site_id != -1:
            fingertip_pos = env.data.site_xpos[fingertip_site_id]
            fingertip_positions.append(fingertip_pos[:2].copy())
            print(f"{names[i]}: Fingertip XY = {fingertip_pos[:2]}")
    
    # Analyze movement patterns
    print("\n=== Movement Pattern Analysis ===")
    default_pos = fingertip_positions[0]
    
    for i in range(1, len(fingertip_positions)):
        movement = fingertip_positions[i] - default_pos
        distance = np.linalg.norm(movement)
        if distance > 0.001:  # Only show significant movements
            direction = movement / distance
            print(f"{names[i]} movement: {movement} (distance: {distance:.4f}, direction: {direction})")
    
    # Test robot reaching towards different positions
    print("\n=== Robot Reaching Analysis ===")
    
    # Get current cube position
    adr_q = int(env.jnt_qposadr[env.cube_free_jid])
    cube_pos = env.data.qpos[adr_q + 4:adr_q + 7]
    cube_xy = cube_pos[:2]
    print(f"Current cube position: {cube_xy}")
    
    # Calculate which direction the robot would naturally push
    default_fingertip = fingertip_positions[0]
    fingertip_to_cube = cube_xy - default_fingertip
    distance = np.linalg.norm(fingertip_to_cube)
    
    if distance > 0:
        natural_push_direction = fingertip_to_cube / distance
        print(f"Fingertip to cube vector: {fingertip_to_cube}")
        print(f"Natural push direction: {natural_push_direction}")
        
        # Suggest optimal cube position
        optimal_distance = 0.05  # 5cm from fingertip
        optimal_cube_pos = default_fingertip + natural_push_direction * optimal_distance
        print(f"Suggested optimal cube position: {optimal_cube_pos}")
        
        # Calculate target based on push direction
        push_distance = 0.06  # 6cm push distance (same as current target)
        suggested_target = optimal_cube_pos + natural_push_direction * push_distance
        print(f"Suggested target position: {suggested_target}")
    
    env.close()
    
    return {
        'default_fingertip': default_pos,
        'cube_position': cube_xy,
        'natural_push_direction': natural_push_direction if distance > 0 else None,
        'optimal_cube_pos': optimal_cube_pos if distance > 0 else None,
        'suggested_target': suggested_target if distance > 0 else None
    }

if __name__ == "__main__":
    results = analyze_robot_geometry()
    
    print("\n=== SUMMARY ===")
    print(f"Default fingertip position: {results['default_fingertip']}")
    print(f"Current cube position: {results['cube_position']}")
    if results['natural_push_direction'] is not None:
        print(f"Natural push direction: {results['natural_push_direction']}")
        print(f"Optimal cube position: {results['optimal_cube_pos']}")
        print(f"Suggested target: {results['suggested_target']}")
