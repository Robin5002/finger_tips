#!/usr/bin/env python3

"""
Diagnose the three main issues:
1. Robot not making full contact with cube
2. Not enough velocity 
3. Robot pushing direction perpendicular to desired location
"""

import numpy as np
from envs.finger_env import FingerPushEnv
import mujoco

def diagnose_robot_issues():
    print("🔍 DIAGNOSING ROBOT ISSUES")
    print("=" * 50)
    
    env = FingerPushEnv()
    obs, info = env.reset()
    
    # Get positions
    adr_q = int(env.jnt_qposadr[env.cube_free_jid])
    cube_pos = env.data.qpos[adr_q + 4:adr_q + 7]
    cube_xy = cube_pos[:2]
    target_xy = env.target_pos_xy
    
    print(f"📦 Cube position: [{cube_xy[0]:.3f}, {cube_xy[1]:.3f}]")
    print(f"🎯 Target position: [{target_xy[0]:.3f}, {target_xy[1]:.3f}]")
    
    # Desired direction (cube to target)
    desired_direction = target_xy - cube_xy
    desired_direction_norm = desired_direction / np.linalg.norm(desired_direction)
    print(f"🎯 Desired direction: [{desired_direction_norm[0]:.3f}, {desired_direction_norm[1]:.3f}]")
    
    # Check fingertip position
    try:
        fingertip_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
        if fingertip_site_id != -1:
            fingertip_pos = env.data.site_xpos[fingertip_site_id][:2]
            print(f"🤖 Fingertip position: [{fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f}]")
            
            # Distance from fingertip to cube
            finger_to_cube_dist = np.linalg.norm(fingertip_pos - cube_xy)
            print(f"📏 Finger-to-cube distance: {finger_to_cube_dist:.3f}m")
            
            # Robot's natural pushing direction (fingertip toward cube and beyond)
            finger_to_cube = cube_xy - fingertip_pos
            if np.linalg.norm(finger_to_cube) > 1e-6:
                robot_push_direction = finger_to_cube / np.linalg.norm(finger_to_cube)
                print(f"🤖 Robot's natural push direction: [{robot_push_direction[0]:.3f}, {robot_push_direction[1]:.3f}]")
                
                # Check alignment between robot's push direction and desired direction
                alignment = np.dot(robot_push_direction, desired_direction_norm)
                angle_diff = np.arccos(np.clip(alignment, -1, 1)) * 180 / np.pi
                
                print(f"🔄 Direction alignment: {alignment:.3f}")
                print(f"🔄 Angle difference: {angle_diff:.1f}°")
                
                if alignment < 0.5:  # Less than 60° alignment
                    print("❌ ISSUE 1: Robot pushing direction and target direction are misaligned!")
                    print("   Solution: Target should be placed in robot's natural push direction")
                
                if finger_to_cube_dist > 0.15:
                    print("❌ ISSUE 2: Robot too far from cube for good contact!")
                    print("   Solution: Improve contact rewards and approach behavior")
            
        else:
            print("⚠️  No fingertip site found - using body positions")
            
            # Try to find fingertip body
            try:
                fingertip_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "finger_lower_link")
                if fingertip_body_id != -1:
                    fingertip_pos = env.data.xpos[fingertip_body_id][:2]
                    print(f"🤖 Fingertip body position: [{fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f}]")
                    
                    # Same analysis with body position
                    finger_to_cube_dist = np.linalg.norm(fingertip_pos - cube_xy)
                    print(f"📏 Finger-to-cube distance: {finger_to_cube_dist:.3f}m")
                    
                    finger_to_cube = cube_xy - fingertip_pos
                    if np.linalg.norm(finger_to_cube) > 1e-6:
                        robot_push_direction = finger_to_cube / np.linalg.norm(finger_to_cube)
                        print(f"🤖 Robot's natural push direction: [{robot_push_direction[0]:.3f}, {robot_push_direction[1]:.3f}]")
                        
                        alignment = np.dot(robot_push_direction, desired_direction_norm)
                        angle_diff = np.arccos(np.clip(alignment, -1, 1)) * 180 / np.pi
                        
                        print(f"🔄 Direction alignment: {alignment:.3f}")
                        print(f"🔄 Angle difference: {angle_diff:.1f}°")
                        
                        if alignment < 0.5:
                            print("❌ ISSUE: Robot pushing direction and target direction are misaligned!")
                        if finger_to_cube_dist > 0.15:
                            print("❌ ISSUE: Robot too far from cube for good contact!")
            except:
                print("❌ Could not find fingertip body either")
    
    except Exception as e:
        print(f"❌ Error checking fingertip: {e}")
    
    # Test velocity issue by simulating some steps
    print(f"\n🚀 Testing Velocity Issues:")
    print("Running 10 steps with maximum action...")
    
    for step in range(10):
        # Apply maximum forward action
        action = np.array([1.0, 1.0, 1.0])[:env.num_dof]  # Max action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check cube velocity
        adr_v = int(env.jnt_dofadr[env.cube_free_jid])
        cube_linvel = env.data.qvel[adr_v + 3:adr_v + 6]
        cube_speed = np.linalg.norm(cube_linvel[:2])  # XY speed
        
        if step % 3 == 0:
            print(f"   Step {step}: Cube speed = {cube_speed:.4f} m/s")
    
    max_speed = cube_speed
    if max_speed < 0.05:
        print("❌ ISSUE 3: Cube velocity too low!")
        print("   Solution: Increase action scaling and velocity rewards")
    
    env.close()
    
    print(f"\n💡 SOLUTIONS TO IMPLEMENT:")
    print("1. 🎯 Fix target positioning to align with robot's natural push direction")
    print("2. 🤖 Improve contact rewards to encourage closer approach")  
    print("3. 🚀 Increase action scaling and velocity-based rewards")
    print("4. 🔄 Add directional alignment rewards")

if __name__ == "__main__":
    diagnose_robot_issues()
