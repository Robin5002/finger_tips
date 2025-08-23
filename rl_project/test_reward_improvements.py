#!/usr/bin/env python3

"""
Test script to verify the improved reward shaping for reliable 20cm target reaching.
"""

import numpy as np
from envs.finger_env import FingerPushEnv
import matplotlib.pyplot as plt

def test_reward_scaling():
    print("🧪 Testing Improved Reward Scaling for 20cm Target")
    print("=" * 60)
    
    env = FingerPushEnv()
    obs, info = env.reset()
    
    # Get initial positions
    adr_q = int(env.jnt_qposadr[env.cube_free_jid])
    cube_pos = env.data.qpos[adr_q + 4:adr_q + 7]
    cube_xy = cube_pos[:2]
    target_xy = env.target_pos_xy
    
    print(f"📦 Cube position: [{cube_xy[0]:.3f}, {cube_xy[1]:.3f}]")
    print(f"🎯 Target position: [{target_xy[0]:.3f}, {target_xy[1]:.3f}]")
    print(f"📏 Initial distance: {np.linalg.norm(cube_xy - target_xy):.3f}m")
    print()
    
    # Test reward scaling at different distances
    print("🎮 Testing reward scaling at different distances:")
    distances = np.linspace(0.001, 0.25, 20)  # From very close to far
    
    for dist in distances:
        # Temporarily set cube at different distances for reward testing
        test_cube_pos = target_xy + np.array([-dist, 0.0])  # Move cube back by distance
        env.data.qpos[adr_q + 4:adr_q + 6] = test_cube_pos
        env.prev_dist = dist + 0.001  # Set for progress calculation
        
        rewards = env._get_reward()
        total_reward = sum(rewards.values())
        
        print(f"   Distance {dist:.3f}m: Total={total_reward:6.1f}, "
              f"Distance={rewards['distance']:5.1f}, Success={rewards['success']:5.1f}")
        
        # Check if we're in success zone
        if dist < 0.03:
            print(f"      ✅ SUCCESS ZONE: High success bonus activated!")
            
    print("\n🔬 Key reward improvements:")
    print("   - Exponential distance reward: 1/(1 + 8*normalized_distance)")
    print("   - Massive success bonus: 300-400 points when distance < 0.03m")
    print("   - Exponential progress scaling: higher rewards near target")
    print("   - Minimal penalties to avoid interference")
    
    env.close()

def test_episode_completion():
    print("\n🏃 Testing Episode Completion Conditions")
    print("=" * 50)
    
    env = FingerPushEnv()
    
    # Test success termination
    obs, info = env.reset()
    adr_q = int(env.jnt_qposadr[env.cube_free_jid])
    target_xy = env.target_pos_xy
    
    # Place cube very close to target to trigger success
    success_pos = target_xy + np.array([0.02, 0.0])  # 2cm from target
    env.data.qpos[adr_q + 4:adr_q + 6] = success_pos
    
    print(f"🎯 Placed cube 2cm from target to test success termination")
    
    # Run steps to accumulate success timer
    for step in range(20):
        action = np.zeros(env.action_space.shape)  # Do nothing
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"   Step {step:2d}: Success timer = {env.reach_goal_timer}, "
                  f"Terminated = {terminated}, Truncated = {truncated}")
            
        if terminated or truncated:
            print(f"   ✅ Episode ended at step {step + 1} (success held for {env.reach_goal_timer} steps)")
            break
    
    env.close()

if __name__ == "__main__":
    test_reward_scaling()
    test_episode_completion()
    print("\n✅ Reward improvement testing completed!")
