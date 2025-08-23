#!/usr/bin/env python3

"""
Test the proper pushing environment to ensure it's working correctly
"""

import os
import sys
import numpy as np
import gymnasium as gym

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Register environments
import envs.register_envs

def test_proper_pushing_env():
    """Test the proper pushing environment"""
    print("🧪 TESTING PROPER PUSHING ENVIRONMENT")
    print("=" * 50)
    
    # Create environment
    env = gym.make("FingerPush-v0", render_mode=None)
    print("✅ Environment created successfully")
    
    # Test episode
    obs, _ = env.reset()
    print(f"✅ Reset successful, observation shape: {obs.shape}")
    
    print("\n📊 ENVIRONMENT SETUP:")
    print(f"   📦 Cube initial position: {env.unwrapped.cube_init_pos}")
    print(f"   🎯 Target position: {env.unwrapped.target_pos_xy}")
    push_distance = np.linalg.norm(env.unwrapped.target_pos_xy - env.unwrapped.cube_init_pos)
    print(f"   📏 Required push distance: {push_distance:.3f}m")
    print(f"   ⏱️  Max episode steps: {env.unwrapped.steps_per_episode}")
    
    # Test a few random actions
    print("\n🎮 TESTING RANDOM ACTIONS:")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"   Step {step+1}: reward = {reward:.3f}")
        if "reward_dict" in info:
            for component, value in info["reward_dict"].items():
                if abs(value) > 0.001:  # Only show non-zero components
                    print(f"      {component}: {value:.3f}")
    
    print(f"\n💰 Total reward from 5 random steps: {total_reward:.3f}")
    print(f"   Average reward per step: {total_reward/5:.3f}")
    
    # Check reward scale
    print("\n🔍 REWARD SCALE ANALYSIS:")
    print("   Expected max reward per step: ~6.0")
    print("   Expected random baseline: ~0.0")
    
    env.close()
    print("\n✅ Environment test completed successfully!")
    
    return True

if __name__ == "__main__":
    test_proper_pushing_env()
