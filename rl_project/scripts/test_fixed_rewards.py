#!/usr/bin/env python3

"""
Test the FIXED reward function to ensure it gives reasonable reward scales
"""

import os
import sys
import numpy as np
import gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Register the FIXED environment
from envs.fixed_finger_env import FixedFingerPushEnv
gym.register(
    id="FixedFingerPush-v0",
    entry_point="envs.fixed_finger_env:FixedFingerPushEnv",
    max_episode_steps=2400
)

def test_fixed_reward_function():
    """Test the fixed reward function to ensure it gives reasonable rewards"""
    print("🧪 TESTING FIXED REWARD FUNCTION")
    print("=" * 50)
    
    # Create environment
    env = gym.make("FixedFingerPush-v0", render_mode=None)
    
    print("📊 Environment Analysis:")
    print(f"   Max episode steps: {env.spec.max_episode_steps}")
    
    # Calculate theoretical rewards with FIXED function
    max_episode_steps = env.spec.max_episode_steps
    
    print(f"\n💰 FIXED Reward Analysis (per {max_episode_steps} steps):")
    print("   Contact reward: 0-1.0 per step (finger touching cube)")
    print(f"   - Total if always in contact: {1.0 * max_episode_steps:.0f}")
    
    print("   Progress reward: up to 10.0 * progress per step (only when touching)")
    print("   - Highly variable, depends on actual pushing")
    
    print("   Distance reward: 0-0.7 per step (small, for final positioning)")
    print(f"   - Total for episode: {0.7 * max_episode_steps:.0f}")
    
    print("   Success reward: 2.0 per step when within 3cm of target")
    print(f"   - Total if always successful: {2.0 * max_episode_steps:.0f}")
    
    print("   Effort penalty: ~-0.001 per step")
    print(f"   - Total effort penalty: {-0.001 * max_episode_steps:.1f}")
    
    print("   Stagnation penalty: -0.05 per step when not touching cube")
    print(f"   - Total if never touching: {-0.05 * max_episode_steps:.0f}")
    
    # Realistic maximum
    realistic_max = (1.0 + 0.7 + 2.0) * max_episode_steps  # Contact + distance + success
    print(f"\n🎯 REALISTIC MAXIMUM REWARD: ~{realistic_max:.0f}")
    print(f"🎯 This is {realistic_max/155976:.1%} of the old maximum")
    
    # Expected training reward
    expected_training = 1.0 * max_episode_steps  # Moderate contact/distance reward
    print(f"🎯 EXPECTED TRAINING REWARD: ~{expected_training:.0f}")
    print(f"🎯 Average per step: {expected_training/max_episode_steps:.3f}")

def run_sample_episode():
    """Run a sample episode with random actions to see reward breakdown"""
    print(f"\n🎮 SAMPLE EPISODE WITH FIXED REWARDS:")
    print("-" * 40)
    
    env = gym.make("FixedFingerPush-v0", render_mode=None)
    obs, _ = env.reset()
    
    episode_rewards = {
        "contact": 0.0,
        "progress": 0.0,
        "distance": 0.0,
        "success": 0.0,
        "effort": 0.0,
        "stagnation": 0.0,
        "total": 0.0
    }
    
    step_count = 0
    distances = []
    contact_count = 0
    success_count = 0
    
    # Get initial state info
    cube_pos_start_idx = 2 * env.unwrapped.num_dof
    initial_cube_pos = obs[cube_pos_start_idx:cube_pos_start_idx + 3]
    target_pos = obs[-env.unwrapped.num_dof - 2:-env.unwrapped.num_dof]
    initial_distance = np.linalg.norm(initial_cube_pos[:2] - target_pos)
    
    print(f"   Initial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}]")
    print(f"   Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
    print(f"   Initial distance: {initial_distance:.3f}m")
    
    while True:
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Track reward components
        if "reward_dict" in info:
            for key in episode_rewards:
                if key in info["reward_dict"]:
                    episode_rewards[key] += info["reward_dict"][key]
                    
            # Count contacts and successes
            if info["reward_dict"]["contact"] > 0:
                contact_count += 1
            if info["reward_dict"]["success"] > 0:
                success_count += 1
                
        episode_rewards["total"] += reward
        
        # Track distance
        cube_pos_start_idx = 2 * env.unwrapped.num_dof
        current_cube_pos = next_obs[cube_pos_start_idx:cube_pos_start_idx + 3]
        target_pos = next_obs[-env.unwrapped.num_dof - 2:-env.unwrapped.num_dof]
        current_distance = np.linalg.norm(current_cube_pos[:2] - target_pos)
        distances.append(current_distance)
        
        step_count += 1
        obs = next_obs
        
        if terminated or truncated:
            break
    
    # Analysis
    final_distance = distances[-1]
    avg_distance = np.mean(distances)
    min_distance = min(distances)
    
    print(f"   Episode length: {step_count} steps")
    print(f"   Final distance: {final_distance:.3f}m")
    print(f"   Average distance: {avg_distance:.3f}m")
    print(f"   Minimum distance: {min_distance:.3f}m")
    print(f"   Contact steps: {contact_count}/{step_count} ({100*contact_count/step_count:.1f}%)")
    print(f"   Success steps: {success_count}/{step_count} ({100*success_count/step_count:.1f}%)")
    
    print(f"\n   Fixed reward breakdown:")
    for key, value in episode_rewards.items():
        print(f"      {key:15}: {value:8.2f} ({value/step_count:6.3f} per step)")
    
    print(f"\n   🎯 COMPARISON:")
    print(f"      Old max per step: ~65.0")
    print(f"      Fixed max per step: ~4.0")
    print(f"      Actual per step: {episode_rewards['total']/step_count:.3f}")
    print(f"      Reduction factor: ~{65.0 / 4.0:.1f}x")
    
    env.close()

if __name__ == "__main__":
    test_fixed_reward_function()
    run_sample_episode()
