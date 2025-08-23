#!/usr/bin/env python3

"""
Simple diagnostic script to understand the reward structure
by examining what rewards are possible in the FingerPush environment
"""

import os
import sys
import numpy as np
import gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Register the environment
from envs.finger_env import FingerPushEnv
gym.register(
    id="FingerPush-v0",
    entry_point="envs.finger_env:FingerPushEnv",
    max_episode_steps=2400
)

def analyze_reward_structure():
    """Analyze the reward structure to understand the ~2388 reward mystery"""
    print("🔍 ANALYZING REWARD STRUCTURE")
    print("=" * 50)
    
    # Create environment
    env = gym.make("FingerPush-v0", render_mode=None)
    
    print("📊 Environment Analysis:")
    print(f"   Max episode steps: {env.spec.max_episode_steps}")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Calculate theoretical maximum rewards
    max_episode_steps = env.spec.max_episode_steps  # Should be 2400
    
    print(f"\n💰 Theoretical Reward Analysis (per {max_episode_steps} steps):")
    print("   Distance reward: (max_dist - dist_to_target) / max_dist * 10.0")
    print("   - Best case: 10.0 per step (cube at target)")
    print(f"   - Total for episode: {10.0 * max_episode_steps:.0f}")
    
    print("   Success bonus: 50.0 per step when within 5cm of target")
    print(f"   - Total if always successful: {50.0 * max_episode_steps:.0f}")
    
    print("   Contact reward: 0-5.0 per step (finger touching cube)")
    print(f"   - Total if always in contact: {5.0 * max_episode_steps:.0f}")
    
    print("   Progress reward: 20.0 * progress per step")
    print("   - Variable, depends on movement")
    
    print("   Penalties:")
    print("   - Action penalty: -0.01 * sum(action²) per step")
    print("   - Time penalty: -0.01 per step")
    print(f"   - Total time penalty: {-0.01 * max_episode_steps:.0f}")
    
    theoretical_max = (10.0 + 50.0 + 5.0) * max_episode_steps - 0.01 * max_episode_steps
    print(f"\n🎯 THEORETICAL MAXIMUM REWARD: ~{theoretical_max:.0f}")
    print(f"🎯 OBSERVED TRAINING REWARD: ~2388")
    print(f"🎯 RATIO: {2388/theoretical_max:.1%} of theoretical maximum")
    
    # Now let's see what reward ~2388 implies
    print(f"\n🔍 REVERSE ENGINEERING ~2388 REWARD:")
    target_reward = 2388
    per_step_reward = target_reward / max_episode_steps
    print(f"   Average reward per step: {per_step_reward:.3f}")
    
    # If we assume minimal penalties and no contact/progress
    # reward ≈ distance_reward + success_bonus - small_penalties
    # 2388 / 2400 = 0.995 per step
    # This suggests: distance_reward + success_bonus ≈ 1.0 per step
    
    # Case 1: Mostly distance reward, little success
    distance_contribution = 1.0  # Close to max distance reward
    success_contribution = 0.0   # No success bonus
    print(f"\n   Scenario 1 - Cube always near target (no success bonus):")
    print(f"   - Distance reward: ~{distance_contribution * 10:.1f} * {max_episode_steps} = {distance_contribution * 10 * max_episode_steps:.0f}")
    print(f"   - Success bonus: ~{success_contribution * 50:.1f} * {max_episode_steps} = {success_contribution * 50 * max_episode_steps:.0f}")
    print(f"   - Total: ~{(distance_contribution * 10 + success_contribution * 50) * max_episode_steps:.0f}")
    
    # Case 2: Mix of distance and some success
    distance_contribution = 0.8  # Good distance reward
    success_contribution = 0.1   # Some success steps
    print(f"\n   Scenario 2 - Good distance, some success:")
    print(f"   - Distance reward: ~{distance_contribution * 10:.1f} * {max_episode_steps} = {distance_contribution * 10 * max_episode_steps:.0f}")
    print(f"   - Success bonus: ~{success_contribution * 50:.1f} * {max_episode_steps} = {success_contribution * 50 * max_episode_steps:.0f}")
    print(f"   - Total: ~{(distance_contribution * 10 + success_contribution * 50) * max_episode_steps:.0f}")
    
    print(f"\n🎯 LIKELY EXPLANATION:")
    print(f"   The agent gets ~1.0 reward per step by keeping the cube close to target")
    print(f"   This is mainly from distance reward (8-10 per step) averaged over episode")
    print(f"   The cube might oscillate around target or stay moderately close")
    print(f"   Success bonus is either rare or the agent stays just outside success threshold")

def run_sample_episode():
    """Run a sample episode with random actions to see reward breakdown"""
    print(f"\n🎮 SAMPLE RANDOM EPISODE:")
    print("-" * 30)
    
    env = gym.make("FingerPush-v0", render_mode=None)
    obs, _ = env.reset()
    
    episode_rewards = {
        "distance": 0.0,
        "success": 0.0,
        "contact": 0.0,
        "progress": 0.0,
        "action_penalty": 0.0,
        "time_penalty": 0.0,
        "total": 0.0
    }
    
    step_count = 0
    distances = []
    
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
    
    print(f"\n   Reward breakdown:")
    for key, value in episode_rewards.items():
        print(f"      {key:15}: {value:8.2f} ({value/step_count:6.3f} per step)")
    
    env.close()

if __name__ == "__main__":
    analyze_reward_structure()
    run_sample_episode()
