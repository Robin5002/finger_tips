#!/usr/bin/env python3
"""
Test script to verify that the reward structure encourages Y-direction pushing.
This script will run a few episodes and check if rewards are properly shaped for Y-direction movement.
"""

import sys
import os
import numpy as np

# Add paths to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)

import envs.register_envs
import gymnasium as gym

def test_y_direction_rewards():
    print("🧪 TESTING Y-DIRECTION REWARD STRUCTURE")
    print("=" * 50)
    
    # Create environment
    env = gym.make("FingerPush-v0", render_mode=None)
    
    # Test 5 episodes to see reward structure
    for episode in range(3):
        print(f"\n📍 Episode {episode + 1}")
        obs, _ = env.reset()
        
        total_rewards = {}
        episode_reward = 0
        
        # Track cube position during episode
        cube_y_positions = []
        
        for step in range(100):  # Short episode for testing
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get cube position
            adr_q = int(env.jnt_qposadr[env.cube_free_jid])
            cube_pos = env.data.qpos[adr_q + 4:adr_q + 7]
            cube_y_positions.append(cube_pos[1])
            
            # Accumulate reward components
            reward_dict = info.get("reward_dict", {})
            for key, value in reward_dict.items():
                if key not in total_rewards:
                    total_rewards[key] = 0
                total_rewards[key] += value
            
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Analyze results
        print(f"   📊 Total episode reward: {episode_reward:.2f}")
        print(f"   📈 Cube Y movement: {cube_y_positions[-1] - cube_y_positions[0]:.4f}m")
        print(f"   🎯 Target Y position: {env.target_pos_xy[1]:.3f}")
        print(f"   📐 Final distance to target: {np.linalg.norm(env.data.qpos[adr_q + 4:adr_q + 6] - env.target_pos_xy):.4f}m")
        
        print(f"   🏆 Reward breakdown:")
        for key, value in total_rewards.items():
            print(f"      • {key}: {value:.2f}")
    
    env.close()
    print("\n✅ Y-direction reward test completed!")

def test_manual_y_movement():
    """Test what happens when we manually move cube in Y direction"""
    print("\n🔧 TESTING MANUAL Y-DIRECTION MOVEMENT")
    print("=" * 50)
    
    env = gym.make("FingerPush-v0", render_mode=None)
    obs, _ = env.reset()
    
    # Get initial positions
    adr_q = int(env.jnt_qposadr[env.cube_free_jid])
    initial_cube_pos = env.data.qpos[adr_q + 4:adr_q + 7].copy()
    
    print(f"   📍 Initial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}]")
    print(f"   🎯 Target position: [{env.target_pos_xy[0]:.3f}, {env.target_pos_xy[1]:.3f}]")
    
    # Manually move cube towards target in Y direction
    steps_y_movement = []
    for i in range(10):
        # Move cube 1cm towards target in Y direction
        new_y = initial_cube_pos[1] + (i + 1) * 0.01  # 1cm increments
        env.data.qpos[adr_q + 5] = new_y  # Set Y position
        
        # Set cube velocity to simulate movement
        adr_v = int(env.jnt_dofadr[env.cube_free_jid])
        env.data.qvel[adr_v + 4] = 0.05  # Positive Y velocity
        
        # Step environment with zero action
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        
        reward_dict = info.get("reward_dict", {})
        
        print(f"   Step {i+1}: Y={new_y:.3f}, Reward={reward:.2f}")
        print(f"      • y_velocity: {reward_dict.get('y_velocity', 0):.2f}")
        print(f"      • y_direction_progress: {reward_dict.get('y_direction_progress', 0):.2f}")
        print(f"      • distance: {reward_dict.get('distance', 0):.2f}")
        
        steps_y_movement.append(reward)
    
    print(f"   📈 Reward trend: {steps_y_movement[:3]} -> {steps_y_movement[-3:]}")
    print("   ✅ Rewards should increase as cube moves toward Y target!")
    
    env.close()

if __name__ == "__main__":
    test_y_direction_rewards()
    test_manual_y_movement()
