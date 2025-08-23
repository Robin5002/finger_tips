#!/usr/bin/env python3

"""
Diagnostic script to understand what the PPO agent is actually doing
and why it's getting ~2388 reward without successful cube pushing
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Register the environment
from envs.finger_env import FingerPushEnv
gym.register(
    id="FingerPush-v0",
    entry_point="envs.finger_env:FingerPushEnv",
    max_episode_steps=2400
)

# Import the PPO agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cleanrl'))
from cleanrl.ppo_continuous_action import Agent

def load_trained_model():
    """Load the trained PPO model"""
    model_path = "runs/FingerPush-v0__ppo_continuous_action__1__1755521249/ppo_continuous_action.cleanrl_model"
    
    # Create environment to get dimensions
    env = gym.make("FingerPush-v0", render_mode=None)
    
    # Create agent
    class MockEnvs:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    mock_envs = MockEnvs(env)
    agent = Agent(mock_envs)
    
    # Load the model weights
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()
    
    env.close()
    return agent

def analyze_agent_behavior(num_episodes=5):
    """Analyze what the agent is actually doing"""
    print("🔍 DIAGNOSING REWARD ISSUE")
    print("=" * 50)
    
    # Load model
    agent = load_trained_model()
    print("✅ Loaded trained model")
    
    # Create environment for analysis
    env = gym.make("FingerPush-v0", render_mode=None)
    
    for episode in range(num_episodes):
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
        
        # Track positions
        initial_cube_pos = None
        final_cube_pos = None
        distances_to_target = []
        
        print(f"\n📊 EPISODE {episode + 1}")
        print("-" * 30)
        
        step_count = 0
        while True:
            # Get agent action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Track cube position
            if step_count == 0:
                # Extract cube position from observation
                # Obs structure: [joint_pos, joint_vel, cube_pos(3), cube_quat(4), cube_linvel(3), cube_angvel(3), target_pos(2), prev_action]
                cube_pos_start_idx = 2 * env.unwrapped.num_dof  # After joint pos and vel
                initial_cube_pos = next_obs[cube_pos_start_idx:cube_pos_start_idx + 3]
                target_pos = next_obs[-env.unwrapped.num_dof - 2:-env.unwrapped.num_dof]  # target_xy before prev_action
                
                print(f"   🎯 Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
                print(f"   📦 Initial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
                
                initial_distance = np.linalg.norm(initial_cube_pos[:2] - target_pos)
                print(f"   📏 Initial distance to target: {initial_distance:.3f}m")
            
            # Track distance to target
            cube_pos_start_idx = 2 * env.unwrapped.num_dof
            current_cube_pos = next_obs[cube_pos_start_idx:cube_pos_start_idx + 3]
            target_pos = next_obs[-env.unwrapped.num_dof - 2:-env.unwrapped.num_dof]
            current_distance = np.linalg.norm(current_cube_pos[:2] - target_pos)
            distances_to_target.append(current_distance)
            
            # Accumulate reward components
            if "reward_dict" in info:
                for key in episode_rewards:
                    if key in info["reward_dict"]:
                        episode_rewards[key] += info["reward_dict"][key]
            episode_rewards["total"] += reward
            
            step_count += 1
            obs = next_obs
            
            if terminated or truncated:
                final_cube_pos = current_cube_pos
                break
        
        # Analysis
        final_distance = np.linalg.norm(final_cube_pos[:2] - target_pos)
        distance_moved = np.linalg.norm(final_cube_pos[:2] - initial_cube_pos[:2])
        min_distance = min(distances_to_target)
        
        print(f"   📦 Final cube position: [{final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f}]")
        print(f"   📏 Final distance to target: {final_distance:.3f}m")
        print(f"   📏 Minimum distance reached: {min_distance:.3f}m")
        print(f"   🏃 Total cube movement: {distance_moved:.3f}m")
        print(f"   ⏱️  Episode length: {step_count} steps")
        
        print(f"\n   💰 REWARD BREAKDOWN:")
        for key, value in episode_rewards.items():
            print(f"      {key:15}: {value:8.2f}")
        
        print(f"\n   📈 REWARD PER STEP:")
        for key, value in episode_rewards.items():
            if step_count > 0:
                print(f"      {key:15}: {value/step_count:8.4f}")
        
        # Success analysis
        success_threshold = 0.05  # 5cm as defined in environment
        if final_distance < success_threshold:
            print(f"   ✅ SUCCESS: Cube reached target (< {success_threshold}m)")
        else:
            print(f"   ❌ FAILED: Cube did not reach target (> {success_threshold}m)")
        
        if min_distance < success_threshold:
            print(f"   🎯 Cube did get close to target at some point")
        else:
            print(f"   💔 Cube never got close to target")
    
    env.close()
    
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS COMPLETE")
    print("\nBased on this analysis, the issue likely is:")
    print("1. The agent gets consistent distance rewards without actually pushing")
    print("2. The reward function may be too generous for 'passive' behaviors")
    print("3. The agent may have learned to stay in a position that maximizes distance reward")
    print("4. Contact and progress rewards might not be sufficient to encourage actual pushing")

if __name__ == "__main__":
    analyze_agent_behavior()
