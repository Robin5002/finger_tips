#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "cleanrl/cleanrl"))

def test_latest_model():
    """Quick test of the latest available model"""
    
    print("🔍 Setting up environment...")
    
    # Import and register environments
    import envs.register_envs
    import gymnasium as gym
    
    # Find the most recent model
    runs_dir = "/home/robin/Desktop/robotics_tutorials-fomr_24_25/rl_and_sampling_projects/rl_project/runs"
    
    model_files = []
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.endswith('.cleanrl_model'):
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                model_files.append((mtime, full_path))
    
    if not model_files:
        print("❌ No trained models found!")
        return
    
    # Get the most recent model
    model_files.sort(reverse=True)
    model_path = model_files[0][1]
    print(f"✅ Found model: {model_path}")
    
    # Create environment
    try:
        env = gym.make("FingerPush-v0", render_mode=None)
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return
    
    # Load model
    try:
        from ppo_continuous_action import Agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = Agent(env).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
        print(f"✅ Model loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test for a few episodes
    print("\n🧪 Testing model...")
    total_rewards = []
    success_count = 0
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        done = False
        while not done and step_count < 1000:  # Max 1000 steps per episode
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy().flatten()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Check distance to target
            env_data = env.unwrapped
            adr_q = int(env_data.jnt_qposadr[env_data.cube_free_jid])
            cube_pos = env_data.data.qpos[adr_q + 4:adr_q + 7]
            cube_xy = cube_pos[:2]
            distance = float(np.linalg.norm(cube_xy - env_data.target_pos_xy))
            
            if distance < 0.05:  # Success threshold
                success_count += 1
                break
        
        total_rewards.append(episode_reward)
        
        final_distance = distance
        status = "✓ SUCCESS" if distance < 0.05 else "✗ FAILED"
        print(f"Episode {episode+1}: Reward={episode_reward:6.2f}, Final Distance={final_distance:.4f}, Steps={step_count}, {status}")
    
    print(f"\n📊 Quick Results:")
    print(f"Average Reward: {np.mean(total_rewards):.3f}")
    print(f"Success Rate: {success_count}/5 episodes")
    print(f"Model: {os.path.basename(model_path)}")
    
    env.close()

if __name__ == "__main__":
    test_latest_model()
