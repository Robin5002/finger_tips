#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # Add project root
sys.path.insert(0, os.path.join(project_root, "cleanrl/cleanrl"))  # Add cleanrl

import gymnasium as gym
from cleanrl.cleanrl.ppo_continuous_action import Agent
import json

def load_hyperparameters():
    """Load hyperparameters from best_params.json"""
    params_path = "/home/robin/Desktop/robotics_tutorials-fomr_24_25/rl_and_sampling_projects/rl_project/scripts/best_params.json"
    with open(params_path, 'r') as f:
        return json.load(f)

def test_trained_model(model_path, num_episodes=10):
    """Test the trained PPO model and analyze performance"""
    
    # Register environments - import the module which automatically registers
    import envs.register_envs
    
    # Create environment
    env = gym.make("FingerPush-v0", render_mode=None)
    
    # Load hyperparameters to get the correct network architecture
    params = load_hyperparameters()
    
    # Initialize agent with same architecture as training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    
    # Try to load the model
    try:
        if os.path.exists(model_path):
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    agent.eval()
    
    # Test the agent
    results = {
        'episodes': [],
        'success_rate': 0,
        'avg_reward': 0,
        'avg_distance_to_target': 0,
        'avg_episode_length': 0
    }
    
    total_rewards = []
    successful_episodes = 0
    final_distances = []
    episode_lengths = []
    
    print(f"\n🧪 Testing trained model for {num_episodes} episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_distances = []
        
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy().flatten()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Track distance to target
            reward_dict = info.get('reward_dict', {})
            
            # Calculate current distance (from environment state)
            # Get cube position and target position to calculate distance
            env_data = env.unwrapped
            adr_q = int(env_data.jnt_qposadr[env_data.cube_free_jid])
            cube_pos = env_data.data.qpos[adr_q + 4:adr_q + 7]
            cube_xy = cube_pos[:2]
            distance = float(np.linalg.norm(cube_xy - env_data.target_pos_xy))
            episode_distances.append(distance)
            
            done = terminated or truncated
        
        final_distance = episode_distances[-1] if episode_distances else float('inf')
        success = final_distance < 0.05  # Success threshold from environment
        
        total_rewards.append(episode_reward)
        final_distances.append(final_distance)
        episode_lengths.append(episode_length)
        
        if success:
            successful_episodes += 1
        
        # Store episode data
        results['episodes'].append({
            'episode': episode,
            'reward': episode_reward,
            'final_distance': final_distance,
            'success': success,
            'length': episode_length,
            'min_distance': min(episode_distances) if episode_distances else float('inf')
        })
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Episode {episode+1:2d}: Reward={episode_reward:6.2f}, Distance={final_distance:.4f}, {status}")
    
    # Calculate statistics
    results['success_rate'] = successful_episodes / num_episodes
    results['avg_reward'] = np.mean(total_rewards)
    results['avg_distance_to_target'] = np.mean(final_distances)
    results['avg_episode_length'] = np.mean(episode_lengths)
    
    # Print summary
    print("=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Success Rate:           {results['success_rate']*100:.1f}% ({successful_episodes}/{num_episodes})")
    print(f"Average Reward:         {results['avg_reward']:.3f}")
    print(f"Average Final Distance: {results['avg_distance_to_target']:.4f}")
    print(f"Average Episode Length: {results['avg_episode_length']:.1f}")
    print(f"Best Final Distance:    {min(final_distances):.4f}")
    print(f"Worst Final Distance:   {max(final_distances):.4f}")
    
    env.close()
    return results

def find_model_file():
    """Find the trained model file"""
    # Look for the most recent .cleanrl_model file in runs directory
    runs_dir = "/home/robin/Desktop/robotics_tutorials-fomr_24_25/rl_and_sampling_projects/rl_project/runs"
    
    if os.path.exists(runs_dir):
        # Find all .cleanrl_model files and get the most recent one
        model_files = []
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                if file.endswith('.cleanrl_model'):
                    full_path = os.path.join(root, file)
                    mtime = os.path.getmtime(full_path)
                    model_files.append((mtime, full_path))
        
        if model_files:
            # Sort by modification time and return the most recent
            model_files.sort(reverse=True)
            return model_files[0][1]
    
    # Fallback: look in scripts directory
    scripts_dir = "/home/robin/Desktop/robotics_tutorials-fomr_24_25/rl_and_sampling_projects/rl_project/scripts"
    
    # Common patterns for saved models
    possible_names = [
        "finger_push_training_optimized.pth",
        "model.pth",
        "agent.pth",
        "ppo_model.pth",
        "best_model.pth"
    ]
    
    for name in possible_names:
        path = os.path.join(scripts_dir, name)
        if os.path.exists(path):
            return path
    
    # Look for any .pth files in the directory
    for file in os.listdir(scripts_dir):
        if file.endswith('.pth'):
            return os.path.join(scripts_dir, file)
    
    return None

if __name__ == "__main__":
    # Find the model file
    model_path = find_model_file()
    
    if model_path is None:
        print("❌ No trained model found. Please ensure you have a .pth file in the scripts directory.")
        print("\nLooking for files in scripts directory:")
        scripts_dir = "/home/robin/Desktop/robotics_tutorials-fomr_24_25/rl_and_sampling_projects/rl_project/scripts"
        files = [f for f in os.listdir(scripts_dir) if f.endswith(('.pth', '.pt'))]
        if files:
            print("Found these model files:")
            for f in files:
                print(f"  - {f}")
        else:
            print("No .pth or .pt files found.")
        sys.exit(1)
    
    print(f"🤖 Testing model: {os.path.basename(model_path)}")
    
    # Test the model
    results = test_trained_model(model_path, num_episodes=20)
    
    if results:
        # Save results
        results_path = os.path.join(os.path.dirname(model_path), "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_path}")
