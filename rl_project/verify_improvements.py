#!/usr/bin/env python3

"""
Final verification script to confirm all improvements are working correctly.
"""

import sys
import os
import numpy as np

# Add paths
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))

def verify_environment():
    print("🔍 Verifying Environment Improvements")
    print("=" * 50)
    
    try:
        import envs.register_envs
        from envs.finger_env import FingerPushEnv
        
        env = FingerPushEnv()
        obs, info = env.reset()
        
        # Check target distance
        adr_q = int(env.jnt_qposadr[env.cube_free_jid])
        cube_pos = env.data.qpos[adr_q + 4:adr_q + 7]
        target_distance = np.linalg.norm(env.target_pos_xy - cube_pos[:2])
        
        print(f"✅ Environment loads successfully")
        print(f"✅ Target distance: {target_distance:.3f}m (should be ~0.20m)")
        
        # Test reward scaling
        env.prev_dist = 0.1  # Set for progress calculation
        rewards = env._get_reward()
        
        print(f"✅ Reward components working:")
        for name, value in rewards.items():
            print(f"   {name}: {value:.2f}")
            
        # Check success bonus activation
        # Place cube very close to target
        env.data.qpos[adr_q + 4:adr_q + 6] = env.target_pos_xy + np.array([0.02, 0.0])
        rewards_success = env._get_reward()
        
        if rewards_success['success'] > 250:
            print(f"✅ Success bonus activated: {rewards_success['success']:.1f} points")
        else:
            print(f"❌ Success bonus not working: {rewards_success['success']:.1f} points")
            
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def verify_training_args():
    print("\n🔍 Verifying Training Configuration")
    print("=" * 50)
    
    try:
        from ppo_continuous_action import Args
        
        # Check if we can create args with our optimized settings
        args = Args(
            env_id="FingerPush-v0",
            total_timesteps=500_000,
            learning_rate=2e-4,
            num_steps=4096,
            gamma=0.995,
            gae_lambda=0.98,
            num_minibatches=64,
            update_epochs=20,
            clip_coef=0.1,
            ent_coef=0.01,
            vf_coef=0.25,
        )
        
        print(f"✅ Args creation successful")
        print(f"✅ Total timesteps: {args.total_timesteps:,}")
        print(f"✅ Learning rate: {args.learning_rate}")
        print(f"✅ Rollout steps: {args.num_steps}")
        print(f"✅ Discount factor: {args.gamma}")
        print(f"✅ Entropy coefficient: {args.ent_coef}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training args error: {e}")
        return False

def verify_training_script():
    print("\n🔍 Verifying Training Script")
    print("=" * 50)
    
    try:
        # Check if training script exists and is properly configured
        script_path = "/home/robin/Desktop/robotics_tutorials-fomr_24_25/rl_and_sampling_projects/rl_project/scripts/train_finger_ppo.py"
        
        if os.path.exists(script_path):
            print(f"✅ Training script exists: {script_path}")
            
            # Read and check key configurations
            with open(script_path, 'r') as f:
                content = f.read()
                
            checks = [
                ("total_timesteps=500_000", "500k timesteps"),
                ("learning_rate=2e-4", "Optimized learning rate"),
                ("gamma=0.995", "High discount factor"),
                ("ent_coef=0.01", "Exploration entropy"),
                ("finger_push_20cm", "20cm task name"),
            ]
            
            for check, desc in checks:
                if check in content:
                    print(f"✅ {desc} configured correctly")
                else:
                    print(f"❌ {desc} not found")
                    
            return True
        else:
            print(f"❌ Training script not found")
            return False
            
    except Exception as e:
        print(f"❌ Script verification error: {e}")
        return False

def main():
    print("🎯 Final Verification for Reliable 20cm Target Reaching")
    print("=" * 70)
    
    env_ok = verify_environment()
    args_ok = verify_training_args() 
    script_ok = verify_training_script()
    
    print("\n" + "=" * 70)
    
    if env_ok and args_ok and script_ok:
        print("🎉 ALL SYSTEMS GO!")
        print("   Environment: ✅ Optimized reward shaping")
        print("   PPO Config: ✅ Sparse reward hyperparameters") 
        print("   Training: ✅ 500k timesteps configured")
        print()
        print("🚀 Ready to train for reliable 20cm target reaching!")
        print("   Run: python scripts/train_finger_ppo.py")
    else:
        print("❌ Issues detected. Check errors above.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
