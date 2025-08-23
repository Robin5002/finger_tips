#!/usr/bin/env python3

"""
Quick training test to verify the improved environment and PPO setup works correctly.
This runs a short training session to validate the changes.
"""

import sys
import os

# Add paths to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))

import envs.register_envs
from ppo_continuous_action import Args, train

def quick_training_test():
    print("🚀 Quick Training Test - Improved Environment")
    print("=" * 50)
    
    # Quick test with minimal training steps
    args = Args(
        env_id="FingerPush-v0",
        exp_name="quick_test_improved",
        total_timesteps=5_000,      # Very short test
        num_envs=1,
        track=False,                # Disable wandb for quick test
        capture_video=False,        # Disable video for quick test
        save_model=False,           # Don't save for quick test
        
        # Use the optimized hyperparameters
        learning_rate=2e-4,
        num_steps=1024,             # Shorter for quick test
        gamma=0.995,
        gae_lambda=0.98,
        num_minibatches=32,         # Smaller for quick test
        update_epochs=10,           # Fewer epochs for quick test
        norm_adv=True,
        clip_coef=0.1,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
    )
    
    print(f"   📦 Environment: {args.env_id}")
    print(f"   🎯 Training steps: {args.total_timesteps:,}")
    print(f"   ⚙️  Learning rate: {args.learning_rate}")
    print(f"   🎮 Rollout steps: {args.num_steps}")
    print(f"   🎲 Entropy coeff: {args.ent_coef}")
    print()
    
    try:
        print("🔥 Starting training...")
        avg_reward = train(args)
        print(f"✅ Training completed successfully!")
        print(f"📊 Final average reward: {avg_reward}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = quick_training_test()
    if success:
        print("\n🎉 Environment and training setup verified!")
        print("   Ready for full 500k timestep training.")
    else:
        print("\n💥 Issues detected. Check the error above.")
