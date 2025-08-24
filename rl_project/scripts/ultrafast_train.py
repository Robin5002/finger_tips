#!/usr/bin/env python3
"""
Ultra-fast training script - minimal configuration for maximum speed
"""
import sys
import os
import time
import torch

# Add paths
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cleanrl"))
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))

print(f"✅ Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

def main():
    try:
        # Register environments by importing
        import envs.register_envs  # This triggers registration
        
        # Import PPO
        from ppo_continuous_action import Args, train
        
        # Ultra-fast configuration
        args = Args(
            env_id="FingerPush-v0",
            exp_name="ultrafast",
            total_timesteps=20_000,        # Very short
            learning_rate=0.001,           # Higher LR for faster learning
            num_envs=1,
            num_steps=256,                 # Much smaller steps
            anneal_lr=False,               # Disable annealing for speed
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=4,             # Fewer minibatches
            update_epochs=3,               # Fewer epochs
            norm_adv=True,
            clip_coef=0.3,                 # Higher clip for faster convergence
            clip_vloss=True,
            ent_coef=0.02,                 # Higher entropy for exploration
            vf_coef=0.5,
            max_grad_norm=1.0,             # Higher grad norm
            target_kl=None,                # Disable target KL
            track=False,                   # DISABLE WANDB for max speed
            capture_video=False,
            save_model=False,              # Don't save for speed
            cuda=torch.cuda.is_available(),
        )

        print("\n⚡ ULTRA-FAST TRAINING:")
        print(f"   🎯 Target: {args.total_timesteps:,} timesteps")
        print(f"   📊 WandB: DISABLED for max speed")
        print(f"   💾 Model saving: DISABLED")
        print(f"   🔧 Steps: {args.num_steps}, Epochs: {args.update_epochs}")
        print()

        start_time = time.time()
        train(args)
        end_time = time.time()
        
        duration = end_time - start_time
        sps = args.total_timesteps / duration
        print(f"\n⚡ COMPLETED in {duration:.1f}s at {sps:.0f} SPS!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
