#!/usr/bin/env python3
"""
Maximum performance training - optimized CPU with aggressive settings
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

# Force CPU for maximum MuJoCo performance
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print(f"✅ Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

def main():
    try:
        # Register environments
        import envs.register_envs  # This triggers registration
        
        # Import PPO
        from ppo_continuous_action import Args, train
        
        # CPU-optimized configuration for maximum speed
        args = Args(
            env_id="FingerPush-v0",
            exp_name="maxspeed_cpu",
            total_timesteps=100_000,       # Back to 100k but optimized
            learning_rate=0.0005,          # Higher LR
            num_envs=1,
            num_steps=1024,                # Balanced for CPU
            anneal_lr=False,               # Disable for speed
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=8,             # Optimized for CPU
            update_epochs=4,               # Reduced epochs
            norm_adv=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,                # Disable target KL
            track=False,                   # DISABLE WANDB for max speed
            capture_video=False,
            save_model=True,               # Save final model
            cuda=False,                    # Force CPU
        )

        print("\n🚀 MAXIMUM SPEED CPU TRAINING:")
        print(f"   🎯 Target: {args.total_timesteps:,} timesteps")
        print(f"   📊 WandB: DISABLED for max speed")
        print(f"   🔧 Device: CPU (forced for max MuJoCo speed)")
        print(f"   ⚡ Config: {args.num_steps} steps, {args.update_epochs} epochs")
        print()

        start_time = time.time()
        train(args)
        end_time = time.time()
        
        duration = end_time - start_time
        sps = args.total_timesteps / duration
        print(f"\n🚀 COMPLETED in {duration:.1f}s at {sps:.0f} SPS!")
        print(f"   Expected at this rate: 500k in {500_000/sps:.1f}s")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
