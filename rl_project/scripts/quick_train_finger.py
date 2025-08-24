#!/usr/bin/env python3
"""
Quick training script for FingerPush environment - runs for just 10,000 timesteps for fast testing
"""
import sys
import os
import time
import json
import torch

# Verify we're in the correct conda environment
expected_envs = ["rl_mujoco_playground", "rl_mujoco_arm64"]
current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
if current_env not in expected_envs:
    print(f"⚠️ Warning: Expected environment {expected_envs}, but current is '{current_env}'")
    print(f"   Run: conda activate rl_mujoco_playground (or rl_mujoco_arm64)")
else:
    print(f"✅ Correct environment activated: {current_env}")

# Add paths to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)  # Add project root
sys.path.append(os.path.join(REPO_ROOT, "cleanrl"))  # Add cleanrl parent directory
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))  # Add cleanrl

print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"🔧 Apple Metal (MPS) available: {mps_available}")

# Select device
if torch.cuda.is_available():
    device_str = f"CUDA ({torch.cuda.get_device_name(0)})"
elif mps_available:
    device_str = "CPU (MPS disabled for compatibility)"  # We disabled MPS due to float64 issues
else:
    device_str = "CPU"
print(f"🔧 Selected device: {device_str}")

def main():
    try:
        # Register environments
        from envs.register_envs import register_finger_envs
        register_finger_envs()
        print("✅ FingerPush-v0 environment registered successfully")
        
        # Import PPO training function
        from ppo_continuous_action import Args, train
        
        # Load best parameters if available
        best_params_path = os.path.join(REPO_ROOT, "best_params.json")
        best_params = {}
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            print("✅ Loaded best params from best_params.json")
        else:
            print("ℹ️ No best_params.json found, using defaults")
        
        # Create training arguments - VERY SHORT TRAINING FOR QUICK TEST
        args = Args(
            env_id="FingerPush-v0",
            exp_name="quick_finger_test",
            total_timesteps=10_000,              # Very short for quick test!
            learning_rate=0.0003,
            num_envs=1,                          # Single environment for simplicity  
            num_steps=512,                       # Smaller steps per update
            anneal_lr=True,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=8,                   # Fewer minibatches 
            update_epochs=4,                     # Fewer epochs per update
            norm_adv=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            track=False,                         # Disable wandb for quick test
            capture_video=False,                 # Disable video for speed
            save_model=False,                    # Don't save model for quick test
            cuda=torch.cuda.is_available(),
        )

        print("\n🚀 STARTING QUICK TRAINING TEST:")
        print(f"   📦 Task: Finger pushing task")
        print(f"   🎯 Target: {args.total_timesteps:,} timesteps (QUICK TEST)")
        print(f"   ⚙️  Hyperparameters: Simplified for speed")
        print(f"   📊 WandB tracking: Disabled")
        print(f"   📹 Video capture: Disabled")
        print(f"   🔧 Device: {device_str}")
        print()

        start_time = time.time()
        
        # Run training
        train(args)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✅ Quick training completed in {duration:.1f} seconds!")
        print(f"   Average SPS: {args.total_timesteps / duration:.0f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Check your environment setup and dependencies")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
