#!/usr/bin/env python3
"""
Test script for FingerPushNoRoll-v0 environment - 100k timesteps
"""
import sys
import os
import time
import json
import torch

# Verify environment
expected_envs = ["rl_mujoco_playground", "rl_mujoco_arm64"]
current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
if current_env not in expected_envs:
    print(f"⚠️ Warning: Expected environment {expected_envs}, but current is '{current_env}'")
else:
    print(f"✅ Correct environment activated: {current_env}")

# Add paths to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "cleanrl"))
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))

print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"🔧 Apple Metal (MPS) available: {mps_available}")

# Select device
if torch.cuda.is_available():
    device_str = f"CUDA ({torch.cuda.get_device_name(0)})"
elif mps_available:
    device_str = "CPU (MPS too slow due to fallback)"  # Based on testing
else:
    device_str = "CPU"
print(f"🔧 Selected device: {device_str}")

def main():
    try:
        # Register environments
        from envs.register_envs import register_finger_envs
        register_finger_envs()
        print("✅ FingerPushNoRoll-v0 environment registered successfully")
        
        # Import PPO
        from ppo_continuous_action import Args, train
        
        # Load best parameters
        best_params_path = os.path.join(REPO_ROOT, "best_params.json")
        best_params = {}
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            print("✅ Loaded best params from best_params.json")
        
        # NoRoll environment test
        args = Args(
            env_id="FingerPushNoRoll-v0",  # Different environment!
            exp_name="noroll_test_100k",
            total_timesteps=100_000,
            learning_rate=0.0003,
            num_envs=1,
            num_steps=2048,
            anneal_lr=True,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=8,
            update_epochs=10,
            norm_adv=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.015,
            track=True,                      # Enable WandB
            capture_video=False,             # Disabled for speed
            save_model=True,
            cuda=torch.cuda.is_available(),
        )
        
        # Override with best parameters
        for key, value in best_params.items():
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"🔧 Set {key} = {value}")

        print("\n🚀 STARTING NOROLL TRAINING TEST:")
        print(f"   📦 Task: Finger pushing task (NoRoll)")
        print(f"   🎯 Target: {args.total_timesteps:,} timesteps")
        print(f"   ⚙️  Hyperparameters: Optimized")
        print(f"   📊 WandB tracking: Enabled")
        print(f"   📹 Video capture: Disabled")
        print(f"   🔧 Device: {device_str}")
        print()

        start_time = time.time()
        train(args)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\n✅ NoRoll training completed in {duration:.1f} seconds!")
        print(f"   Average SPS: {args.total_timesteps / duration:.0f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
