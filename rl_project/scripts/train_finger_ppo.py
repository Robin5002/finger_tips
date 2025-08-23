import sys
import os
import time
import json
import torch

# Verify we're in the correct conda environment
expected_env = "rl_mujoco_playground"
current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
if current_env != expected_env:
    print(f"⚠️ Warning: Expected environment '{expected_env}', but current is '{current_env}'")
    print(f"   Run: conda activate {expected_env}")
else:
    print(f"✅ Correct environment activated: {current_env}")

# Add paths to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)  # Add project root
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))  # Add cleanrl

# Check GPU availability
print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 CUDA device: {torch.cuda.get_device_name()}")
    print(f"🔧 CUDA version: {torch.version.cuda}")
    print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("🔧 Running on CPU")

import envs.register_envs

if __name__ == "__main__":
    # Import after path setup
    try:
        from ppo_continuous_action import Args, train
    except ImportError as e:
        print(f"❌ Failed to import PPO: {e}")
        print("Make sure you're in the correct environment and cleanrl is accessible")
        sys.exit(1)
    
    # Load best params if JSON exists
    best_params_path = os.path.join(REPO_ROOT, "best_params.json")
    try:
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        print("✅ Loaded best params from best_params.json")
    except FileNotFoundError:
        best_params = {}
        print("⚠️ No best_params.json found, using default Args")

    # Check if we have a registered environment
    try:
        import gymnasium as gym
        env = gym.make("FingerPush-v0")
        env.close()
        print("✅ FingerPush-v0 environment registered successfully")
    except Exception as e:
        print(f"❌ Environment registration failed: {e}")
        print("Available environments might be limited")

    # Build Args with best params override
    args = Args(
        env_id="FingerPush-v0",
        exp_name="finger_push_optimized_hyperparams",
        total_timesteps=5_000,      # Start with fewer timesteps for testing
        num_envs=1,                 # Single environment for now
        track=True,                 # Use wandb tracking
        wandb_project_name="mujoco_tutorial",  # WandB project name
        wandb_entity="mdabu-hanif-university-of-trento",  # WandB entity
        capture_video=True,         # Record videos
        save_model=True,            # Enable model saving
        cuda=torch.cuda.is_available(),  # Use CUDA if available
    )
    
    # Override with best parameters from JSON
    for key, value in best_params.items():
        if hasattr(args, key):
            setattr(args, key, value)
            print(f"🔧 Set {key} = {value}")
        else:
            print(f"⚠️ Unknown parameter: {key}")
    
    print(f"\n🚀 STARTING OPTIMIZED TRAINING:")
    print(f"   📦 Task: Finger pushing task")
    print(f"   🎯 Target: {args.total_timesteps:,} timesteps")
    print(f"   ⚙️  Hyperparameters: {'Optimized' if best_params else 'Default'}")
    print(f"   📊 WandB tracking: {'Enabled' if args.track else 'Disabled'}")
    print(f"   📹 Video capture: {'Enabled' if args.capture_video else 'Disabled'}")
    print(f"   🔧 Device: {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
    print()

    # Run training
    try:
        avg_reward = train(args)
        print(f"🎯 Final average reward: {avg_reward}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Check your environment setup and dependencies")
        sys.exit(1)
