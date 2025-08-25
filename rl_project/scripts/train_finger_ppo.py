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

# Check GPU availability

print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"🔧 Apple Metal (MPS) available: {mps_available}")

# Summarize what will be used (mirrors selection in the PPO file)
if torch.cuda.is_available():
    device_str = f"CUDA ({torch.cuda.get_device_name(0)})"
elif mps_available:
    device_str = "MPS (Apple Silicon GPU)"
else:
    device_str = "CPU"
print(f"🔧 Selected device: {device_str}")

# Curriculum env var and print
EASY_CURRICULUM = os.environ.get("EASY_CURRICULUM", "0") == "1"
if EASY_CURRICULUM:
    print("🎚️ EASY_CURRICULUM=1 (env + PPO tweaks enabled)")

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

    args = Args(
        env_id="FingerPush-v0",           # Part (a)
        exp_name="finger_push_ppo",
        total_timesteps=int(os.getenv("TOTAL_TIMESTEPS", "500000")), # Allow override for grid search; default 500k
        num_envs=1,
        track=True,                        # W&B ON
        wandb_project_name="mujoco_tutorial",
        wandb_entity="mdabu-hanif-university-of-trento",
        capture_video=True,                # keep videos enabled
        save_model=True,
        cuda=False,                        # force CPU; faster for MuJoCo on Apple Silicon
    )

    # --- Apply recovered run hyperparameters as defaults, but allow environment overrides ---
    # Recovered best-run defaults:
    # learning_rate=0.0003, clip_coef=0.2, ent_coef=0, num_steps=2048,
    # num_minibatches=32, update_epochs=10, total_timesteps=200000
    args.learning_rate = float(os.getenv("LEARNING_RATE", str(getattr(args, 'learning_rate', 0.0003))))
    args.clip_coef = float(os.getenv("CLIP_COEF", str(getattr(args, 'clip_coef', 0.2))))
    args.ent_coef = float(os.getenv("ENT_COEF", str(getattr(args, 'ent_coef', 0.0))))
    args.num_steps = int(os.getenv("NUM_STEPS", str(getattr(args, 'num_steps', 2048))))
    args.num_minibatches = int(os.getenv("NUM_MINIBATCHES", str(getattr(args, 'num_minibatches', 32))))
    args.update_epochs = int(os.getenv("UPDATE_EPOCHS", str(getattr(args, 'update_epochs', 10))))
    args.total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", str(getattr(args, 'total_timesteps', 500000))))


    # ASCII-safe banner to avoid UnicodeEncodeError in some terminals/Python builds
    print("\nFAST CPU TRAINING WITH WANDB:")
    print("   Task: Finger pushing task")
    print(f"   Target: {args.total_timesteps:,} timesteps")
    print(f"   Hyperparameters: {'Optimized' if best_params else 'Default'}")
    print(f"   WandB tracking: {'Enabled' if args.track else 'Disabled'}")
    print(f"   Video capture: {'Enabled' if args.capture_video else 'Disabled'}")
    print(f"   Device: {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU (Optimized for MuJoCo)'}")
    print(f"   Curriculum: {'ON (easy)' if EASY_CURRICULUM else 'OFF'}")
    print(f"   Env ID: {args.env_id}")
    print()
    # Display key env knobs (if set) so runs are reproducible from the log
    for k in [
        # reward & termination shaping
        "SUCCESS_BONUS","STAY_NEAR_BONUS","TIME_PENALTY_COEF",
        "SUCCESS_HOLD_SIMSTEPS","HOLD_AGENT_THRESHOLD","SUCCESS_RADIUS","SUCCESS_RADIUS_MIN",
        "SUCCESS_RADIUS_EASY","SUCCESS_RADIUS_HARD",
        # curriculum & target
        "AUTO_CURRICULUM","AUTO_CURRICULUM_MODE","CURRICULUM_WINDOW","CURRICULUM_STEP_UP",
        "CURRICULUM_STEP_DOWN","CURRICULUM_UP_THRESH","CURRICULUM_DOWN_THRESH",
        "TARGET_OFFSET_EASY","TARGET_OFFSET_HARD",
        # control & horizon
        "EPISODE_SECONDS","KP","KD","ACTION_SCALE","DELTA_FRAC",
        # task toggles & eval
        "NO_ROLL","NOROLL_MAX_DEG","VEL_SUCCESS_TOL","ANGVEL_SUCCESS_TOL",
        "EVAL_ONE_EPISODE","EVAL_FREEZE_CURRICULUM","EVAL_RUN_FULL_HORIZON","EVAL_MIN_SECONDS","EVAL_FPS"
    ]:
        if os.environ.get(k) is not None:
            print(f"   {k}={os.environ.get(k)}")

    # Run training
    try:
        avg_reward = train(args)
        print(f"🎯 Final average reward: {avg_reward}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Check your environment setup and dependencies")
        sys.exit(1)
