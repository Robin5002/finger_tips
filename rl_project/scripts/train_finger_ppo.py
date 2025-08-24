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
        total_timesteps=500_000,           # recommended for convergence
        num_envs=1,
        track=True,                        # W&B ON
        wandb_project_name="mujoco_tutorial",
        wandb_entity="mdabu-hanif-university-of-trento",
        capture_video=True,                # keep videos enabled
        save_model=True,
        cuda=False,                        # force CPU; faster for MuJoCo on Apple Silicon
    )

    # Allow environment override via ENV_ID or NO_ROLL=1
    env_id_override = os.environ.get("ENV_ID")
    if os.environ.get("NO_ROLL", "0") == "1":
        env_id_override = "FingerPushNoRoll-v0"
    if env_id_override:
        print(f"🔁 Overriding env_id → {env_id_override}")
        args.env_id = env_id_override

    # Recommended PPO knobs for this task
    args.learning_rate = 3e-4
    args.num_steps = 1024
    args.num_minibatches = 32
    args.update_epochs = 10
    args.ent_coef = 0.01
    args.clip_coef = 0.2
    args.target_kl = 0.015

    if EASY_CURRICULUM:
        # Softer updates + more exploration
        args.clip_coef = 0.10
        args.ent_coef = 0.02
        # More frequent updates
        args.num_steps = 512
        # Slightly stricter KL target
        args.target_kl = 0.012

    # --- Optional overrides via environment variables ---
    def _envf(name, default=None):
        v = os.environ.get(name)
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default
    def _envi(name, default=None):
        v = os.environ.get(name)
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    lr = _envf("LEARNING_RATE", None)
    if lr is not None:
        args.learning_rate = lr
        print(f"🔧 Override: learning_rate = {args.learning_rate}")
    ns = _envi("NUM_STEPS", None)
    if ns is not None:
        args.num_steps = ns
        print(f"🔧 Override: num_steps = {args.num_steps}")
    nmb = _envi("NUM_MINIBATCHES", None)
    if nmb is not None:
        args.num_minibatches = nmb
        print(f"🔧 Override: num_minibatches = {args.num_minibatches}")
    ue = _envi("UPDATE_EPOCHS", None)
    if ue is not None:
        args.update_epochs = ue
        print(f"🔧 Override: update_epochs = {args.update_epochs}")
    ec = _envf("ENT_COEF", None)
    if ec is not None:
        args.ent_coef = ec
        print(f"🔧 Override: ent_coef = {args.ent_coef}")
    cc = _envf("CLIP_COEF", None)
    if cc is not None:
        args.clip_coef = cc
        print(f"🔧 Override: clip_coef = {args.clip_coef}")
    tkl = _envf("TARGET_KL", None)
    if tkl is not None:
        args.target_kl = tkl
        print(f"🔧 Override: target_kl = {args.target_kl}")
    
    # Optional: allow overriding via best_params.json only when explicitly enabled
    use_best = os.environ.get("USE_BEST_PARAMS", "0") == "1"
    if use_best and best_params:
        print("✅ Applying overrides from best_params.json (USE_BEST_PARAMS=1)")
        for key, value in best_params.items():
            if hasattr(args, key):
                setattr(args, key, value)
                print(f"🔧 Overrode {key} = {value}")
            else:
                print(f"⚠️ Unknown parameter in best_params: {key}")
    else:
        print("ℹ️ Skipping best_params.json overrides (set USE_BEST_PARAMS=1 to enable).")
    
    print(f"\n🚀 FAST CPU TRAINING WITH WANDB:")
    print(f"   📦 Task: Finger pushing task")
    print(f"   🎯 Target: {args.total_timesteps:,} timesteps")
    print(f"   ⚙️  Hyperparameters: {'Optimized' if best_params else 'Default'}")
    print(f"   📊 WandB tracking: {'Enabled' if args.track else 'Disabled'}")
    print(f"   📹 Video capture: {'Enabled' if args.capture_video else 'Disabled'}")
    print(f"   🔧 Device: {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU (Optimized for MuJoCo)'}")
    print(f"   🎚️ Curriculum: {'ON (easy)' if EASY_CURRICULUM else 'OFF'}")
    print(f"   🧪 Env ID: {args.env_id}")
    print()
    # Display key env knobs (if set) so runs are reproducible from the log
    for k in [
        "SUCCESS_BONUS","STAY_NEAR_BONUS","TIME_PENALTY_COEF","SUCCESS_HOLD_SIMSTEPS",
        "HOLD_AGENT_THRESHOLD","SUCCESS_RADIUS","SUCCESS_RADIUS_EASY","SUCCESS_RADIUS_HARD",
        "AUTO_CURRICULUM","AUTO_CURRICULUM_MODE","CURRICULUM_WINDOW","CURRICULUM_STEP_UP",
        "CURRICULUM_STEP_DOWN","CURRICULUM_UP_THRESH","CURRICULUM_DOWN_THRESH","TARGET_OFFSET_EASY",
        "TARGET_OFFSET_HARD","EPISODE_SECONDS","KP","KD","ACTION_SCALE","DELTA_FRAC"
    ]:
        if os.environ.get(k) is not None:
            print(f"   ⚙️ {k}={os.environ.get(k)}")

    # Run training
    try:
        avg_reward = train(args)
        print(f"🎯 Final average reward: {avg_reward}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Check your environment setup and dependencies")
        sys.exit(1)
