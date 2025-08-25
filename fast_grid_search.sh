#!/bin/zsh

# --- Grid Search Script ---
# This script performs a grid search over various hyperparameters for the RL training.

# Narrowed grid (centered on the previously best run). 3 * 2 * 2 = 12 combos.
GRID_LEARNING_RATES=(1e-4 3e-4 5e-4)
GRID_ENT_COEFS=(0.01 0.02)
GRID_CLIP_COEFS=(0.08 0.10)

# Fixed knobs (from the strong run)
FIX_NUM_STEPS=512
FIX_NUM_MINIBATCHES=32
FIX_UPDATE_EPOCHS=10

# Runner defaults
TOTAL_TIMESTEPS_DEFAULT=${TOTAL_TIMESTEPS:-200000}
CONDA_ENV=${CONDA_ENV:-rl_mujoco_arm64}

# Iterate over all combinations of hyperparameters
for LR in "${GRID_LEARNING_RATES[@]}"; do
  for EC in "${GRID_ENT_COEFS[@]}"; do
    for CC in "${GRID_CLIP_COEFS[@]}"; do
      echo "Running with LR=$LR, NS=${FIX_NUM_STEPS}, NMB=${FIX_NUM_MINIBATCHES}, UE=${FIX_UPDATE_EPOCHS}, EC=$EC, CC=$CC"

      # Export the hyperparameters as environment variables (quote to force string types)
      export LEARNING_RATE="${LR}"
      export NUM_STEPS="${FIX_NUM_STEPS}"
      export NUM_MINIBATCHES="${FIX_NUM_MINIBATCHES}"
      export UPDATE_EPOCHS="${FIX_UPDATE_EPOCHS}"
      export ENT_COEF="${EC}"
      export CLIP_COEF="${CC}"
      export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS_DEFAULT}"

      # UTF-8 safety
      export PYTHONIOENCODING="utf-8"
      export LANG="en_US.UTF-8"

      # Try to activate conda env if available
      if command -v conda >/dev/null 2>&1; then
        # Ensure shell hook is initialized for non-interactive shells
        eval "$(conda shell.zsh hook)" >/dev/null 2>&1 || true
        conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true
      fi

      # Run the training script (if DRY_RUN=1 do not execute, just print)
      if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN: python3 rl_project/scripts/train_finger_ppo.py"
      else
        python3 rl_project/scripts/train_finger_ppo.py
      fi
    done
  done
done

# Note: This script assumes that the training script reads hyperparameters from environment variables.
# Use DRY_RUN=1 zsh fast_grid_search.sh to test loop expansion without launching training.
