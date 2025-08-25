#!/bin/zsh

# Quick grid subset — small, fast runs for smoke testing
# 6 combinations: LR in [3e-4, 1e-4, 5e-5] x ENT_COEF in [0.01, 0.02]

# Optionally set CONDA_ENV to auto-activate. Defaults to rl_mujoco_arm64.
CONDA_ENV=${CONDA_ENV:-rl_mujoco_arm64}
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]; then
  if command -v conda >/dev/null 2>&1; then
    echo "Activating conda environment: ${CONDA_ENV}"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  else
    echo "Conda not found; please activate ${CONDA_ENV} manually and re-run the script."
    exit 1
  fi
fi

# Ensure Python prints UTF-8 safely (fixes UnicodeEncodeError for emoji in logs)
export PYTHONIOENCODING=${PYTHONIOENCODING:-utf-8}
export LANG=${LANG:-en_US.UTF-8}

# Quick subset parameters
QUICK_LRS=(3e-4 1e-4 5e-5)
QUICK_ENT=(0.01 0.02)
NS=512
NMB=32
UE=10
CC=0.2

# Small total timesteps for quick smoke tests
export TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-20000}

for LR in "${QUICK_LRS[@]}"; do
  for EC in "${QUICK_ENT[@]}"; do
    echo "=== START: LR=$LR EC=$EC NS=$NS NMB=$NMB UE=$UE CC=$CC TT=${TOTAL_TIMESTEPS} ==="
    export LEARNING_RATE="$LR"
    export NUM_STEPS="$NS"
    export NUM_MINIBATCHES="$NMB"
    export UPDATE_EPOCHS="$UE"
    export ENT_COEF="$EC"
    export CLIP_COEF="$CC"
    # Set a run-specific tag for logs (used by train script printing)
    export GRID_RUN_TAG="quick_lr_${LR}_ent_${EC}"

    # Run training serially (this will block until the run completes)
    python3 rl_project/scripts/train_finger_ppo.py
    echo "=== FINISH: LR=$LR EC=$EC ==="
    # Short pause between runs
    sleep 1
  done
done

echo "All quick runs complete."
