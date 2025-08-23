#!/bin/bash
# Activate rl_mujoco_playground environment
echo "🚀 Activating rl_mujoco_playground environment..."
conda activate rl_mujoco_playground

# Check if activation was successful
if [[ "$CONDA_DEFAULT_ENV" == "rl_mujoco_playground" ]]; then
    echo "✅ Environment activated successfully"
    echo "🔧 Python path: $(which python)"
    echo "🔧 Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "❌ Failed to activate environment"
fi
