# RL and Sampling Projects

This repository contains multiple robotics projects focused on reinforcement learning and sampling methods.

## Projects

### 🤖 RL Project (`rl_project/`)
Reinforcement Learning project for finger robot control and cube manipulation.

**Features:**
- Custom MuJoCo environment for finger pushing tasks
- PPO implementation with CleanRL
- Optimized hyperparameters
- GPU acceleration support (CUDA 11.8)
- WandB integration for experiment tracking

**Environment:** `rl_mujoco_playground`

### 📊 Sampling Project (`sampling_project/`)
Sampling methods and techniques for robotics applications.

## Setup

Each project has its own environment and setup instructions. Check the individual project README files for detailed setup instructions.

### Quick Start
1. Navigate to the desired project directory
2. Follow the setup instructions in that project's README
3. Activate the appropriate conda environment

## Requirements

- Python 3.8+
- Conda/Anaconda
- CUDA-capable GPU (recommended)
- MuJoCo license (for simulation)

## Repository Structure

```
├── rl_project/              # Reinforcement Learning project
│   ├── envs/               # Custom environments
│   ├── scripts/            # Training scripts
│   ├── cleanrl/            # CleanRL implementation
│   └── requirements.txt    # RL-specific dependencies
└── sampling_project/        # Sampling methods project
    └── ...                 # Project-specific files
```
