# Contact-Rich Robotics Course, SoSe, 2025

This directory provides the environment and scripts that will allow you to complete the RL part of the final project for your Contact-Rich Robotics course. The aim of the project is to push a box to a desired location using a fixed base 3-DOF finger robot. A base environment using [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) wrapped inside [Gym](https://gymnasium.farama.org/) interface is already provided and linked to [cleanRL](https://github.com/Atarilab/cleanrl) repository for testing different RL algorithms. [cleanRL](https://github.com/Atarilab/cleanrl) provides single-file implementations of different RL algorithms, making it a good educational tool for understanding different RL algorithms.

The part of your project was created by Shafeef Omar, PhD student at Prof. Majid's chair. In case you run into any issues, feel free to contact him at [shafeef.omar@tum.de](mailto:shafeef.omar@tum.de).

## Getting started

### Setup Instructions
- Install Miniconda (if already not installed), a minimal conda installer, allowing us to create virtual environments easily. The installation instructions can be found [here](https://docs.anaconda.com/free/miniconda/).
- Create a new virtual environment with the necessary packages already defined in `environment.yml` and activate it using the following commands:

  ```bash
  conda env create -f environment.yml
  conda activate rl_mujoco_playground
  ```

- Fetch the submodules linked to this project (forked cleanRL submodule linked to the repo):
  ```bash
  git submodule update --init --recursive
  ```

- Install ffmpeg for rendering
  ```bash
  sudo apt install -y ffmpeg
  ```

- Install cleanRL
  ```bash
  cd cleanrl && pip install -e .
  ```

- Optional
  Based on the GPU you have, different torch versions might be required. Training on GPU may not be necessary for our use case. In case you would like to train on GPU, first uninstall existing installation of `torch` from cleanRL. 

  ```bash
  pip uninstall torch
  ```
  Check your CUDA version using:
  ```bash
  nvidia-smi
  ```

  For CUDA 11.8:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu1181     
  ```

  For CUDA 12.1:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121              
  ```
## Project Instructions
To complete this project, you will mostly only need to complete the functions in the environment defined in [rl_project/envs/finger_env.py](https://github.com/Atarilab/contact_rich_robotics/blob/main/rl_project/envs/finger_env.py). It creates a MuJoCo simulation environment wrapped inside the Gymnasium interface, which is a standard for RL to be used across various RL algorithm libraries. Additionally, if you want to tune the hyperparameters of your algorithm, you can do so in [rl_project/scripts/train_finger_ppo.py](https://github.com/Atarilab/contact_rich_robotics/blob/main/rl_project/scripts/train_finger_ppo.py). You can further dig deep into the implementation of the `PPO` algorithm we use in completing this project [here](https://github.com/Atarilab/cleanrl/blob/047a51da154bccc0081e343ae809a21bae7882d4/cleanrl/ppo_continuous_action.py). You could also try out other learning algorithms such as [DDPG](https://github.com/Atarilab/cleanrl/blob/047a51da154bccc0081e343ae809a21bae7882d4/cleanrl/ddpg_continuous_action.py).


### Training your finger robot to push the box
```bash
cd ../ # move to contact_rich_robotics/rl_project
python scripts/train_finger_ppo.py # update wandb entity and login
```

### Playing the finger robot env
```bash
python scripts/play_finger_env.py
```

### Weights & Biases (WandB)
You can use W&B to view the training progress online, during and after training. It's useful to see the performance of your policy and compare it to other runs. Create an account and login to WandB to view training logs online. Make sure to use your WandB username in the `wandb_entity` argument in `scripts/train_finger_ppo.py`.
