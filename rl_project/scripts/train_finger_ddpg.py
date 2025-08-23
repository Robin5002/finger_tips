import sys
import runpy

# Register the env
from envs import register_envs

# command-line arguments for CleanRL DDPG
sys.argv = [
    "ddpg_continuous_action.py",
    "--env-id", "FingerPush-v0",
    "--track", "--wandb_entity=shafeefo", "--wandb_project_name=mujoco_tutorial",
    "--total_timesteps=200000",
    "--save-model",
    "--capture-video"
]

# Run the CleanRL script in the current interpreter
runpy.run_path("cleanrl/cleanrl/ddpg_continuous_action.py", run_name="__main__")