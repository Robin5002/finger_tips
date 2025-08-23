from setuptools import setup, find_packages

setup(
    name="mujoco_finger_env",
    version="0.1",
    description="A MuJoCo environment for a finger robot pushing a cube",
    author="Shafeef Omar",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "mujoco",
        "numpy",
        "wandb",
    ],
    python_requires=">=3.8",
    entry_points={
        "gymnasium.envs": [
            "FingerPush-v0 = envs.finger_env:FingerPushEnv",
        ],
    },
)