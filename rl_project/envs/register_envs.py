import gymnasium as gym
from envs.finger_env import FingerPushEnv

# Standard variant: allow rolling
gym.register(
    id="FingerPush-v0",
    entry_point="envs.finger_env:FingerPushEnv",
    kwargs={"no_roll": False},
)

# No-roll constraint variant
gym.register(
    id="FingerPushNoRoll-v0",
    entry_point="envs.finger_env:FingerPushEnv",
    kwargs={"no_roll": True},
)
