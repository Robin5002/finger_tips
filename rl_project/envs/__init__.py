from gymnasium.envs.registration import register

from envs.finger_env import FingerPushEnv

register(
    id="FingerPush-v0",
    entry_point="envs.finger_env:FingerPushEnv",
    max_episode_steps=200,
)