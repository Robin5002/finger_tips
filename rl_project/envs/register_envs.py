"""
Register custom environments with Gymnasium.
"""

import gymnasium as gym

# Import available environment classes
try:
    from envs.finger_env import FingerPushEnv
    print("✅ Successfully imported FingerPushEnv")
except ImportError as e:
    print(f"⚠️ finger_env not found: {e}")
    FingerPushEnv = None

# Register environments that are available
if FingerPushEnv is not None:
    gym.register(
        id='FingerPush-v0',
        entry_point='envs.finger_env:FingerPushEnv',
        max_episode_steps=2400,
    )
    print("✅ Registered FingerPush-v0 environment")

print("✅ Environment registration completed") 
