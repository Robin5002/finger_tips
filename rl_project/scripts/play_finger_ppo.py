# This script is modified From cleanrl/ppo_eval.py to run in real-time

import os
import torch
import numpy as np
import gymnasium as gym
from cleanrl.ppo_continuous_action import Agent
from gymnasium.vector import SyncVectorEnv
import time
import argparse

import envs.register_envs 

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode="human")
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory name")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes to evaluate")
    args = parser.parse_args()

    model_path = f"runs/{args.checkpoint}/ppo_continuous_action.cleanrl_model"

    envs = SyncVectorEnv([make_env("FingerPush-v0", 0, False, "FingerPush-v0-eval", 0.99)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    timestep = envs.envs[0].model.opt.timestep
    decimation = envs.envs[0].decimation
    step_duration = timestep * decimation

    try:
        while len(episodic_returns) < args.eval_episodes:
            start_time = time.time()
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            next_obs, _, terminated, truncated, infos = envs.step(actions.cpu().numpy())

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]

            obs = next_obs

            elapsed = time.time() - start_time
            if step_duration > elapsed:
                time.sleep(step_duration - elapsed)

    finally:
        envs.close()

if __name__ == "__main__":
    main()