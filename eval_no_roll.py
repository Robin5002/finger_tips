import os
import sys
import torch
import numpy as np
# Ensure imports behave like training scripts: add rl_project to sys.path so `envs` is a top-level package
sys.path.insert(0, os.path.abspath("rl_project"))
from envs.finger_env import FingerPushEnv

MODEL_PATH = "runs/FingerPush-v0__finger_push_ppo__1__1756157582/finger_push_ppo.cleanrl_model"

def load_model(path):
    # CleanRL model is PyTorch state_dict saved with torch.save
    try:
        return torch.load(path, map_location=torch.device('cpu'))
    except Exception as e:
        print('failed to load model:', e)
        return None


def run_eval():
    os.environ['NO_ROLL'] = '1'
    env = FingerPushEnv(render_mode=None, no_roll=True)
    model = load_model(MODEL_PATH)
    if model is None:
        print('no model loaded; exiting')
        return
    # If the saved artifact contains a policy network, try to get its state
    # For this smoke eval we'll just run a random policy to observe orientation traces
    obs, _ = env.reset()
    done = False
    step = 0
    print('step,orientation_deg,ang_speed')
    while not done and step < env.steps_per_episode:
        action = np.zeros(env.num_dof, dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        # compute current orientation wrt init_quat
        adr_q = int(env.jnt_qposadr[env.cube_free_jid])
        cube_quat = env.data.qpos[adr_q+3:adr_q+7].astype(np.float32)
        angle = env._quat_angle(env.init_quat, cube_quat)
        ang_speed = float(np.linalg.norm(env.data.qvel[int(env.jnt_dofadr[env.cube_free_jid])+3:int(env.jnt_dofadr[env.cube_free_jid])+6]))
        print(f"{step},{np.degrees(angle):.3f},{ang_speed:.3f}")
        step += 1
        if term or trunc:
            done = True
    env.close()

if __name__ == '__main__':
    run_eval()
