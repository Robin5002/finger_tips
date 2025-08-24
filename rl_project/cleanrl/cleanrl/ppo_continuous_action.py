# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# # Import our environment registration
# import scripts.register_envs

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "mujoco_tutorial"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Debugging/verbosity
    debug: bool = False
    """print periodic training diagnostics to the terminal"""
    debug_print_every: int = 4096
    """how many environment steps between debug prints"""
    iter_print_every: int = 8
    """print [ITER]/[STATS] every N iterations (always print for the first few)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        # Build base env
        env = gym.make(env_id)

        # Enforce episode endings using the env’s agent-step horizon
        try:
            max_steps = getattr(env.unwrapped, "steps_per_episode", None)
            if max_steps is None:
                max_steps = getattr(env.unwrapped, "agent_steps_per_episode", None)
            if max_steps is not None and int(max_steps) > 0:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_steps))
        except Exception:
            pass

        # Wrapper stack (RecordEpisodeStatistics outermost so it sees terminal flags)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        try:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        except TypeError:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # Start with a slightly smaller std to reduce heavy clipping in [-1,1]
        self.actor_logstd = nn.Parameter(torch.full((1, int(np.prod(envs.single_action_space.shape))), -0.5))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)



def _get_info_scalar(infos, key):
    """Extract a scalar from vectorized env infos for key, if present; else returns None."""
    if not isinstance(infos, dict):
        return None
    if key not in infos:
        return None
    val = infos[key]
    try:
        # Handle numpy arrays from SyncVectorEnv (shape: (nenv,))
        import numpy as _np
        if isinstance(val, _np.ndarray):
            if val.size == 0:
                return None
            return float(val.reshape(-1)[0])
        # Handle lists/tuples
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return None
            return float(val[0])
        # Fallback: try direct cast
        return float(val)
    except Exception:
        return None


def _action_saturation_fraction(action_np: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of action dims within 2% of finite bounds. Ignores +/-inf and NaN limits."""
    try:
        a = np.asarray(action_np, dtype=np.float32).reshape(-1)
        lo = np.asarray(low, dtype=np.float32).reshape(-1)
        hi = np.asarray(high, dtype=np.float32).reshape(-1)
        n = min(a.size, lo.size, hi.size)
        a, lo, hi = a[:n], lo[:n], hi[:n]

        finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        if not np.any(finite):
            return 0.0
        span = hi[finite] - lo[finite]
        eps = 0.02 * (span + 1e-8)
        near_low = a[finite] <= (lo[finite] + eps)
        near_high = a[finite] >= (hi[finite] - eps)
        return float(np.mean(np.logical_or(near_low, near_high)))
    except Exception:
        return 0.0


def _render_eval_video(agent, args, run_name, device, obs_rms=None, max_steps=600):
    """Create a *single* evaluation video using the same wrapper stack as training
    (Flatten, ClipAction, NormalizeObservation, TransformObservation[clip]).
    Use deterministic (mean) actions to avoid sampling noise in the video.
    """
    import os
    import imageio.v2 as imageio
    import gymnasium as gym
    import numpy as np
    import torch

    # Configurable output cadence
    fps = int(os.getenv("EVAL_FPS", "12"))
    min_seconds = float(os.getenv("EVAL_MIN_SECONDS", "5"))  # ensure at least N seconds of footage
    min_frames = max(1, int(fps * min_seconds))

    # Prompt E: allow full-horizon eval video by disabling success termination
    run_full_horizon = os.getenv("EVAL_RUN_FULL_HORIZON", "1") == "1"
    if run_full_horizon:
        os.environ["EVAL_DISABLE_SUCCESS_TERMINATION"] = "1"
        # Ensure the episode lasts at least the requested min_seconds
        try:
            current_sec = float(os.getenv("EPISODE_SECONDS", "4.0"))
        except Exception:
            current_sec = 4.0
        desired_sec = max(current_sec, min_seconds + 1.0)
        os.environ["EPISODE_SECONDS"] = str(desired_sec)

    # Eval behavior knobs
    one_episode = os.getenv("EVAL_ONE_EPISODE", "1") == "1"  # default: single episode to avoid moving target between resets
    eval_seed = int(os.getenv("EVAL_SEED", str(args.seed)))
    freeze_curriculum = os.getenv("EVAL_FREEZE_CURRICULUM", "0") == "1"

    os.makedirs(f"videos/{run_name}", exist_ok=True)

    # Build eval env with SAME wrapper order as training to avoid distribution shift
    base = gym.make(args.env_id, render_mode="rgb_array")

    # Enforce episode endings using the env’s agent-step horizon (same as training)
    try:
        max_steps_env = getattr(base.unwrapped, "steps_per_episode", None)
        if max_steps_env is None:
            max_steps_env = getattr(base.unwrapped, "agent_steps_per_episode", None)
        if max_steps_env is not None and int(max_steps_env) > 0:
            base = gym.wrappers.TimeLimit(base, max_episode_steps=int(max_steps_env))
    except Exception:
        pass

    env = gym.wrappers.ClipAction(base)
    norm_obs = gym.wrappers.NormalizeObservation(env)
    try:
        env = gym.wrappers.TransformObservation(norm_obs, lambda obs: np.clip(obs, -10, 10))
    except TypeError:
        env = gym.wrappers.TransformObservation(norm_obs, lambda obs: np.clip(obs, -10, 10), norm_obs.observation_space)
    env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Optionally freeze curriculum dynamics during eval so target/radius don't drift
    if freeze_curriculum:
        try:
            base_unwrapped = env.unwrapped
            if hasattr(base_unwrapped, "auto_curriculum"):
                base_unwrapped.auto_curriculum = False
            if hasattr(base_unwrapped, "curriculum_mode"):
                base_unwrapped.curriculum_mode = "linear"  # fixed by progress value
            if hasattr(base_unwrapped, "curriculum_progress"):
                # keep current difficulty, but stop changing it
                base_unwrapped.curriculum_progress = float(getattr(base_unwrapped, "curriculum_progress", 1.0))
        except Exception:
            pass

    # If we have running obs stats from training, copy them to eval and freeze updates
    try:
        if obs_rms is not None and hasattr(norm_obs, "obs_rms") and norm_obs.obs_rms is not None:
            if "mean" in obs_rms and obs_rms["mean"] is not None:
                norm_obs.obs_rms.mean = np.asarray(obs_rms["mean"], dtype=np.float64)
            if "var" in obs_rms and obs_rms["var"] is not None:
                norm_obs.obs_rms.var = np.asarray(obs_rms["var"], dtype=np.float64)
            if "count" in obs_rms and obs_rms["count"] is not None:
                norm_obs.obs_rms.count = float(obs_rms["count"])
            # Freeze normalization updates during eval if supported
            if hasattr(norm_obs, "training"):
                norm_obs.training = False
    except Exception:
        pass

    frames = []

    if one_episode:
        # Capture exactly one episode so the target remains fixed within the video
        try:
            obs, _ = env.reset(seed=eval_seed)
        except TypeError:
            obs, _ = env.reset()
        for _ in range(max_steps):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_mean = agent.actor_mean(obs_t)
                action = torch.clamp(action_mean, -1.0, 1.0)
            obs, _, term, trunc, _ = env.step(action.detach().cpu().numpy()[0])
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if term or trunc:
                break
    else:
        # Keep recording multiple episodes until we reach the minimum frame budget
        episodes_captured = 0
        while len(frames) < min_frames and episodes_captured < 10:
            try:
                obs, _ = env.reset(seed=(eval_seed + episodes_captured))
            except TypeError:
                obs, _ = env.reset()
            for _ in range(max_steps):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action_mean = agent.actor_mean(obs_t)
                    action = torch.clamp(action_mean, -1.0, 1.0)
                obs, _, term, trunc, _ = env.step(action.detach().cpu().numpy()[0])
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                if term or trunc:
                    episodes_captured += 1
                    break
            if episodes_captured == 0 and len(frames) >= max_steps:
                episodes_captured = 1

    out_path = f"videos/{run_name}/eval.mp4"
    if frames:
        # Ensure minimum duration: if the episode ended quickly, pad with the last frame
        if len(frames) < min_frames:
            frames.extend([frames[-1]] * (min_frames - len(frames)))
        imageio.mimsave(out_path, frames, fps=fps)
        return out_path
    return None


def train(args):
    """Main training function that can be imported and called from other scripts."""
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Rolling buffer of recent episode stats for terminal debugging
    ep_hist = deque(maxlen=100)

    # Fallback episode accumulators (single-env)
    ep_ret_acc = 0.0
    ep_len_acc = 0
    ep_last_dist = float('nan')
    ep_last_succ = 0.0

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    elif False and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Temporarily disable MPS
        device = torch.device("mps")
        # Aggressive MPS optimization
        torch.set_default_dtype(torch.float32)
        torch.set_default_tensor_type('torch.FloatTensor')
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("🔧 Using MPS (Apple Silicon GPU) with aggressive float32 optimization")
    else:
        device = torch.device("cpu")
        print("🔧 Using CPU (forced for maximum MuJoCo performance)")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    # Debug: show steps per episode of the base env
    try:
        base_env = envs.envs[0].unwrapped
        spe = getattr(base_env, "steps_per_episode", None)
        if spe is not None:
            print(f"ℹ️ steps_per_episode = {spe}")
    except Exception:
        pass
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Cache action space bounds for saturation diagnostics
    _act_low = None
    _act_high = None
    try:
        _act_low = np.asarray(envs.single_action_space.low, dtype=np.float32)
        _act_high = np.asarray(envs.single_action_space.high, dtype=np.float32)
    except Exception:
        pass

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape,
                      dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape,
                          dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    for iteration in range(1, args.num_iterations + 1):
        # ---- per-iteration debug accumulators ----
        iter_min_dist = float("inf")
        iter_success_hits = 0  # count of steps where env reports success=True (not necessarily held)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # Fallback accounting (quiet): accumulate return/length and last known distance/success
            try:
                ep_ret_acc += float(np.mean(reward))
            except Exception:
                ep_ret_acc += float(reward)
            ep_len_acc += 1
            d_now = _get_info_scalar(infos, "dist_to_target")
            if d_now is not None:
                ep_last_dist = float(d_now)
            s_now = _get_info_scalar(infos, "success_held")
            if s_now is None:
                s_now = _get_info_scalar(infos, "success")
            if s_now is not None:
                ep_last_succ = float(s_now)
            # ---- accumulate quick diagnostics from infos ----
            try:
                d_now = _get_info_scalar(infos, "dist_to_target")
                if d_now is not None:
                    iter_min_dist = min(iter_min_dist, d_now)
            except Exception:
                pass
            try:
                s_now = _get_info_scalar(infos, "success_held")
                if s_now is None:
                    s_now = _get_info_scalar(infos, "success")
                if s_now is not None and s_now >= 0.5:
                    iter_success_hits += 1
            except Exception:
                pass
            if args.debug and (global_step % args.debug_print_every == 0):
                # reward and distance snapshot
                dist_now = _get_info_scalar(infos, "dist_to_target")
                try:
                    r_mean = float(np.mean(reward))
                except Exception:
                    r_mean = float(reward)
                sat_frac = None
                try:
                    if _act_low is not None and _act_high is not None:
                        sat_frac = _action_saturation_fraction(action.detach().cpu().numpy(), _act_low, _act_high)
                except Exception:
                    pass
                msg = f"[DBG] step={global_step} r={r_mean:.3f}"
                if dist_now is not None:
                    msg += f" dist={dist_now:.3f}m"
                if sat_frac is not None:
                    msg += f" action_sat={sat_frac*100:.1f}%"
                # basic NaN guards (numpy or torch)
                try:
                    has_nan_obs = (np.isnan(next_obs).any() if isinstance(next_obs, np.ndarray)
                                   else (torch.isnan(next_obs).any().item() if torch.is_tensor(next_obs) else False))
                except Exception:
                    has_nan_obs = False
                if has_nan_obs:
                    msg += " WARN: next_obs has NaNs"
                try:
                    has_nan_val = bool(torch.isnan(values[step]).any().item())
                except Exception:
                    has_nan_val = False
                if has_nan_val:
                    msg += " WARN: value has NaNs"
                print(msg)
            next_done = np.logical_or(terminations, truncations)
            if args.debug:
                try:
                    if np.asarray(next_done).astype(bool).any():
                        print(f"[PPO] done detected at global_step={global_step}")
                except Exception:
                    pass
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs, next_done = torch.as_tensor(next_obs, dtype=torch.float32, device=device), torch.as_tensor(next_done, dtype=torch.float32, device=device)

            # If the vector env reports done but `final_info` is missing, log a compact episode summary
            if (not ("final_info" in infos and infos["final_info"])) and (bool(next_done[0].item()) if torch.is_tensor(next_done) else bool(next_done)):
                writer.add_scalar("charts/episodic_return", ep_ret_acc, global_step)
                writer.add_scalar("charts/episodic_length", ep_len_acc, global_step)
                # Push compact stats into rolling buffer for [STATS]
                ep_hist.append({
                    "ret": ep_ret_acc,
                    "len": ep_len_acc,
                    "succ": 1.0 if ep_last_succ >= 0.5 else 0.0,
                    "dist": ep_last_dist,
                })
                # reset accumulators
                ep_ret_acc, ep_len_acc = 0.0, 0
                ep_last_dist, ep_last_succ = float('nan'), 0.0

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        # Reset fallback accumulators when proper final_info is available
                        ep_ret_acc, ep_len_acc = 0.0, 0
                        ep_last_dist, ep_last_succ = float('nan'), 0.0

                        # Terminal diagnostics
                        succ = (info.get("success_held") if isinstance(info, dict) and ("success_held" in info)
                                else (info.get("success") if isinstance(info, dict) else None))
                        dist = info.get("dist_to_target") if isinstance(info, dict) else None
                        extra = []
                        if succ is not None:
                            extra.append(f"success={succ}")
                        if dist is not None:
                            try:
                                extra.append(f"dist={float(dist):.3f}m")
                            except Exception:
                                pass
                        if "orientation_deg" in info:
                            try:
                                extra.append(f"ori={float(info['orientation_deg']):.1f}°")
                            except Exception:
                                pass
                        if "done_cause" in info:
                            try:
                                extra.append(f"cause={info['done_cause']}")
                            except Exception:
                                pass
                        if extra:
                            print("    ↳ " + ", ".join(extra))
                        if "reward_dict" in info and isinstance(info["reward_dict"], dict):
                            r = info["reward_dict"]
                            # Print a compact snapshot of key components from the last step
                            keys = [k for k in ["distance","forward_progress","lateral","success","time_penalty","action_penalty","no_roll_penalty"] if k in r]
                            if keys:
                                snap = ", ".join([f"{k}={r[k]:+.2f}" for k in keys])
                                print(f"    ↳ reward_snapshot: {snap}")

                        # Log success/dist/orientation if provided by the env
                        succ_for_log = (info.get("success_held") if "success_held" in info else info.get("success") if "success" in info else None)
                        if succ_for_log is not None:
                            writer.add_scalar("episode/success", 1.0 if succ_for_log else 0.0, global_step)
                            if args.track:
                                import wandb
                                wandb.log({"episode/success": 1.0 if succ_for_log else 0.0})
                        if "dist_to_target" in info:
                            writer.add_scalar("episode/dist_to_target", info["dist_to_target"], global_step)
                        if "orientation_deg" in info:
                            writer.add_scalar("episode/orientation_deg", info["orientation_deg"], global_step)

                        # Log individual reward components if available
                        if "reward_dict" in info:
                            for reward_name, reward_value in info["reward_dict"].items():
                                writer.add_scalar(f"rewards/{reward_name}", reward_value, global_step)

                        # (disabled) per-episode video logging removed — we'll log only one final eval video at training end

                        # Collect compact episode stats for rolling summaries
                        try:
                            ep_ret = float(info["episode"]["r"]) if "episode" in info else float("nan")
                            ep_len = float(info["episode"]["l"]) if "episode" in info else float("nan")
                        except Exception:
                            ep_ret, ep_len = float("nan"), float("nan")
                        succ_f = None
                        if "success" in info:
                            try:
                                succ_f = bool(info["success"]) * 1.0
                            except Exception:
                                succ_f = None
                        dist_f = None
                        if "dist_to_target" in info:
                            try:
                                dist_f = float(info["dist_to_target"])
                            except Exception:
                                dist_f = None
                        ep_hist.append({"ret": ep_ret, "len": ep_len, "succ": succ_f, "dist": dist_f})
                        # Print a compact single-line summary
                        comp = []
                        if not np.isnan(ep_ret):
                            comp.append(f"R={ep_ret:.2f}")
                        if not np.isnan(ep_len):
                            comp.append(f"L={ep_len:.0f}")
                        if dist_f is not None:
                            comp.append(f"D={dist_f:.3f}m")
                        if succ_f is not None:
                            comp.append(f"S={int(succ_f)}")
                        if comp:
                            print("    ↳ epi_summary: " + ", ".join(comp))

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, dtype=torch.float32, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if args.debug:
            print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # Print iteration summary sparingly
        _print_now = (iteration <= 3) or (iteration % max(1, args.iter_print_every) == 0)
        if np.isfinite(iter_min_dist):
            if _print_now:
                print(f"[ITER] {iteration}/{args.num_iterations} step={global_step} minDist={iter_min_dist:.3f}m success_hits={iter_success_hits}")
            writer.add_scalar("stats/iter_min_dist", iter_min_dist, global_step)
        else:
            if _print_now:
                print(f"[ITER] {iteration}/{args.num_iterations} step={global_step} minDist=nan success_hits={iter_success_hits}")
        writer.add_scalar("stats/iter_success_hits", iter_success_hits, global_step)

        # Rolling statistics over recent episodes (debug)
        if len(ep_hist) > 0 and _print_now:
            # Use last 20 if available, else all
            last_n = list(ep_hist)[-20:] if len(ep_hist) >= 20 else list(ep_hist)
            def _safe_mean(k):
                vals = [x[k] for x in last_n if x.get(k) is not None and not (isinstance(x.get(k), float) and np.isnan(x.get(k)))]
                return float(np.mean(vals)) if len(vals) > 0 else float("nan")
            sr = _safe_mean("succ")  # success rate (0/1)
            mr = _safe_mean("ret")   # mean episodic return
            md = _safe_mean("dist")  # mean terminal distance
            ml = _safe_mean("len")   # mean episode length
            sr_txt = f"{sr*100:.1f}%" if not np.isnan(sr) else "nan"
            mr_txt = f"{mr:.2f}" if not np.isnan(mr) else "nan"
            md_txt = f"{md:.3f}m" if not np.isnan(md) else "nan"
            ml_txt = f"{ml:.0f}" if not np.isnan(ml) else "nan"
            print(f"[STATS] iter={iteration}/{args.num_iterations} step={global_step} SR@20={sr_txt} meanR@20={mr_txt} meanDist@20={md_txt} meanLen@20={ml_txt}")
            # TensorBoard logging
            if not np.isnan(sr):
                writer.add_scalar("stats/success_rate_20", sr, global_step)
            if not np.isnan(mr):
                writer.add_scalar("stats/mean_return_20", mr, global_step)
            if not np.isnan(md):
                writer.add_scalar("stats/mean_terminal_dist_20", md, global_step)
            if not np.isnan(ml):
                writer.add_scalar("stats/mean_ep_length_20", ml, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
        # TEMPORARILY DISABLED FOR WANDB TESTING
        # from cleanrl_utils.evals.ppo_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=Agent,
        #     device=device,
        #     gamma=args.gamma,
        #     capture_video=False,  # Disable video recording during evaluation to prevent duplicates
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            # TEMPORARILY DISABLED FOR WANDB TESTING
            # from cleanrl_utils.huggingface import push_to_hub

            # repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            # repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            # push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")
            print("Model upload disabled during WandB testing")
            pass

    # --- Extract obs normalization stats from the training env (envs.envs[0]) ---
    obs_rms = None
    try:
        # envs.envs[0] is RecordEpisodeStatistics -> Flatten -> NormalizeReward -> TransformObs -> NormalizeObs -> ClipAction -> Base
        w = envs.envs[0]
        # Walk down the wrapper chain to find NormalizeObservation
        for _ in range(8):
            if hasattr(w, "obs_rms") and w.obs_rms is not None:
                obs_rms = {
                    "mean": np.array(w.obs_rms.mean, copy=True),
                    "var": np.array(w.obs_rms.var, copy=True),
                    "count": float(getattr(w.obs_rms, "count", 1.0)),
                }
                break
            if hasattr(w, "env"):
                w = w.env
            else:
                break
    except Exception:
        obs_rms = None

    # --- Fallback eval video upload to W&B ---
    if args.track and args.capture_video:
        try:
            import wandb, os
            out = _render_eval_video(agent, args, run_name, device, obs_rms=obs_rms, max_steps=400)
            if out is not None and os.path.exists(out):
                print(f"🎥 Eval video saved: {out}")
                # Log directly to the run
                wandb.log({"video": wandb.Video(out, fps=12, format="mp4")})
                # Also log as an artifact to guarantee it appears in the run files
                art = wandb.Artifact(name=f"{run_name}-videos", type="video")
                art.add_file(out)
                wandb.log_artifact(art)
                wandb.log({"_video_eval_path": out})
            else:
                print("⚠️ No eval video generated (no frames or file missing)")
        except Exception as e:
            print(f"⚠️ Failed to render/upload eval video: {e}")

    envs.close()
    writer.close()
    
    # Return final performance metric (average of last few episodes)
    return 0.0  # Placeholder return value


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
