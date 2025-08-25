import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import os
from torch.utils.tensorboard import SummaryWriter
import time


class FingerPushEnv(gym.Env):
    """A Gym environment for a finger robot pushing a cube to a target position."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(self,
                 xml_path: str = "../robot_description/finger_edu_description/xml/finger_edu_scene_cube.xml",
                 render_mode: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 no_roll: bool = False):
        super().__init__()

        self.no_roll = no_roll
        # Max allowed orientation deviation (degrees) for no-roll success (override via NOROLL_MAX_DEG)
        self.no_roll_max_deg = float(os.getenv("NOROLL_MAX_DEG", "5.0"))

        # Curriculum toggle (set EASY_CURRICULUM=1 in the environment)
        self.curriculum = os.getenv("EASY_CURRICULUM", "0") == "1"

        # --- Automatic curriculum (works even if EASY_CURRICULUM=0) ---
        self.auto_curriculum = os.getenv("AUTO_CURRICULUM", "1") == "1"
        # Offsets in meters
        self.easy_target_offset = float(os.getenv("TARGET_OFFSET_EASY", "0.12"))
        self.hard_target_offset = float(os.getenv("TARGET_OFFSET_HARD", "0.20"))
        # Success radii in meters
        self.easy_success_radius = float(os.getenv("SUCCESS_RADIUS_EASY", "0.08"))
        self.hard_success_radius = float(os.getenv("SUCCESS_RADIUS_HARD", "0.05"))
        # Number of episodes over which to linearly anneal from easy→hard
        self.curriculum_episodes = int(os.getenv("CURRICULUM_EPISODES", "300" if self.curriculum else "600"))

        # --- Performance-based curriculum (advances only when success rate is high) ---
        from collections import deque
        self.curriculum_mode = os.getenv("AUTO_CURRICULUM_MODE", "performance")  # "performance" or "linear"
        self.recent_successes = deque(maxlen=int(os.getenv("CURRICULUM_WINDOW", "50")))
        self.curriculum_progress = 0.0  # 0.0=easiest, 1.0=hardest
        self.curr_step_up = float(os.getenv("CURRICULUM_STEP_UP", "0.05"))
        self.curr_step_down = float(os.getenv("CURRICULUM_STEP_DOWN", "0.02"))
        self.curr_up_threshold = float(os.getenv("CURRICULUM_UP_THRESH", "0.6"))   # advance when SR >= 60%
        self.curr_down_threshold = float(os.getenv("CURRICULUM_DOWN_THRESH", "0.2"))  # relax when SR <= 20%

        # TensorBoard logging setup
        self.writer = None
        self.episode_reward_sum = 0.0
        self.episode_step_count = 0
        self.episode_counter = 0
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        # get scene model path
        model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), xml_path))

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # ---------- Identify actuated joints ----------
        self.actuated_joint_ids = []
        for i in range(self.model.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            if jid >= 0:
                self.actuated_joint_ids.append(jid)
        if len(self.actuated_joint_ids) == 0:
            self.actuated_joint_ids = [i for i, t in enumerate(self.model.jnt_type)
                                       if t == mujoco.mjtJoint.mjJNT_HINGE][: self.model.nu]

        self.num_dof = len(self.actuated_joint_ids)

        # Collect joint limits
        lows, highs = [], []
        for jid in self.actuated_joint_ids:
            if self.model.jnt_limited[jid]:
                r = self.model.jnt_range[jid]
                lows.append(r[0])
                highs.append(r[1])
            else:
                lows.append(-np.pi)
                highs.append(np.pi)
        self.joint_lower = np.array(lows, dtype=np.float32)
        self.joint_upper = np.array(highs, dtype=np.float32)

        # Actions are *normalized deltas* in [-1, 1] around a nominal pose
        self.action_space = spaces.Box(
            low=-np.ones(self.num_dof, dtype=np.float32),
            high=np.ones(self.num_dof, dtype=np.float32),
            shape=(self.num_dof,),
            dtype=np.float32,
        )
        self.prev_action = np.zeros(self.num_dof, dtype=np.float32)

        # Per-joint delta amplitude (fraction of span); curriculum keeps it gentler unless overridden
        span = (self.joint_upper - self.joint_lower).astype(np.float32)
        default_frac = 0.20 if self.curriculum else 0.30
        delta_frac = float(os.getenv("DELTA_FRAC", str(default_frac)))
        self.delta_scale = float(delta_frac) * span

        # ---------- Cube setup ----------
        free_joint_ids = np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if len(free_joint_ids) == 0:
            raise RuntimeError("No free joint found for the cube.")
        self.cube_free_jid = int(free_joint_ids[0])

        self.jnt_qposadr = self.model.jnt_qposadr.copy()
        self.jnt_dofadr = self.model.jnt_dofadr.copy()

        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        cube_qpos = self.data.qpos[adr_q:adr_q + 7]
        # MuJoCo free joint qpos layout: [x, y, z, qw, qx, qy, qz]
        cube_pos_xyz = cube_qpos[0:3]

        self.cube_init_pos = cube_pos_xyz[:2].copy()
        # Target will be set at each reset: forward (+x) from the cube start
        self.forward_axis = np.array([1.0, 0.0], dtype=np.float32)  # +x unit vector
        # These will be set per-reset via curriculum schedule
        self.target_offset = None
        self.target_pos_xy = None
        self.success_radius = None
        self.reach_goal_timer = 0
        # Success hold in *sim* steps; can override via env var
        self.reach_goal_threshold = int(os.getenv("SUCCESS_HOLD_SIMSTEPS", str(4 if self.curriculum else 10)))
        # Agent-step based hold (robust with decimation)
        self.goal_hold_agent_steps = 0
        # Require N consecutive agent steps within success radius (override via HOLD_AGENT_THRESHOLD)
        self.hold_agent_threshold = int(os.getenv("HOLD_AGENT_THRESHOLD", "1"))
        # Velocity tolerance to consider the cube "settled" near goal (m/s)
        self.vel_success_tol = float(os.getenv("VEL_SUCCESS_TOL", "0.06"))
        self.angvel_success_tol = float(os.getenv("ANGVEL_SUCCESS_TOL", "0.5"))
        self.eval_disable_success_term = os.getenv("EVAL_DISABLE_SUCCESS_TERMINATION", "0") == "1"
        # Last termination cause string for logging in infos (no printing by default)
        self._last_done_cause = None  # one of {"success","time_limit","oob", None}

        # Track whether success_held was ever achieved within the current episode
        self.success_ever = False

        # ---------- Controller parameters ----------
        self.kp = 2.0   # Position gain (prof spec)
        self.kd = 0.1   # Velocity gain (prof spec)
        self.action_scale = 1.0  # base
        if self.curriculum:
            self.kp = 4.0
            self.kd = 0.2
            self.action_scale = 1.2
        else:
            # a bit more authority on hard mode so the cube can travel 0.20 m reliably
            self.action_scale = 1.2
        # Optional overrides from environment variables
        self.kp = float(os.getenv("KP", str(self.kp)))
        self.kd = float(os.getenv("KD", str(self.kd)))
        self.action_scale = float(os.getenv("ACTION_SCALE", str(self.action_scale)))
        self.default_joint_pos = np.array([0.0, 0.5, -0.75], dtype=np.float32)[: self.num_dof]
        self.decimation = 4

        # --- Reward coefficient overrides ---
        self.success_bonus = float(os.getenv("SUCCESS_BONUS", "150.0" if not self.curriculum else "80.0"))
        self.stay_near_bonus = float(os.getenv("STAY_NEAR_BONUS", "3.0" if not self.curriculum else "1.5"))
        self.time_penalty_coef = float(os.getenv("TIME_PENALTY_COEF", "0.005" if not self.curriculum else "0.01"))

        # New: Allow environment variable overrides for penalty coefficients
        self.no_roll_penalty_coef = float(os.getenv("NO_ROLL_PENALTY_COEF", "5.0"))
        self.angvel_penalty_coef = float(os.getenv("ANGVEL_PENALTY_COEF", "0.05"))
        self.action_penalty_coef = float(os.getenv("ACTION_PENALTY_COEF", "0.01"))

        # Restoring torque gains for no-roll mode (applied to cube rotational DOFs)
        # Stronger defaults to resist visible rolling; tune these if translation is impacted
        # Bumped to high values to firmly resist roll; reduce if translation is impaired
    # Safer default gains to avoid instabilities; tune via env vars if needed
    self.no_roll_torque_kp = float(os.getenv("NO_ROLL_TORQUE_KP", "40.0"))
    self.no_roll_torque_kd = float(os.getenv("NO_ROLL_TORQUE_KD", "4.0"))
    self.no_roll_max_torque = float(os.getenv("NO_ROLL_MAX_TORQUE", "200.0"))
    # Additional global scale (multiply raw gains) to quickly dampen torque magnitude
    self.no_roll_torque_scale = float(os.getenv("NO_ROLL_TORQUE_SCALE", "0.02"))

        # ---------- Episode settings ----------
        # Longer horizon in hard mode helps complete 0.20 m pushes; override via EPISODE_SECONDS
        self.episode_length = float(os.getenv("EPISODE_SECONDS", "2.0" if self.curriculum else "4.0"))
        # Count both sim steps and agent steps (agent acts every `decimation` simsteps)
        self.sim_steps_per_episode = int(self.episode_length / self.model.opt.timestep)
        self.agent_steps_per_episode = max(1, int(self.sim_steps_per_episode / max(1, self.decimation)))
        # Expose agent-steps horizon for outside debuggers
        self.steps_per_episode = self.agent_steps_per_episode
        self.current_step = 0

        # Curriculum-sensitive shaping knobs
        self.lateral_coef = -1.0   # gentler in hard mode so it doesn't fight forward pushing
        self.init_xy_jitter = 0.06  # reduce lateral spawn spread in hard mode for reachability
        if self.curriculum:
            self.lateral_coef = -0.5
            self.init_xy_jitter = 0.05

        # ---------- Observation space ----------
        obs_dim = 2 * self.num_dof + 13 + 2 + self.num_dof
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ---------- Renderer ----------
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif self.render_mode == "rgb_array":
            self.viewer = mujoco.Renderer(self.model)

    def step(self, action: np.ndarray):
        # Keep legacy simstep-based hold counter for metrics
        self._check_goal_progress()
        # Count *agent* steps
        self.current_step += 1

        processed_action = self._process_action(action)

        # Advance simulation with decimation
        for _ in range(self.decimation):
            self._apply_action(processed_action)
            mujoco.mj_step(self.model, self.data)
            # If no_roll mode is enabled, force the cube to keep its initial orientation
            # and zero its angular velocity so it can translate (be kicked) without rolling.
            # This is a minimal, deterministic enforcement that preserves linear motion.
            if self.no_roll and hasattr(self, 'init_quat'):
                try:
                    adr_q = int(self.jnt_qposadr[self.cube_free_jid])
                    adr_v = int(self.jnt_dofadr[self.cube_free_jid])
                    # Read current orientation and angular velocity
                    q_cur = self.data.qpos[adr_q + 3: adr_q + 7].astype(np.float32)
                    ang_vel = self.data.qvel[adr_v + 3: adr_v + 6].astype(np.float32)

                    # Quaternion error: q_err = q_cur * conj(q_init)
                    q_init = self.init_quat
                    # conj of q_init
                    q_init_conj = np.array([q_init[0], -q_init[1], -q_init[2], -q_init[3]], dtype=np.float32)
                    a0, a1, a2, a3 = q_cur[0], q_cur[1], q_cur[2], q_cur[3]
                    b0, b1, b2, b3 = q_init_conj[0], q_init_conj[1], q_init_conj[2], q_init_conj[3]
                    # quaternion multiply a * b
                    q_err = np.empty(4, dtype=np.float32)
                    q_err[0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
                    q_err[1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
                    q_err[2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
                    q_err[3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0

                    # Use vector part of q_err as small-angle proportional error measure
                    q_err_vec = q_err[1:4]

                    # PD torque: proportional on quaternion vector part, derivative on ang vel
                    kp = float(getattr(self, 'no_roll_torque_kp', 20.0))
                    kd = float(getattr(self, 'no_roll_torque_kd', 2.0))
                    max_t = float(getattr(self, 'no_roll_max_torque', 100.0))
                    torque = (-kp * q_err_vec) - (kd * ang_vel)
                    # clip
                    torque = np.clip(torque, -max_t, max_t).astype(np.float32)

                    # Apply torques to rotational generalized forces (wx,wy,wz) for the free joint
                    try:
                        self.data.qfrc_applied[adr_v + 3: adr_v + 6] += torque
                    except Exception:
                        # If qfrc_applied isn't available, fallback to body forces (best-effort)
                        try:
                            # Apply equal and opposite body torques via xfrc_applied placeholder
                            self.data.xfrc_applied[adr_q + 0: adr_q + 3] += 0.0
                        except Exception:
                            pass
                except Exception:
                    pass
            self._check_goal_progress()
            if self.render_mode == "human" and self.viewer is not None:
                self.render()

        # --- Compute instantaneous success at the agent-step boundary ---
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        free_qpos = self.data.qpos[adr_q:adr_q + 7]
        free_qvel = self.data.qvel[int(self.jnt_dofadr[self.cube_free_jid]):int(self.jnt_dofadr[self.cube_free_jid]) + 6]
        # MuJoCo: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, wx,wy,wz]
        cube_pos = free_qpos[0:3].astype(np.float32)
        cube_quat = free_qpos[3:7].astype(np.float32)
        cube_xy = cube_pos[:2]
        cube_linvel = free_qvel[0:3].astype(np.float32)
        cube_angvel = free_qvel[3:6].astype(np.float32)
        dist_to_target = float(np.linalg.norm(cube_xy - self.target_pos_xy))
        success_flag = dist_to_target < self.success_radius
        orientation_deg = 0.0
        if self.no_roll:
            angle = self._quat_angle(self.init_quat, cube_quat)
            orientation_deg = float(np.rad2deg(angle))
            success_flag = success_flag and (angle < np.deg2rad(self.no_roll_max_deg))

        # Update *agent-step* hold timer (robust to decimation jitter)
        if success_flag:
            self.goal_hold_agent_steps += 1
        else:
            self.goal_hold_agent_steps = 0

        # Held in agent steps and/or in raw sim steps
        success_held_agent = (self.goal_hold_agent_steps >= self.hold_agent_threshold)
        success_held_sim = (self.reach_goal_timer >= self.reach_goal_threshold)
        # Unified held flag
        success_held = bool(success_held_agent or success_held_sim)

        if success_held:
            self.success_ever = True

        # Prepare observation and reward
        next_obs = self._get_obs()
        reward_dict = self._get_reward()
        total_reward = float(sum(reward_dict.values()))

        # Termination / truncation (override with success-held if necessary)
        terminated, truncated = self._get_dones()
        if success_held and not self.eval_disable_success_term:
            terminated = True
            self._last_done_cause = "success"

        # Ensure time-limit truncation at the environment level as a hard guard
        if not terminated and not truncated and (self.current_step >= self.agent_steps_per_episode):
            truncated = True
            self._last_done_cause = "time_limit"

        # Info dict for debugging/metrics
        info_dict = {
            "reward_dict": reward_dict,
            # Report unified success (held by either criterion) for training metrics
            "success": bool(success_held),
            "success_flag": bool(success_flag),
            "success_held": bool(success_held),
            "dist_to_target": dist_to_target,
            "hold_steps": int(self.goal_hold_agent_steps),
            # expose radius & instantaneous speeds for diagnosing SR=0 issues
            "success_radius": float(self.success_radius),
            "lin_speed": float(np.linalg.norm(cube_linvel[:2])),
            "ang_speed": float(np.linalg.norm(cube_angvel)),
            "success_ever": bool(self.success_ever),
        }
        if self.no_roll:
            info_dict["orientation_deg"] = orientation_deg
            # boolean gating at the agent boundary for quick triage
            info_dict["vel_ok"] = bool(np.linalg.norm(cube_linvel[:2]) <= float(self.vel_success_tol))
            info_dict["ang_ok"] = bool(
                (orientation_deg < float(self.no_roll_max_deg)) and
                (np.linalg.norm(cube_angvel) <= float(self.angvel_success_tol))
            )
        # also expose whether sim-step criterion is active
        info_dict["sim_held"] = bool(success_held_sim)

        # Attach done_cause if episode ended
        if terminated or truncated:
            info_dict["done_cause"] = self._last_done_cause

        # Save previous *normalized* action in [-1,1] for observations
        self.prev_action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Episode accounting
        self.episode_reward_sum += total_reward
        self.episode_step_count += 1
        if terminated or truncated:
            if self.writer:
                for k, v in reward_dict.items():
                    self.writer.add_scalar(f"reward_components/{k}", v, self.episode_counter)
                self.writer.add_scalar("episode/total_reward", self.episode_reward_sum, self.episode_counter)
                self.writer.add_scalar("episode/length", self.episode_step_count, self.episode_counter)
            # Update performance-based curriculum using outcome of this episode
            try:
                self.recent_successes.append(bool(success_held))
                if self.curriculum_mode == "performance" and self.auto_curriculum:
                    sr = float(np.mean(self.recent_successes)) if len(self.recent_successes) > 0 else 0.0
                    if sr >= self.curr_up_threshold:
                        self.curriculum_progress = min(1.0, self.curriculum_progress + self.curr_step_up)
                    elif sr <= self.curr_down_threshold:
                        self.curriculum_progress = max(0.0, self.curriculum_progress - self.curr_step_down)
            except Exception:
                pass

            self.episode_counter += 1
            self.episode_reward_sum = 0.0
            self.episode_step_count = 0

        return next_obs, total_reward, terminated, truncated, info_dict

    def _get_obs(self):
        q_list, qd_list = [], []
        for jid in self.actuated_joint_ids:
            qadr = int(self.jnt_qposadr[jid])
            dadr = int(self.jnt_dofadr[jid])
            q_list.append(self.data.qpos[qadr])
            qd_list.append(self.data.qvel[dadr])
        q = np.array(q_list, dtype=np.float32)
        qd = np.array(qd_list, dtype=np.float32)

        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        adr_v = int(self.jnt_dofadr[self.cube_free_jid])
        free_qpos = self.data.qpos[adr_q:adr_q + 7]
        free_qvel = self.data.qvel[adr_v:adr_v + 6]

        # Correct MuJoCo layout
        cube_pos = free_qpos[0:3].astype(np.float32)
        cube_quat = free_qpos[3:7].astype(np.float32)
        cube_linvel = free_qvel[0:3].astype(np.float32)
        cube_angvel = free_qvel[3:6].astype(np.float32)

        target_xy = self.target_pos_xy.astype(np.float32)

        obs = np.concatenate([q, qd, cube_pos, cube_quat, cube_linvel, cube_angvel, target_xy, self.prev_action],
                             dtype=np.float32)
        return obs


    def _quat_angle(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Absolute rotation angle between orientations q1 and q2 (MuJoCo quats are [w,x,y,z])."""
        dot = float(np.dot(q1, q2))
        dot = max(-1.0, min(1.0, abs(dot)))
        return 2.0 * np.arccos(dot)
    def _get_reward(self):
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        adr_v = int(self.jnt_dofadr[self.cube_free_jid])

        free_qpos = self.data.qpos[adr_q:adr_q + 7]
        free_qvel = self.data.qvel[adr_v:adr_v + 6]

        cube_pos = free_qpos[0:3].astype(np.float32)
        cube_quat = free_qpos[3:7].astype(np.float32)
        cube_xy = cube_pos[:2]
        cube_linvel = free_qvel[0:3].astype(np.float32)
        cube_angvel = free_qvel[3:6].astype(np.float32)

        # Forward velocity shaping along +x (encourages continuous push)
        vel_forward = float(np.dot(cube_linvel[:2], self.forward_axis))

        dist_to_target = float(np.linalg.norm(cube_xy - self.target_pos_xy))

        rewards = {}

        # 1) Dense distance shaping (penalize distance directly; removes positive baseline at start)
        rewards["distance"] = -5.0 * dist_to_target

        # 2) Lateral penalty (keep path straight)
        lateral_err = float(abs((cube_xy - self.target_pos_xy)[1]))
        rewards["lateral"] = float(self.lateral_coef) * lateral_err

        # 3) Forward progress along +x toward the target (positive only)
        # Use the vector from cube -> target so that moving forward (increasing cube_x) DECREASES this projection
        proj = float(np.dot((self.target_pos_xy - cube_xy), self.forward_axis))
        if hasattr(self, "prev_proj"):
            rewards["forward_progress"] = 10.0 * max(0.0, self.prev_proj - proj)
        else:
            rewards["forward_progress"] = 0.0
        self.prev_proj = proj
        # 3b) Reward for velocity toward the target (continuous incentive to keep pushing)
        rewards["vel_towards"] = 5.0 * max(0.0, vel_forward)

        # 4) Contact encouragement (robust to missing site)
        contact_reward = 0.0
        center_bonus = 0.0
        try:
            fingertip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
            if fingertip_site_id != -1:
                fingertip_pos = self.data.site_xpos[fingertip_site_id][:2]
                fingertip_to_cube = np.linalg.norm(fingertip_pos - cube_xy)
                if fingertip_to_cube < 0.03:
                    contact_reward = 5.0
                elif fingertip_to_cube < 0.05:
                    contact_reward = 2.0
                # Center-hit bonus: encourage contact near cube face center (small lateral offset)
                # Project fingertip->cube onto lateral axis (y if forward_axis is +x)
                try:
                    lateral_offset = abs((fingertip_pos - cube_xy)[1])
                    # bonus decreases linearly up to 4 cm lateral offset
                    if fingertip_to_cube < 0.05 and lateral_offset < 0.04:
                        center_bonus = 6.0 * (0.04 - lateral_offset)  # stronger incentive
                    else:
                        center_bonus = 0.0
                except Exception:
                    center_bonus = 0.0
        except Exception:
            center_bonus = 0.0
            pass
        rewards["contact"] = contact_reward
        rewards["center_hit"] = float(center_bonus)

        # 5) Success bonus (use unified success radius)
        rewards["success"] = self.success_bonus if dist_to_target < self.success_radius else 0.0
        # 5b) Stay-near reward (encourage holding the goal region)
        rewards["stay_near"] = self.stay_near_bonus if dist_to_target < self.success_radius else 0.0

        # 6) Efficiency
        rewards["action_penalty"] = -self.action_penalty_coef * np.sum(np.square(self.prev_action))
        rewards["time_penalty"] = -float(self.time_penalty_coef)

        # 7) No-roll shaping if enabled
        if self.no_roll:
            angle = self._quat_angle(self.init_quat, cube_quat)  # radians
            rewards["no_roll_penalty"] = -self.no_roll_penalty_coef * (angle ** 2)
            rewards["angvel_penalty"] = -self.angvel_penalty_coef * float(np.dot(cube_angvel, cube_angvel))

        # 8) Euclidean progress (non-negative)
        if hasattr(self, 'prev_dist'):
            progress = max(0.0, self.prev_dist - dist_to_target)
            rewards["progress"] = 20.0 * progress
        else:
            rewards["progress"] = 0.0

        self.prev_dist = dist_to_target
        return rewards

    def _get_dones(self):
        # Time-limit in *agent* steps
        time_limit_reached = self.current_step >= self.agent_steps_per_episode

        # Success if held for N consecutive agent steps, or for a number of raw sim steps
        success_held_agent = self.goal_hold_agent_steps >= self.hold_agent_threshold
        success_held_sim = self.reach_goal_timer >= self.reach_goal_threshold
        success_held = bool(success_held_agent or success_held_sim)

        # Out-of-bounds truncation based on cube XY
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        cube_xy = self.data.qpos[adr_q + 0:adr_q + 2]
        cube_out_of_bounds = (np.abs(cube_xy).max() > 0.3)

        terminated = bool(success_held) and (not self.eval_disable_success_term)
        truncated = bool(time_limit_reached or cube_out_of_bounds)

        # Record cause for one-line info (no prints here)
        if terminated:
            self._last_done_cause = "success"
        elif time_limit_reached:
            self._last_done_cause = "time_limit"
        elif cube_out_of_bounds:
            self._last_done_cause = "oob"
        else:
            self._last_done_cause = None

        return terminated, truncated

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        # clamp to normalized range
        a = np.clip(a, -1.0, 1.0)
        # ensure correct shape
        if a.shape[0] != self.num_dof:
            a = np.resize(a, (self.num_dof,)).astype(np.float32)
        # map to absolute targets around default pose
        target = self.default_joint_pos + (self.action_scale * a * self.delta_scale)
        # final safety clip to limits
        target = np.clip(target.astype(np.float32), self.joint_lower, self.joint_upper)
        return target

    def _apply_action(self, joint_pos_target: np.ndarray):
        q = np.array([self.data.qpos[int(self.jnt_qposadr[jid])] for jid in self.actuated_joint_ids], dtype=np.float32)
        qd = np.array([self.data.qvel[int(self.jnt_dofadr[jid])] for jid in self.actuated_joint_ids], dtype=np.float32)
        torque = self.kp * (joint_pos_target - q) - self.kd * qd
        self.data.ctrl[:self.num_dof] = torque.astype(np.float32)

    def _check_goal_progress(self):
        """
        Count sim-step holds only when distance, (optional) no-roll, and low velocity are satisfied.
        """
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        adr_v = int(self.jnt_dofadr[self.cube_free_jid])
        qpos = self.data.qpos[adr_q:adr_q + 7]
        qvel = self.data.qvel[adr_v:adr_v + 6]

        cube_xy = qpos[0:2]
        cube_quat = qpos[3:7]
        cube_linvel = qvel[0:3]
        cube_angvel = qvel[3:6]

        dist_ok = (np.linalg.norm(cube_xy - self.target_pos_xy) < self.success_radius)
        vel_ok = (np.linalg.norm(cube_linvel[:2]) <= float(self.vel_success_tol))

        noroll_ok = True
        if self.no_roll:
            angle_rad = self._quat_angle(self.init_quat, cube_quat)
            ang_ok = (np.linalg.norm(cube_angvel) <= float(self.angvel_success_tol))
            noroll_ok = (angle_rad < np.deg2rad(self.no_roll_max_deg)) and ang_ok

        if dist_ok and vel_ok and noroll_ok:
            self.reach_goal_timer += 1
        else:
            self.reach_goal_timer = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.reach_goal_timer = 0
        self.current_step = 0
        self.prev_action = np.zeros(self.num_dof, dtype=np.float32)
        self.goal_hold_agent_steps = 0
        self.success_ever = False

        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        adr_v = int(self.jnt_dofadr[self.cube_free_jid])

        mujoco.mj_resetData(self.model, self.data)

        # Curriculum-aware reset randomization
        jitter = float(getattr(self, "init_xy_jitter", 0.10))
        cube_init_xy = self.cube_init_pos + np.random.uniform(-jitter, +jitter, size=2).astype(np.float32)
        self.data.qpos[adr_q + 0:adr_q + 2] = cube_init_xy

        # Log fingertip proximity at reset (helps diagnose reachability)
        try:
            fingertip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
            if fingertip_site_id != -1:
                fingertip_xy = self.data.site_xpos[fingertip_site_id][:2].copy()
                reset_dist = float(np.linalg.norm(fingertip_xy - cube_init_xy))
                if self.writer is not None:
                    self.writer.add_scalar("reset/fingertip_to_cube", reset_dist, self.episode_counter)
        except Exception:
            pass

        # Save and normalize initial orientation for no-roll metric
        free_qpos = self.data.qpos[adr_q:adr_q + 7]
        q = free_qpos[3:7].astype(np.float32).copy()
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        self.init_quat = q

        # --- Curriculum schedule: linearly anneal offset & success radius over episodes ---
        # Compute progress fraction for auto curriculum
        if self.auto_curriculum:
            if self.curriculum_mode == "performance":
                frac = float(self.curriculum_progress)
            else:
                frac = min(1.0, float(self.episode_counter) / max(1, int(self.curriculum_episodes)))
        else:
            frac = 1.0 if not self.curriculum else 0.0
        # If EASY_CURRICULUM=1, start easier (frac near 0 early); otherwise still anneal if AUTO_CURRICULUM=1
        start_off = self.easy_target_offset if (self.curriculum or self.auto_curriculum) else self.hard_target_offset
        end_off = self.hard_target_offset
        self.target_offset = (1.0 - frac) * start_off + frac * end_off

        start_sr = self.easy_success_radius if (self.curriculum or self.auto_curriculum) else self.hard_success_radius
        end_sr = self.hard_success_radius
        self.success_radius = float((1.0 - frac) * start_sr + frac * end_sr)
        # Allow explicit SUCCESS_RADIUS override to win
        if "SUCCESS_RADIUS" in os.environ:
            try:
                self.success_radius = float(os.environ["SUCCESS_RADIUS"])  # explicit override
            except Exception:
                pass

        # Clamp to a reasonable minimum to avoid accidental zero/tiny radius (e.g., bad env var)
        try:
            min_sr = float(os.getenv("SUCCESS_RADIUS_MIN", "0.03"))  # 3 cm default floor
            if self.success_radius < min_sr:
                self.success_radius = float(min_sr)
        except Exception:
            pass

        # Set target forward of the current cube start with the scheduled offset
        self.target_pos_xy = cube_init_xy + float(self.target_offset) * self.forward_axis

        # Initialize robot joints near the nominal pose (small noise), curriculum gentler
        span = (self.joint_upper - self.joint_lower).astype(np.float32)
        noise_scale = 0.02 if self.curriculum else 0.05  # fraction of span
        for i, jid in enumerate(self.actuated_joint_ids):
            qadr = int(self.jnt_qposadr[jid])
            jitter = float(np.random.uniform(-noise_scale, noise_scale)) * float(span[i])
            self.data.qpos[qadr] = float(np.clip(self.default_joint_pos[i] + jitter,
                                                 self.joint_lower[i], self.joint_upper[i]))

        # Track distance/projection for shaping
        self.prev_dist = float(np.linalg.norm(self.data.qpos[adr_q + 0:adr_q + 2] - self.target_pos_xy))
        self.prev_proj = float(np.dot((self.target_pos_xy - self.data.qpos[adr_q + 0:adr_q + 2]), self.forward_axis))

        # Move visual target marker if it exists
        try:
            marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
            self.model.geom_pos[marker_id] = np.array([self.target_pos_xy[0], self.target_pos_xy[1], 0.05], dtype=np.float32)
        except Exception:
            pass

        if self.writer is not None:
            try:
                self.writer.add_scalar("curriculum/progress", float(self.curriculum_progress), self.episode_counter)
                self.writer.add_scalar("curriculum/success_radius", float(self.success_radius), self.episode_counter)
                self.writer.add_scalar("curriculum/target_offset", float(self.target_offset), self.episode_counter)
            except Exception:
                pass
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except Exception:
                    return None
            if self.viewer is not None:
                try:
                    self.viewer.sync()
                except Exception:
                    pass
            return None
        elif self.render_mode == 'rgb_array':
            if self.viewer is None:
                try:
                    self.viewer = mujoco.Renderer(self.model)
                except Exception:
                    return None
            if self.viewer is not None:
                try:
                    self.viewer.update_scene(self.data)
                    return self.viewer.render()
                except Exception:
                    pass
        return None

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if self.writer:
            self.writer.close()