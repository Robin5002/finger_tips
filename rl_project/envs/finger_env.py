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
                 log_dir: Optional[str] = None):
        super().__init__()

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

        self.action_space = spaces.Box(
            low=-np.ones(self.num_dof, dtype=np.float32),
            high=np.ones(self.num_dof, dtype=np.float32),
            shape=(self.num_dof,),
            dtype=np.float32
        )
        self.prev_action = np.zeros(self.num_dof, dtype=np.float32)

        # ---------- Cube setup ----------
        free_joint_ids = np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if len(free_joint_ids) == 0:
            raise RuntimeError("No free joint found for the cube.")
        self.cube_free_jid = int(free_joint_ids[0])

        self.jnt_qposadr = self.model.jnt_qposadr.copy()
        self.jnt_dofadr = self.model.jnt_dofadr.copy()

        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        cube_qpos = self.data.qpos[adr_q:adr_q + 7]
        cube_pos_xyz = cube_qpos[4:7]
        self.cube_init_pos = cube_pos_xyz[:2].copy()

        # Target is very close to make task achievable
        self.target_pos_xy = self.cube_init_pos + np.array([0.20, 0.0], dtype=np.float32)  # Only 20cm away!
        # Change to different direction (e.g., 15cm at 45 degrees):
        self.target_pos_xy = self.cube_init_pos + np.array([0.106, 0.106], dtype=np.float32)
        self.reach_goal_timer = 0
        self.reach_goal_threshold = 100

        # ---------- Controller parameters ----------
        self.kp = 20.0  # Very strong PD gain for precise control
        self.kd = 1.0   # Good damping
        self.action_scale = 1.0  # More conservative actions
        self.default_joint_pos = np.array([0.0, 0.5, -0.75], dtype=np.float32)[: self.num_dof]
        self.decimation = 4

        # ---------- Episode settings ----------
        self.episode_length = 30  # Even more time per episode
        self.steps_per_episode = int(self.episode_length / (self.model.opt.timestep * self.decimation))
        self.current_step = 0

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
        self._check_goal_progress()
        self.current_step += 1

        processed_action = self._process_action(action)

        for _ in range(self.decimation):
            self._apply_action(processed_action)
            mujoco.mj_step(self.model, self.data)
            self._check_goal_progress()
            if self.render_mode == "human" and self.viewer is not None:
                self.render()

        next_obs = self._get_obs()
        reward_dict = self._get_reward()
        total_reward = float(sum(reward_dict.values()))
        terminated, truncated = self._get_dones()

        self.prev_action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Update episode tracking
        self.episode_reward_sum += total_reward
        self.episode_step_count += 1

        if terminated or truncated:
            if self.writer:
                for k, v in reward_dict.items():
                    self.writer.add_scalar(f"reward_components/{k}", v, self.episode_counter)
                self.writer.add_scalar("episode/total_reward", self.episode_reward_sum, self.episode_counter)
                self.writer.add_scalar("episode/length", self.episode_step_count, self.episode_counter)
            self.episode_counter += 1
            self.episode_reward_sum = 0.0
            self.episode_step_count = 0

        return next_obs, total_reward, terminated, truncated, {"reward_dict": reward_dict}

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

        cube_quat = free_qpos[0:4].astype(np.float32)
        cube_pos = free_qpos[4:7].astype(np.float32)
        cube_angvel = free_qvel[0:3].astype(np.float32)
        cube_linvel = free_qvel[3:6].astype(np.float32)

        target_xy = self.target_pos_xy.astype(np.float32)

        obs = np.concatenate([q, qd, cube_pos, cube_quat, cube_linvel, cube_angvel, target_xy, self.prev_action],
                             dtype=np.float32)
        return obs

    def _get_reward(self):
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        cube_pos = self.data.qpos[adr_q + 4:adr_q + 7]
        cube_xy = cube_pos[:2]
        dist_to_target = float(np.linalg.norm(cube_xy - self.target_pos_xy))
        
        rewards = {}
        
        # 1. MAIN REWARD: Simple dense reward based on negative distance
        # The closer to target, the higher the reward
        max_dist = 0.3  # Maximum possible distance in our setup
        distance_reward = (max_dist - dist_to_target) / max_dist  # 0 to 1 scale
        rewards["distance"] = distance_reward * 10.0  # Scale to 0-10
        
        # 2. SUCCESS BONUS: Large reward for being very close to target
        success_threshold = 0.05  # 5cm threshold
        if dist_to_target < success_threshold:
            rewards["success"] = 50.0  # Big bonus for success
        else:
            rewards["success"] = 0.0
            
        # 3. CONTACT REWARD: Encourage finger to touch cube
        contact_reward = 0.0
        try:
            fingertip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "finger_tip")
            if fingertip_site_id != -1:
                fingertip_pos = self.data.site_xpos[fingertip_site_id][:2]
                fingertip_to_cube = np.linalg.norm(fingertip_pos - cube_xy)
                if fingertip_to_cube < 0.03:  # Very close contact
                    contact_reward = 5.0
                elif fingertip_to_cube < 0.05:  # Near contact
                    contact_reward = 2.0
        except Exception:
            pass
        rewards["contact"] = contact_reward
        
        # 4. SIMPLE PROGRESS REWARD
        if hasattr(self, 'prev_dist'):
            progress = max(0, self.prev_dist - dist_to_target)  # Only positive progress
            rewards["progress"] = progress * 20.0  # Reward for getting closer
        else:
            rewards["progress"] = 0.0
            
        # 5. Small penalties to encourage efficiency
        rewards["action_penalty"] = -0.01 * np.sum(np.square(self.prev_action))
        rewards["time_penalty"] = -0.01  # Small time penalty
        
        # Update for next step
        self.prev_dist = dist_to_target
        
        return rewards

    def _get_dones(self):
        # Episode ends when time limit reached
        terminated = self.current_step >= self.steps_per_episode
        
        # Early termination if task is successfully completed and maintained
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        cube_pos = self.data.qpos[adr_q + 4:adr_q + 7]
        cube_xy = cube_pos[:2]
        dist = float(np.linalg.norm(cube_xy - self.target_pos_xy))
        
        # Success: cube at target for several consecutive steps
        if dist < 0.03:
            self.reach_goal_timer += 1
        else:
            self.reach_goal_timer = 0
            
        # Terminate early if goal maintained for enough steps
        success_termination = self.reach_goal_timer >= 10  # Hold for 10 steps
        
        # Also terminate if cube goes too far out of bounds
        cube_out_of_bounds = (np.abs(cube_xy).max() > 0.3)  # 30cm from origin
        
        truncated = success_termination or cube_out_of_bounds
        
        return bool(terminated), bool(truncated)

    def _process_action(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self.default_joint_pos + self.action_scale * a
        target = np.clip(target, self.joint_lower, self.joint_upper)
        return target

    def _apply_action(self, joint_pos_target: np.ndarray):
        q = np.array([self.data.qpos[int(self.jnt_qposadr[jid])] for jid in self.actuated_joint_ids], dtype=np.float32)
        qd = np.array([self.data.qvel[int(self.jnt_dofadr[jid])] for jid in self.actuated_joint_ids], dtype=np.float32)
        torque = self.kp * (joint_pos_target - q) - self.kd * qd
        self.data.ctrl[:self.num_dof] = torque.astype(np.float32)

    def _check_goal_progress(self):
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        cube_pos = self.data.qpos[adr_q + 4:adr_q + 7]
        cube_xy = cube_pos[:2]
        if np.linalg.norm(cube_xy - self.target_pos_xy) < 0.05:
            self.reach_goal_timer += 1
        else:
            self.reach_goal_timer = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.reach_goal_timer = 0
        self.current_step = 0
        self.prev_action = np.zeros(self.num_dof, dtype=np.float32)
        # Randomize initial cube and robot positions for generalization
        adr_q = int(self.jnt_qposadr[self.cube_free_jid])
        mujoco.mj_resetData(self.model, self.data)
        # Randomize cube position (xy)
        cube_init_xy = self.cube_init_pos + np.random.uniform(-0.10, 0.10, size=2).astype(np.float32)  # ±10cm variation
        self.data.qpos[adr_q + 4:adr_q + 6] = cube_init_xy
        # Randomize robot joint positions within limits
        for i, jid in enumerate(self.actuated_joint_ids):
            low, high = self.joint_lower[i], self.joint_upper[i]
            self.data.qpos[int(self.jnt_qposadr[jid])] = float(np.random.uniform(low, high))
        # Set prev_dist for progress reward
        self.prev_dist = float(np.linalg.norm(self.data.qpos[adr_q + 4:adr_q + 6] - self.target_pos_xy))
        try:
            marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
            self.model.geom_pos[marker_id] = np.array([self.target_pos_xy[0], self.target_pos_xy[1], 0.05], dtype=np.float32)
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
