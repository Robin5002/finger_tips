import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Finger(Task):
    """A 3DoF finger box push task."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/finger/finger_edu_scene_cube.xml"
        )
        super().__init__(mj_model)

        # ----------------- Sensors -----------------
        # We'll observe:
        # - Cube position (XY)
        # - Target position (fixed)
        # - Joint controls
        self.target_xy = jnp.array([0.20, 0.0])  # 20cm in x from cube init (matches project Q3a)
        
    # ----------------------------------------------------- #
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """
        Running cost ℓ(xₜ, uₜ).
        Penalize cube distance to target at each step + control effort.
        """
        # Cube pos from free joint (last 3 entries of qpos of free joint)
        cube_pos = state.qpos[-3:]  # xyz of cube
        cube_xy = cube_pos[:2]

        dist_cost = jnp.sum((cube_xy - self.target_xy) ** 2)
        control_cost = 0.01 * jnp.sum(control ** 2)

        return dist_cost + control_cost

    # ----------------------------------------------------- #
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """
        Terminal cost ℓ_T(x_T).
        Strongly penalize final cube distance to target.
        """
        cube_pos = state.qpos[-3:]
        cube_xy = cube_pos[:2]

        return 10.0 * jnp.sum((cube_xy - self.target_xy) ** 2)
