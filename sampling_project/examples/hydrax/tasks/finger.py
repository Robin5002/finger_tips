import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Finger(Task):
    """A 3DoF finger box push task"""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/finger/finger_edu_scene_cube.xml"
        )
        super().__init__(mj_model)

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Target position for the cube
        target_xy = jnp.array([0.20, 0.0])  # 20cm in x from cube init
        
        # Get cube position from sensor data
        cube_pos = state.sensordata[self.block_position_sensor:self.block_position_sensor+3]
        cube_xy = cube_pos[:2]  # x, y coordinates
        
        # Distance cost - penalize cube distance to target
        dist_cost = jnp.sum((cube_xy - target_xy) ** 2)
        
        # Control effort cost
        control_cost = 0.01 * jnp.sum(control ** 2)
        
        return dist_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""
        # Target position for the cube
        target_xy = jnp.array([0.20, 0.0])  # 20cm in x from cube init
        
        # Get cube position from sensor data
        cube_pos = state.sensordata[self.block_position_sensor:self.block_position_sensor+3]
        cube_xy = cube_pos[:2]  # x, y coordinates
        
        # Strongly penalize final distance to target
        return 10.0 * jnp.sum((cube_xy - target_xy) ** 2)
