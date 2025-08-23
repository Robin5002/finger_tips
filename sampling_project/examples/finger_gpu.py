import argparse
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.finger import Finger
import os

# Configure JAX for GPU support with CPU fallback
def setup_jax_device():
    """Setup JAX to use GPU if available, otherwise fallback to CPU."""
    try:
        # Try to detect available devices
        devices = jax.devices()
        print(f"Available devices: {devices}")
        
        # Check for GPU devices (check platform instead of device_kind)
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        
        if gpu_devices:
            print(f"✓ GPU detected: {gpu_devices[0]}")
            print(f"✓ GPU name: {gpu_devices[0].device_kind}")
            print(f"✓ Using GPU for acceleration")
            return "gpu"
        else:
            print("! No GPU detected, using CPU")
            os.environ["JAX_PLATFORMS"] = "cpu"
            return "cpu"
    except Exception as e:
        print(f"! Error detecting GPU, falling back to CPU: {e}")
        os.environ["JAX_PLATFORMS"] = "cpu"
        return "cpu"

# Setup the device
device_type = setup_jax_device()

"""
Run an interactive simulation of the finger push task using MPPI controller,
with logging & plotting of rollout costs.
"""


def main(gui=True, num_steps=20):
    # Define the task (includes sensors and costs)
    task = Finger()

    # ------------------ Set up MPPI Controller ------------------ #
    # Adjust parameters based on device type for optimal performance
    if device_type == "gpu":
        # Use moderate parameters for GPU acceleration
        plan_horizon = 8
        num_samples = 50
        print("✓ Using GPU-optimized parameters for faster training")
    else:
        # Use lower parameters for CPU to maintain reasonable speed
        plan_horizon = 3
        num_samples = 10
        print("! Using CPU-optimized parameters")
    
    ctrl = MPPI(
        task=task,
        plan_horizon=plan_horizon,        # planning horizon (timesteps)
        num_samples=num_samples,          # number of sampled trajectories
        temperature=1.0,                  # temperature parameter for weighting
        noise_level=0.3,                  # exploration noise std
        #u_min=-1.0,                      # control lower bound
        #u_max=1.0,                       # control upper bound
        seed=0,
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.001
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50

    mj_data = mujoco.MjData(mj_model)
    # Initial qpos for finger + cube
    mj_data.qpos = np.array([0, 0, -0.15, 0.1, -0.05, 0.05, 1, 0, 0, 0], dtype=np.float64)
    mujoco.mj_forward(mj_model, mj_data)  # Initialize the simulation state

    # ------------------ Logging Setup ------------------ #
    rollout_costs = []
    running_costs = []
    terminal_costs = []

    # Initialize controller policy parameters
    policy_params = ctrl.init_params()

    # Custom rollout loop with logging
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")
        
        # Convert to JAX state first
        state = mjx.put_data(mj_model, mj_data)

        # Update policy parameters and get control action
        policy_params, _ = ctrl.optimize(state, policy_params)
        u = ctrl.get_action(policy_params, state.time)

        # Step simulation with chosen control
        mj_data.ctrl[:] = u
        mujoco.mj_step(mj_model, mj_data)

        # Log costs
        rc = task.running_cost(state, u)
        tc = task.terminal_cost(state)
        total = rc + tc

        running_costs.append(float(rc))
        terminal_costs.append(float(tc))
        rollout_costs.append(float(total))

        if gui:
            mujoco.mj_forward(mj_model, mj_data)

    # ------------------ Plot Results ------------------ #
    plt.figure(figsize=(10, 6))
    plt.plot(running_costs, label="Running Cost")
    plt.plot(terminal_costs, label="Terminal Cost")
    plt.plot(rollout_costs, label="Total Cost")
    plt.xlabel("Timestep")
    plt.ylabel("Cost")
    plt.title("MPPI Rollout Cost Breakdown")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI (headless run with logging only)")
    parser.add_argument("--steps", type=int, default=20, help="Number of timesteps to simulate")
    args = parser.parse_args()

    main(gui=not args.no_gui, num_steps=args.steps)
