#!/usr/bin/env python
import mujoco
from mujoco import mjx
import numpy as np
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import argparse
import os
import random
import sys
from tqdm import trange
import imageio
import json
import pandas as pd
import seaborn as sns

from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.finger import Finger

# Configure JAX for GPU support with CPU fallback
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
    
device_type = setup_jax_device()

# Modified Finger class with configurable cost weights
class ConfigurableFinger(Finger):
    """Extends Finger class with configurable cost weights."""
    
    def __init__(self, control_weight=0.01, terminal_weight=10.0):
        super().__init__()
        self.control_weight = control_weight
        self.terminal_weight = terminal_weight
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """
        Running cost ℓ(xₜ, uₜ).
        Penalize cube distance to target at each step + control effort.
        """
        cube_pos = state.qpos[-3:]  # xyz of cube
        cube_xy = cube_pos[:2]

        dist_cost = jnp.sum((cube_xy - self.target_xy) ** 2)
        control_cost = self.control_weight * jnp.sum(control ** 2)

        return dist_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """
        Terminal cost ℓ_T(x_T).
        Strongly penalize final cube distance to target.
        """
        cube_pos = state.qpos[-3:]
        cube_xy = cube_pos[:2]

        return self.terminal_weight * jnp.sum((cube_xy - self.target_xy) ** 2)

def run_simulation(params, num_steps=40, save_video=True, verbose=False):
    """
    Run simulation with given parameters and return final distance to target.
    
    Args:
        params: Dictionary with parameters
        num_steps: Number of simulation steps
        save_video: Whether to save a video of the best run
        verbose: Whether to print progress information
    
    Returns:
        final_distance: Distance between final cube position and target
    """
    # Extract parameters
    control_weight = params['control_weight']
    terminal_weight = params['terminal_weight'] 
    temperature = params['temperature']
    
    
    # Define the task with custom weights
    task = ConfigurableFinger(
        control_weight=control_weight,
        terminal_weight=terminal_weight
    )
    
    # Set up MPPI Controller with fixed plan_horizon and noise_level
    if device_type == "gpu":
        plan_horizon = 8
        num_samples = 50
    else:
        plan_horizon = 3
        num_samples = 10
    
    ctrl = MPPI(
        task=task,
        plan_horizon=plan_horizon,
        num_samples=num_samples,
        temperature=temperature,
        noise_level=0.3,  # Fixed as requested
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

    # Initialize controller policy parameters
    policy_params = ctrl.init_params()

    # Video setup if needed
    video_frames = []
    if save_video:
        width, height = 640, 480
        renderer = mujoco.Renderer(mj_model, width=width, height=height)

    # Simulation loop with timing info
    sim_start_time = time.time()
    costs = []
    
    print(f"\n--- STARTING SIMULATION ---")
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime())}")
    print(f"Parameters: control_weight={control_weight:.4f}, terminal_weight={terminal_weight:.4f}, temperature={temperature:.4f}")
    print(f"Steps: {num_steps}")
    
    # Interactive progress display setup
    progress_lines = 6  # Number of lines in our progress display
    
    # Run simulation with interactive updates
    for step in range(num_steps):
        step_start = time.time()
        step_timestamp = time.strftime('%H:%M:%S')
        
        # Convert to JAX state
        state = mjx.put_data(mj_model, mj_data)

        # Update policy parameters and get control action
        policy_params, _ = ctrl.optimize(state, policy_params)
        u = ctrl.get_action(policy_params, state.time)

        # Step simulation with chosen control
        mj_data.ctrl[:] = u
        mujoco.mj_step(mj_model, mj_data)
        
        # Calculate costs for monitoring
        rc = task.running_cost(state, u)
        tc = task.terminal_cost(state)
        total_cost = float(rc) + float(tc)
        costs.append(total_cost)

        # Save video frame if requested
        if save_video:
            renderer.update_scene(mj_data)
            frame = renderer.render()
            video_frames.append(frame)
            
        # Calculate timing and progress information
        cube_pos = state.qpos[-3:]
        cube_xy = cube_pos[:2]
        current_dist = float(jnp.sqrt(jnp.sum((cube_xy - task.target_xy) ** 2)))
        step_time = time.time() - step_start
        elapsed = time.time() - sim_start_time
        remaining = (num_steps - step - 1) * (elapsed / (step + 1))
        eta_time = time.localtime(time.time() + remaining)
        
        # Clear previous progress display (go up 'progress_lines' lines)
        if step > 0:
            # Move cursor up to overwrite previous progress lines
            sys.stdout.write(f"\033[{progress_lines}A")
            sys.stdout.flush()
        
        # Print updated progress information (will replace previous lines)
        print(f"Step {step+1}/{num_steps} [{step_timestamp}]" + " " * 20)
        print(f"  Distance to target: {current_dist:.4f} m" + " " * 20)
        print(f"  Current cost: {total_cost:.4f} (running: {float(rc):.4f}, terminal: {float(tc):.4f})" + " " * 20)
        print(f"  Time: {step_time:.3f}s per step, {elapsed:.1f}s elapsed" + " " * 20)
        print(f"  Progress: {(step+1)/num_steps*100:.1f}%, ETA: {time.strftime('%H:%M:%S', eta_time)}" + " " * 20)
        print(f"  {'-'*30}" + " " * 20)
        sys.stdout.flush()
    # Calculate final distance to target
    final_state = mjx.put_data(mj_model, mj_data)
    cube_pos = final_state.qpos[-3:]
    cube_xy = cube_pos[:2]
    target_xy = task.target_xy
    final_distance = float(jnp.sqrt(jnp.sum((cube_xy - target_xy) ** 2)))
    
    # Print simulation summary
    sim_time = time.time() - sim_start_time
    cost_stats = {
        'mean': np.mean(costs),
        'min': np.min(costs),
        'max': np.max(costs),
        'final': costs[-1] if costs else 0
    }
    
    print(f"\n{'='*50}")
    print(f" SIMULATION SUMMARY")
    print(f"{'='*50}")
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(sim_start_time))}")
    print(f"End time:   {time.strftime('%H:%M:%S', time.localtime())}")
    print(f"Duration:   {sim_time:.2f}s ({sim_time/60:.2f} minutes)")
    print(f"\nResults:")
    print(f"  Final distance to target: {final_distance:.4f} m")
    print(f"  Final position: [{cube_xy[0]:.4f}, {cube_xy[1]:.4f}]")
    print(f"  Target position: [{target_xy[0]:.4f}, {target_xy[1]:.4f}]")
    print(f"\nCost statistics:")
    print(f"  Mean cost:  {cost_stats['mean']:.4f}")
    print(f"  Min cost:   {cost_stats['min']:.4f}")
    print(f"  Max cost:   {cost_stats['max']:.4f}")
    print(f"  Final cost: {cost_stats['final']:.4f}")
    print(f"\nParameters:")
    print(f"  control_weight: {control_weight:.4f}")
    print(f"  terminal_weight: {terminal_weight:.4f}")
    print(f"  temperature: {temperature:.4f}")
    print(f"{'='*50}")
    
    # Save video if requested and this is the best run
    if save_video:
        # Ensure videos directory exists
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/finger_simulation_{params['id']}.mp4"
        imageio.mimsave(video_path, video_frames, fps=24)
        print(f"Video saved to {video_path}")
    
    return final_distance

def random_search(n_trials=20, save_best_video=True):
    """
    Perform random search over parameters to find optimal values.
    
    Args:
        n_trials: Number of random trials
        save_best_video: Whether to save video of best run
        
    Returns:
        best_params: Dictionary with best parameters
        best_distance: Best (lowest) final distance achieved
    """
    # Parameter ranges
    param_ranges = {
        'control_weight': (0.001, 0.1),  # Control cost weight
        'terminal_weight': (10.0, 200.0),  # Terminal cost weight
        'temperature': (0.1, 2.0),  # MPPI temperature
    }
    
    results = []
    best_distance = float('inf')
    best_params = None
    
    # Track overall time
    overall_start_time = time.time()
    
    # Interactive trial display setup
    trial_progress_lines = 2
    
    print(f"\n{'='*60}")
    print(f" RANDOM SEARCH: Starting with {n_trials} trials")
    print(f" Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Parameter ranges:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: [{min_val}, {max_val}]")
    print(f"{'='*60}\n")
    
    for i in range(n_trials):
        # Track time for this trial
        trial_start_time = time.time()
        
        # Clear previous overall progress if not first trial
        if i > 0:
            sys.stdout.write(f"\033[{trial_progress_lines}A")
            sys.stdout.flush()
        
        # Display overall progress information (will be updated each trial)
        elapsed_total = time.time() - overall_start_time if i > 0 else 0
        avg_time_per_trial = elapsed_total / max(1, i)
        remaining_trials = n_trials - i
        est_remaining_time = remaining_trials * avg_time_per_trial
        eta = time.localtime(time.time() + est_remaining_time)
        
        print(f"Overall progress: {i}/{n_trials} trials complete ({i/n_trials*100:.1f}%)" + " " * 30)
        print(f"Elapsed: {elapsed_total/60:.1f} minutes, Estimated completion: {time.strftime('%H:%M:%S', eta)}" + " " * 30)
        sys.stdout.flush()
        
        # Sample random parameters
        params = {
            'id': i,
            'control_weight': random.uniform(*param_ranges['control_weight']),
            'terminal_weight': random.uniform(*param_ranges['terminal_weight']),
            'temperature': random.uniform(*param_ranges['temperature']),
        }
        
        # Print trial header (new lines since this shouldn't be overwritten)
        print(f"\n\n{'='*20} TRIAL {i+1}/{n_trials} {'='*20}")
        print(f"Trial start time: {time.strftime('%H:%M:%S')}")
        print("Testing parameters:")
        for key, value in params.items():
            if key != 'id':
                print(f"  {key}: {value:.4f}")
        
        # Run simulation with these parameters
        final_distance = run_simulation(params, save_video=False, verbose=False)
        
        # Calculate and display time for this trial
        trial_time = time.time() - trial_start_time
        elapsed_total = time.time() - overall_start_time
        avg_time_per_trial = elapsed_total / (i + 1)
        remaining_trials = n_trials - (i + 1)
        est_remaining_time = remaining_trials * avg_time_per_trial
        
        # Store results
        params['final_distance'] = final_distance
        params['trial_time'] = trial_time
        results.append(params)
        
        # Print trial results (won't be overwritten)
        print(f"\nTrial {i+1} result: distance = {final_distance:.4f}")
        print(f"Trial time: {trial_time:.2f}s | Avg per trial: {avg_time_per_trial:.2f}s")
        print(f"Elapsed: {elapsed_total/60:.2f}min | Est. remaining: {est_remaining_time/60:.2f}min")
        
        # Update best parameters if this run is better
        if final_distance < best_distance:
            best_distance = final_distance
            best_params = params.copy()
            print(f"\n=== NEW BEST FOUND ===")
            print(f"Distance: {best_distance:.4f}")
            for key, value in best_params.items():
                if key not in ['id', 'final_distance', 'trial_time']:
                    print(f"{key}: {value:.4f}")
            print("=====================")
    
    # Calculate total search time
    total_search_time = time.time() - overall_start_time
    print(f"\nTotal search completed in {total_search_time/60:.2f} minutes")
    
    # Run again with best parameters to save video
    if save_best_video and best_params is not None:
        print(f"\nRunning best parameters to save video:")
        for key, value in best_params.items():
            if key not in ['id', 'final_distance', 'trial_time']:
                print(f"  {key}: {value:.4f}")
        run_simulation(best_params, save_video=True, verbose=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save DataFrame to CSV
    df.to_csv('random_search_results.csv', index=False)
    
    # Create and save scatter plots for each parameter vs final distance
    os.makedirs("plots", exist_ok=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Create individual scatter plots for each parameter
    for param in ['control_weight', 'terminal_weight', 'temperature']:
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(x=param, y='final_distance', data=df, alpha=0.7, s=100)
        
        # Highlight the best point
        best_point = df.loc[df['final_distance'] == df['final_distance'].min()]
        sns.scatterplot(x=param, y='final_distance', data=best_point, 
                        color='red', s=200, marker='X', label='Best')
        
        # Add trend line
        sns.regplot(x=param, y='final_distance', data=df, scatter=False, 
                   line_kws={"color":"red", "alpha":0.5})
        
        plt.title(f'Effect of {param} on Final Distance to Target')
        plt.xlabel(param)
        plt.ylabel('Final Distance')
        plt.tight_layout()
        plt.savefig(f'plots/{param}_vs_distance.png')
        plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_columns = ['control_weight', 'terminal_weight', 'temperature', 'final_distance']
    corr = df[corr_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Parameter Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    # Save all results to JSON file as well
    with open('random_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_params, best_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Number of random trials")
    args = parser.parse_args()
    
    print(f"Starting random search with {args.trials} trials")
    best_params, best_distance = random_search(n_trials=args.trials)
    
    print("\n=== BEST PARAMETERS ===")
    print(f"Final distance to target: {best_distance:.4f}")
    print(f"Control weight: {best_params['control_weight']:.4f}")
    print(f"Terminal weight: {best_params['terminal_weight']:.4f}")
    print(f"Temperature: {best_params['temperature']:.4f}")
    print("=======================")
    
    print("\n=== OUTPUTS ===")
    print("1. Best parameter video saved to videos/finger_simulation_*.mp4")
    print("2. Parameter scatter plots saved to plots/ directory:")
    print("   - control_weight_vs_distance.png")
    print("   - terminal_weight_vs_distance.png")
    print("   - temperature_vs_distance.png")
    print("   - correlation_heatmap.png")
    print("3. Raw data saved to:")
    print("   - random_search_results.csv (tabular data)")
    print("   - random_search_results.json (JSON format)")
    print("================")
