#!/usr/bin/env python3
"""
Simple hyperparameter tuning for FingerPush-v0 using Optuna.
This script runs the training script as a subprocess to avoid import issues.
"""

import optuna
import subprocess
import json
import os
import re
import tempfile
import sys
from typing import Dict, Any


def run_training_trial(params: Dict[str, Any], trial_number: int) -> float:
    """Run a single training trial with given parameters."""
    
    # Create a temporary script for this trial
    script_content = f"""
import sys
import os
import time
import json

# Add cleanrl to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))

import envs.register_envs

# Build command line arguments
args = [
    "python3", "cleanrl/cleanrl/ppo_continuous_action.py",
    "--env-id", "FingerPush-v0",
    "--exp-name", "optuna_trial_{trial_number}",
    "--total-timesteps", "50000",  # Shorter for faster tuning
    "--num-envs", "1",
    "--learning-rate", "{params['learning_rate']}",
    "--num-minibatches", "{params['num_minibatches']}",
    "--update-epochs", "{params['update_epochs']}",
    "--ent-coef", "{params['ent_coef']}",
    "--num-steps", "{params['num_steps']}",
    "--vf-coef", "{params['vf_coef']}",
    "--max-grad-norm", "{params['max_grad_norm']}",
    "--clip-coef", "{params['clip_coef']}",
    "--track", "False",  # Disable wandb during tuning
    "--capture-video", "False",
]

# Run the training
import subprocess
result = subprocess.run(args, capture_output=True, text=True, cwd=os.getcwd())
print("Training completed with return code:", result.returncode)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    print("STDOUT:", result.stdout)
"""
    
    try:
        # Write and execute the script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        # Run the script
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, 
                              cwd="/home/robin/Desktop/robotics_tutorials-fomr_24_25/rl_and_sampling_projects/rl_project",
                              timeout=600)  # 10 minute timeout
        
        # Clean up
        os.unlink(temp_script)
        
        if result.returncode != 0:
            print(f"Trial {trial_number} failed:")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return -100.0  # Return very low score for failed trials
            
        # Extract the final episodic return from output
        # Look for pattern like "global_step=X, episodic_return=Y"
        output = result.stdout
        returns = re.findall(r'episodic_return=([-+]?\d*\.?\d+)', output)
        
        if returns:
            # Take the average of the last few returns
            last_returns = [float(r) for r in returns[-5:]]  # Last 5 episodes
            avg_return = sum(last_returns) / len(last_returns)
            print(f"Trial {trial_number}: Average return = {avg_return:.3f}")
            return avg_return
        else:
            print(f"Trial {trial_number}: No returns found in output")
            return -50.0
            
    except subprocess.TimeoutExpired:
        print(f"Trial {trial_number}: Timeout")
        return -200.0
    except Exception as e:
        print(f"Trial {trial_number}: Exception {e}")
        return -300.0


def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    
    # Suggest hyperparameters
    params = {{
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "num_minibatches": trial.suggest_categorical("num_minibatches", [4, 8, 16]),
        "update_epochs": trial.suggest_categorical("update_epochs", [5, 10]),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.05, log=True),
        "num_steps": trial.suggest_categorical("num_steps", [512, 1024]),
        "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.0),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0]),
        "clip_coef": trial.suggest_float("clip_coef", 0.1, 0.3),
    }}
    
    return run_training_trial(params, trial.number)


def main():
    """Main function to run hyperparameter tuning."""
    
    print("🔍 Starting hyperparameter tuning for FingerPush-v0...")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="finger_ppo_tune",
        storage="sqlite:///finger_ppo_tune.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        sampler=optuna.samplers.TPESampler()
    )
    
    # Run optimization
    n_trials = 10  # Start with fewer trials
    study.optimize(objective, n_trials=n_trials)
    
    print("✅ Hyperparameter tuning completed!")
    print(f"🏆 Best score: {{study.best_value:.3f}}")
    print(f"📊 Best params: {{study.best_params}}")
    
    # Save best parameters to JSON
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("💾 Best parameters saved to best_params.json")
    
    return study.best_params, study.best_value


if __name__ == "__main__":
    try:
        best_params, best_score = main()
        
        print("\\n" + "="*50)
        print("🎯 HYPERPARAMETER TUNING RESULTS")
        print("="*50)
        print(f"Best Score: {{best_score:.3f}}")
        print("Best Parameters:")
        for key, value in best_params.items():
            print(f"  {{key}}: {{value}}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\\n⚠️ Tuning interrupted by user")
    except Exception as e:
        print(f"\\n❌ Tuning failed: {{e}}")
"""
