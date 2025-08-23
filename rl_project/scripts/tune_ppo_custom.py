import optuna
import subprocess
import json
import os
import sys
import tempfile
from typing import Dict, Any

# Add cleanrl to path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))

import envs.register_envs
from ppo_continuous_action import Args, train


def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    
    # Suggest hyperparameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "num_minibatches": trial.suggest_categorical("num_minibatches", [4, 8, 16, 32]),
        "update_epochs": trial.suggest_categorical("update_epochs", [5, 10, 20]),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
        "num_steps": trial.suggest_categorical("num_steps", [512, 1024, 2048]),
        "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.0),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0]),
        "clip_coef": trial.suggest_float("clip_coef", 0.1, 0.3),
    }
    
    # Create Args with suggested parameters
    args = Args(
        env_id="FingerPush-v0",
        exp_name=f"optuna_trial_{trial.number}",
        total_timesteps=100000,  # Shorter for faster tuning
        num_envs=1,
        track=False,  # Disable wandb during tuning
        capture_video=False,
        **params
    )
    
    try:
        # Run training
        avg_reward = train(args)
        return avg_reward
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return -float('inf')  # Return very low score for failed trials


def run_hyperparameter_tuning(n_trials: int = 20):
    """Run Optuna hyperparameter tuning."""
    
    print("🔍 Starting hyperparameter tuning for FingerPush-v0...")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="finger_ppo_tune",
        storage=f"sqlite:///finger_ppo_tune.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        sampler=optuna.samplers.TPESampler()
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    print("✅ Hyperparameter tuning completed!")
    print(f"🏆 Best score: {study.best_value:.3f}")
    print(f"📊 Best params: {study.best_params}")
    
    # Save best parameters to JSON
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("💾 Best parameters saved to best_params.json")
    
    return study.best_params, study.best_value


if __name__ == "__main__":
    # Run tuning
    best_params, best_score = run_hyperparameter_tuning(n_trials=20)
    
    print("\n" + "="*50)
    print("🎯 HYPERPARAMETER TUNING RESULTS")
    print("="*50)
    print(f"Best Score: {best_score:.3f}")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*50)
