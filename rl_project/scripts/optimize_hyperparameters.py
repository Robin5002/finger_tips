#!/usr/bin/env python3

"""
Optimized hyperparameter tuning for the proper pushing task using Optuna
"""

import sys
import os
import time
import json
import optuna
import numpy as np

# Add paths to Python path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)  # Add project root
sys.path.append(os.path.join(REPO_ROOT, "cleanrl/cleanrl"))  # Add cleanrl

import envs.register_envs

def objective(trial):
    """Objective function for Optuna optimization"""
    # Import training function
    from ppo_continuous_action import Args, train
    
    # Suggest hyperparameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "num_minibatches": trial.suggest_categorical("num_minibatches", [4, 8, 16]),
        "update_epochs": trial.suggest_categorical("update_epochs", [5, 10, 20]),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
        "num_steps": trial.suggest_categorical("num_steps", [1024, 2048]),
        "vf_coef": trial.suggest_float("vf_coef", 0.25, 1.0),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0]),
        "clip_coef": trial.suggest_float("clip_coef", 0.1, 0.3),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999)
    }
    
    # Create Args with tuned parameters
    args = Args(
        env_id="FingerPush-v0",
        exp_name=f"finger_push_optuna_trial_{trial.number}",
        total_timesteps=50_000,  # Short trials for quick optimization
        num_envs=1,
        track=False,  # Disable wandb for speed
        capture_video=False,  # Disable videos for speed
        save_model=False,  # Don't save models during tuning
        **params  # Apply suggested hyperparameters
    )
    
    try:
        # Run training and get average reward
        avg_reward = train(args)
        return float(avg_reward)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -float('inf')  # Return very low score for failed trials

def run_hyperparameter_optimization():
    """Run hyperparameter optimization with Optuna"""
    print("🎯 OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print("🏋️  Optimizing PPO for proper cube pushing task")
    print("📦 Cube starts at [0.0, -0.15], target at [0.0, 0.15]")
    print("🎮 Each trial: 50k steps (quick optimization)")
    print()
    
    # Create study
    study_name = f"finger_push_proper_optimization_{int(time.time())}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=10,
            interval_steps=5
        )
    )
    
    print(f"📊 Study name: {study_name}")
    print("🔍 Starting optimization...")
    print()
    
    try:
        # Run optimization
        study.optimize(objective, n_trials=15, timeout=7200)  # 15 trials, max 2 hours
        
        print("✅ OPTIMIZATION COMPLETED!")
        print("=" * 40)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"🏆 Best reward: {best_value:.2f}")
        print("🎯 Best parameters:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        
        # Save best parameters
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        print()
        print("💾 Best parameters saved to best_params.json")
        print("🚀 Ready to start optimized training!")
        
        # Show summary statistics
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) > 1:
            rewards = [t.value for t in completed_trials]
            print()
            print("📈 OPTIMIZATION SUMMARY:")
            print(f"   Completed trials: {len(completed_trials)}")
            print(f"   Average reward: {np.mean(rewards):.2f}")
            print(f"   Best reward: {np.max(rewards):.2f}")
            print(f"   Improvement: {(best_value - np.mean(rewards)):.2f}")
        
        return best_params
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        print("💡 Saving best parameters found so far...")
        
        if study.best_trial is not None:
            best_params = study.best_params
            with open("best_params.json", "w") as f:
                json.dump(best_params, f, indent=2)
            print("💾 Partial results saved to best_params.json")
            return best_params
        else:
            print("❌ No completed trials to save")
            return None
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

if __name__ == "__main__":
    best_params = run_hyperparameter_optimization()
    
    if best_params:
        print("\n🎉 SUCCESS! Ready for optimized training:")
        print("   Run: python scripts/train_finger_ppo.py")
        print("   It will automatically load best_params.json")
    else:
        print("\n⚠️ No optimized parameters available")
        print("   Will use default parameters for training")
