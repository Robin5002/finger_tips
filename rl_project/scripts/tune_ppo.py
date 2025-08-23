import optuna
from cleanrl_utils.tuner import Tuner
import envs.register_envs
import time

# Generate unique study name with timestamp
study_name = f"finger_ppo_tune_{int(time.time())}"

# Fixed tuner configuration for your FingerPush environment
tuner = Tuner(
    script="../cleanrl/cleanrl/ppo_continuous_action.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    study_name=study_name,
    wandb_kwargs={
        "project": "rl_finger_ppo_tune",
        "entity": "mdabu-hanif-university-of-trento",  # Your wandb username
    },
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "FingerPush-v0": [0, 100],  # Adjusted target score range
    },
    params_fn=lambda trial: {
        "env-id": "FingerPush-v0",
        "learning-rate": trial.suggest_float("learning-rate", 1e-4, 1e-3, log=True),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [4, 8, 16, 32]),
        "update-epochs": trial.suggest_categorical("update-epochs", [5, 10, 20]),
        "ent-coef": trial.suggest_float("ent-coef", 0.001, 0.1, log=True),
        "num-steps": trial.suggest_categorical("num-steps", [512, 1024, 2048]),
        "vf-coef": trial.suggest_float("vf-coef", 0.25, 1.0),
        "max-grad-norm": trial.suggest_categorical("max-grad-norm", [0.5, 1.0]),
        "clip-coef": trial.suggest_float("clip-coef", 0.1, 0.3),
        "total-timesteps": 100000,  # Shorter for faster tuning
        "num-envs": 1,  # Fixed to 1 for stability
        "track": True,
        "capture-video": False,  # Disable videos during tuning for speed
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)

if __name__ == "__main__":
    print("🔍 Starting hyperparameter tuning for FingerPush-v0...")
    print(f"📊 Study name: {study_name}")
    
    try:
        tuner.tune(
            num_trials=50,  # Reduced for faster tuning
            num_seeds=2,    # Reduced for faster tuning
        )
        print("✅ Hyperparameter tuning completed!")
        
        # Save best parameters to JSON for easy loading
        import json
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_name}.db")
        best_params = study.best_params
        
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"💾 Best parameters saved to best_params.json")
        print(f"🏆 Best score: {study.best_value}")
        print(f"📊 Best params: {best_params}")
        
    except Exception as e:
        print(f"❌ Error during tuning: {e}")
        print("💡 Trying to load existing study if available...")
        try:
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_name}.db")
            print(f"📊 Found existing study with {len(study.trials)} trials")
            if study.best_trial:
                best_params = study.best_params
                with open("best_params.json", "w") as f:
                    json.dump(best_params, f, indent=2)
                print(f"💾 Best parameters saved to best_params.json")
                print(f"🏆 Best score: {study.best_value}")
        except:
            print("❌ Could not load existing study either")
