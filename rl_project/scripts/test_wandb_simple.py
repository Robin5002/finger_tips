#!/usr/bin/env python3

"""
Simple WandB test to verify logging works
"""

import wandb
import numpy as np
import time

def test_wandb():
    # Initialize WandB
    run = wandb.init(
        project="mujoco_tutorial",
        entity="mdabu-hanif-university-of-trento",
        name="simple_test",
        config={
            "test": True,
            "lr": 0.001,
            "batch_size": 32
        }
    )
    
    print("🔧 WandB initialized successfully!")
    print(f"   Run: {run.name}")
    print(f"   URL: {run.url}")
    
    # Log some simple metrics
    for i in range(10):
        wandb.log({
            "step": i,
            "loss": np.random.random() * 0.1 + 0.05,
            "accuracy": np.random.random() * 0.2 + 0.8,
            "learning_rate": 0.001 * (0.9 ** i)
        })
        print(f"   Logged step {i}")
        time.sleep(0.1)
    
    print("✅ WandB test completed!")
    print(f"   View run at: {run.url}")
    
    wandb.finish()

if __name__ == "__main__":
    test_wandb()
