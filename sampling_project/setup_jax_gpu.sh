#!/bin/bash

# JAX GPU Setup Script for GTX 1050 (CUDA 11.x)
# Run this script to install compatible JAX packages

echo "=========================================="
echo "JAX GPU Setup for GTX 1050"
echo "=========================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "sampling_mujoco_playground" ]]; then
    echo "⚠️  Activating sampling_mujoco_playground environment..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate sampling_mujoco_playground
    
    if [[ "$CONDA_DEFAULT_ENV" != "sampling_mujoco_playground" ]]; then
        echo "❌ Failed to activate environment. Please run:"
        echo "conda activate sampling_mujoco_playground"
        echo "Then run this script again."
        exit 1
    fi
fi

echo "✅ Environment: $CONDA_DEFAULT_ENV"

# Step 1: Uninstall existing JAX packages
echo "Step 1: Uninstalling existing JAX packages..."
pip uninstall -y jax jaxlib jax-cuda12 jax-cuda11 jaxopt 2>/dev/null || true
pip uninstall -y jax jaxlib --force-reinstall 2>/dev/null || true

# Step 2: Clear cache
echo "Step 2: Clearing pip cache..."
pip cache purge

# Step 3: Install JAX with CUDA 11 support
echo "Step 3: Installing JAX with CUDA 11.x support..."

# Try primary installation method
echo "Trying primary installation method..."
pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
if [ $? -eq 0 ]; then
    pip install jax==0.4.23
    INSTALL_SUCCESS=true
else
    echo "Primary method failed, trying alternative..."
    pip install --upgrade 'jax[cuda11_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    INSTALL_SUCCESS=$?
fi

# Step 4: Verify installation
echo "Step 4: Verifying JAX GPU setup..."
python -c "
import jax
import jax.numpy as jnp

print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())

# Test GPU functionality
try:
    x = jnp.array([1, 2, 3, 4, 5])
    result = jnp.sum(x ** 2)
    gpu_count = len([d for d in jax.devices() if d.device_kind == 'gpu'])
    
    print(f'GPU devices found: {gpu_count}')
    if gpu_count > 0:
        print('✅ SUCCESS: JAX can use GPU!')
    else:
        print('❌ WARNING: No GPU detected')
except Exception as e:
    print(f'❌ ERROR: {e}')
"

echo "=========================================="
echo "Setup complete!"
echo "You can now test your finger.py script."
echo "=========================================="
