#!/usr/bin/env python3
"""
JAX GPU Setup Script for GTX 1050 (Compute Capability 6.1)
This script uninstalls existing JAX packages and installs compatible versions for CUDA 11.x
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and print its output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"✓ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def check_conda_env():
    """Check if we're in the correct conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"Current conda environment: {conda_env}")
    
    if conda_env != 'sampling_mujoco_playground':
        print("\n⚠️  WARNING: You should activate the sampling_mujoco_playground environment first!")
        print("Run: conda activate sampling_mujoco_playground")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    return True

def main():
    print("JAX GPU Setup for GTX 1050 (Compute Capability 6.1)")
    print("This will install JAX with CUDA 11.x support")
    
    # Check environment
    if not check_conda_env():
        print("Exiting...")
        sys.exit(1)
    
    # Step 1: Uninstall existing JAX packages
    uninstall_commands = [
        "pip uninstall -y jax jaxlib jax-cuda12 jax-cuda11 jaxopt",
        "pip uninstall -y jax jaxlib --force-reinstall"  # Force removal
    ]
    
    for cmd in uninstall_commands:
        run_command(cmd, "Uninstalling existing JAX packages")
    
    # Step 2: Clear pip cache to avoid conflicts
    run_command("pip cache purge", "Clearing pip cache")
    
    # Step 3: Install compatible JAX version for CUDA 11.x
    # For GTX 1050 (Compute Capability 6.1), we need CUDA 11.x compatible versions
    # JAX 0.4.x series has good CUDA 11 support
    
    install_commands = [
        # Install specific jaxlib version with CUDA 11 support
        "pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        # Install corresponding JAX version
        "pip install jax==0.4.23"
    ]
    
    success = True
    for cmd in install_commands:
        if not run_command(cmd, f"Installing JAX with CUDA 11 support"):
            success = False
            break
    
    if not success:
        print("\n⚠️  Primary installation failed, trying alternative approach...")
        # Alternative: Install latest stable with CUDA 11 support
        alt_commands = [
            "pip install --upgrade 'jax[cuda11_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        ]
        
        for cmd in alt_commands:
            run_command(cmd, "Installing JAX (alternative method)")
    
    # Step 4: Verify installation
    print(f"\n{'='*60}")
    print("VERIFICATION: Testing JAX GPU detection")
    print(f"{'='*60}")
    
    verification_script = '''
import jax
import jax.numpy as jnp
import os

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("Default device:", jax.default_device())

# Check for CUDA
try:
    # Try to create an array on GPU
    x = jnp.array([1, 2, 3, 4, 5])
    print("Array device:", x.device())
    
    # Check if computation works on GPU
    result = jnp.sum(x ** 2)
    print("GPU computation result:", result)
    print("Result device:", result.device())
    
    # Count GPU devices
    gpu_count = len([d for d in jax.devices() if d.device_kind == 'gpu'])
    print(f"GPU devices found: {gpu_count}")
    
    if gpu_count > 0:
        print("✅ SUCCESS: JAX successfully detects and can use GPU!")
    else:
        print("❌ WARNING: No GPU devices detected by JAX")
        
except Exception as e:
    print(f"❌ ERROR during GPU verification: {e}")
'''
    
    # Write verification script to temp file and run it
    with open('/tmp/jax_verification.py', 'w') as f:
        f.write(verification_script)
    
    run_command("python /tmp/jax_verification.py", "Verifying JAX GPU setup")
    
    # Clean up
    try:
        os.remove('/tmp/jax_verification.py')
    except:
        pass
    
    print(f"\n{'='*60}")
    print("SETUP COMPLETE!")
    print(f"{'='*60}")
    print("If GPU was detected successfully, you can now run your finger.py script.")
    print("If not, you may need to:")
    print("1. Install CUDA 11.x drivers")
    print("2. Install cuDNN for CUDA 11.x") 
    print("3. Check that your GPU is properly configured")

if __name__ == "__main__":
    main()
