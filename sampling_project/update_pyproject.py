#!/usr/bin/env python
"""
Script to update pyproject.toml based on the current installed dependencies.
"""
import tomli
import tomli_w
import subprocess
import re
from pathlib import Path

# Read the existing pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)

# Get the currently installed packages using uv pip freeze
try:
    result = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True, check=True)
    installed_packages = result.stdout.strip().split("\n")
except subprocess.CalledProcessError:
    print("Error running 'uv pip freeze'. Falling back to regular pip...")
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
    installed_packages = result.stdout.strip().split("\n")

# Parse the installed packages and versions
package_versions = {}
for line in installed_packages:
    # Skip the current project if it's installed in editable mode
    if line.startswith("-e "):
        continue
    
    # Parse package name and version
    match = re.match(r"([^=]+)==([^=]+)", line)
    if match:
        package_name, version = match.groups()
        package_versions[package_name.lower()] = version

# Core dependencies to include (add/remove packages as needed)
core_dependencies = [
    "evosax",
    "flax",
    "huggingface_hub",
    "interpax",
    "jax",
    "mujoco",
    "mujoco-mjx",
    "tqdm",
    "imageio",
    "numpy",
    "pandas",
    "seaborn",
    "matplotlib",
]

# Development dependencies
dev_dependencies = [
    "matplotlib",
    "pytest",
    "ruff",
    "pre-commit",
]

# Update the dependencies in pyproject.toml
updated_dependencies = []
for package in core_dependencies:
    package_lower = package.lower()
    if package_lower in package_versions:
        if package_lower == "jax":
            # Add CUDA support for JAX
            updated_dependencies.append(f"jax[cuda12-local]>={package_versions[package_lower]}")
        else:
            updated_dependencies.append(f"{package}>={package_versions[package_lower]}")
    elif package_lower == "huggingface_hub" and "huggingface-hub" in package_versions:
        # Handle dash vs underscore in package names
        updated_dependencies.append(f"huggingface_hub>={package_versions['huggingface-hub']}")

# Update dev dependencies
updated_dev_dependencies = []
for package in dev_dependencies:
    package_lower = package.lower()
    if package_lower in package_versions:
        updated_dev_dependencies.append(f"{package}>={package_versions[package_lower]}")

# Add back any missing dev dependencies from the original pyproject.toml
original_dev_deps = pyproject["project"]["optional-dependencies"].get("dev", [])
for dep in original_dev_deps:
    package_name = dep.split(">=")[0].split(">")[0].split("==")[0].split("<=")[0].strip()
    if not any(d.lower().startswith(package_name.lower()) for d in updated_dev_dependencies):
        updated_dev_dependencies.append(dep)

# Update the pyproject.toml
pyproject["project"]["dependencies"] = updated_dependencies
pyproject["project"]["optional-dependencies"]["dev"] = updated_dev_dependencies

# Add CPU-only option as an optional dependency
if "cpu" not in pyproject["project"]["optional-dependencies"]:
    pyproject["project"]["optional-dependencies"]["cpu"] = []

# For CPU-only mode, use JAX without CUDA
if "jax" in package_versions:
    pyproject["project"]["optional-dependencies"]["cpu"] = [f"jax>={package_versions['jax']}"]

# Write the updated pyproject.toml
with open("pyproject.toml", "rb") as f:
    original_pyproject = tomli.load(f)

# Preserve the original license format
if "license" in original_pyproject["project"]:
    pyproject["project"]["license"] = original_pyproject["project"]["license"]

with open("pyproject.toml", "wb") as f:
    tomli_w.dump(pyproject, f)

print("Updated pyproject.toml with current dependencies:")
print("\nCore dependencies:")
for dep in updated_dependencies:
    print(f"  - {dep}")

print("\nDev dependencies:")
for dep in updated_dev_dependencies:
    print(f"  - {dep}")
