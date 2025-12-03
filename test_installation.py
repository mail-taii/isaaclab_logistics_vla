#!/usr/bin/env python3
"""Test script to verify isaaclab_logistics_vla installation."""

import sys

print("Testing isaaclab_logistics_vla installation...")
print("-" * 50)

# Test 1: Import package
try:
    import isaaclab_logistics_vla
    print(f"✓ Successfully imported isaaclab_logistics_vla")
    print(f"  Version: {isaaclab_logistics_vla.__version__}")
except ImportError as e:
    print(f"✗ Failed to import isaaclab_logistics_vla: {e}")
    sys.exit(1)

# Test 2: Import task config
try:
    from isaaclab_logistics_vla.tasks.single_arm_sorting import SingleArmSortingEnvCfg
    print(f"✓ Successfully imported SingleArmSortingEnvCfg")
except ImportError as e:
    print(f"✗ Failed to import SingleArmSortingEnvCfg: {e}")
    sys.exit(1)

# Test 3: Import Franka config
try:
    from isaaclab_logistics_vla.tasks.single_arm_sorting.config.franka import FrankaSingleArmSortingEnvCfg
    print(f"✓ Successfully imported FrankaSingleArmSortingEnvCfg")
except ImportError as e:
    print(f"✗ Failed to import FrankaSingleArmSortingEnvCfg: {e}")
    sys.exit(1)

# Test 4: Check Gym registration
try:
    import gymnasium as gym
    import isaaclab_logistics_vla  # This should register the environments
    
    # Check if environment is registered
    env_id = "Isaac-Logistics-SingleArmSorting-Franka-v0"
    if env_id in gym.envs.registry.env_specs:
        print(f"✓ Environment '{env_id}' is registered in Gym")
    else:
        print(f"⚠ Environment '{env_id}' is not registered yet")
        print("  This is normal if you haven't imported the config module yet")
except Exception as e:
    print(f"⚠ Could not check Gym registration: {e}")

print("-" * 50)
print("Installation test completed!")

