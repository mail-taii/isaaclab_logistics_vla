# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run order_series task with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Order Series task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args (includes --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_logistics_vla  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Import the order series environment config
from isaaclab_logistics_vla.tasks.test_tasks.order_series.env_cfg import OrderEnvCfg


def main():
    # Register the order series task
    gym.register(
        id="Isaac-Order-Series-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaaclab_logistics_vla.tasks.test_tasks.order_series.env_cfg:OrderEnvCfg",
        },
    )
    
    # Create environment configuration
    task_name = "Isaac-Order-Series-v0"
    env_cfg = parse_env_cfg(
        task_name, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    
    # Create environment
    env = gym.make(task_name, cfg=env_cfg)

    # Print info (this is vectorized environment)
    print(f"[INFO]: Task: {task_name}")
    print(f"[INFO]: Number of environments: {env.unwrapped.num_envs}")
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"[INFO]: Environment reset successful!")
    
    i = 0
    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Sample random actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # Apply actions
            obs, reward, terminated, truncated, info = env.step(actions)
            
            i += 1
            if i % 100 == 0 or i < 10:
                print(f"[INFO]: Step {i}, Reward: {reward[0].item():.4f}")
            
            # Reset if episode terminated
            if terminated.any() or truncated.any():
                obs, info = env.reset()
                print(f"[INFO]: Environment reset at step {i}")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
