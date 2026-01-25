# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401   只要它被导入，它内部的 __init__.py 就会自动运行，并执行所有 gym.register() 操作
import isaaclab_logistics_vla  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)




def main():
    gym.register(
        id="Isaac-Realman-lift",
        
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        
        disable_env_checker=True,
        
        kwargs={
            "env_cfg_entry_point": f"isaaclab_logistics_vla.tasks.realman_lift.realman_lift_env_cfg:LiftEnvCfg",
            # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RealmanPPORunnerCfg",
        },
    )
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    i = 1
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            i+=1
            #actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            actions = 2* torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)
            if i%100==0 or i<10:
                # 1. 获取底层的 Isaac Lab 环境
                isaac_env = env.unwrapped
                
                # 2. 从 scene 中按名称访问你的机器人
                #    这个 "robot" 名称来自你的 realman_env_cfg.py
                robot_asset = isaac_env.scene.articulations["robot"]
                
            
                default_state_tensor = robot_asset.data.root_state_w
                
                print("\n" + "="*50)
                print("Default Root State of 'robot' Asset:")
                print(f"Shape: {default_state_tensor.shape}")
                print(f"Data:\n{default_state_tensor[:, 0:3]}")
                print("="*50 + "\n")
                

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()