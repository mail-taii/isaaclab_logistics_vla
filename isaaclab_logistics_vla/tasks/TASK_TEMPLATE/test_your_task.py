#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""测试脚本：验证新任务环境是否正常工作。

使用方法：
    LIVESTREAM=2 HEADLESS=1 ENABLE_CAMERAS=1 ./isaaclab.sh -p source/isaaclab_logistics_vla/isaaclab_logistics_vla/tasks/your_task_name/test_your_task.py
"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数支持
parser = argparse.ArgumentParser(description="测试新任务环境")
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下导入需要在应用启动后进行"""

import numpy as np
import torch
import gymnasium as gym

# 导入任务（自动注册环境）
import isaaclab_logistics_vla


def main():
    """测试新任务环境"""
    
    print("="*70)
    print("测试新任务环境")
    print("="*70)
    
    # TODO: 修改环境名称
    env_name = "Isaac-Logistics-YourTaskName-Realman-v0"
    print(f"\n创建环境: {env_name}")
    
    try:
        from isaaclab_tasks.utils import parse_env_cfg
        
        env_cfg = parse_env_cfg(
            env_name,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not getattr(args_cli, 'disable_fabric', False),
        )
        env = gym.make(env_name, cfg=env_cfg)
    except ImportError:
        # 如果 parse_env_cfg 不可用，直接使用环境配置类
        # TODO: 导入你的环境配置
        # from isaaclab_logistics_vla.tasks.your_task_name.config.realman import (
        #     RealmanYourTaskNameEnvCfg,
        # )
        # 
        # env_cfg = RealmanYourTaskNameEnvCfg()
        # if args_cli.num_envs is not None:
        #     env_cfg.scene.num_envs = args_cli.num_envs
        # 
        # env = gym.make(env_name, cfg=env_cfg)
        print("✗ 环境配置未实现")
        return
    
    print("✓ 环境创建成功")
    
    # 获取动作空间信息
    print(f"\n动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 重置环境
    print("\n重置环境...")
    try:
        obs, info = env.reset()
        print("✓ 环境重置成功")
        print(f"  观测形状: {obs.shape if isinstance(obs, (torch.Tensor, np.ndarray)) else type(obs)}")
        if isinstance(obs, dict):
            for key, value in obs.items():
                print(f"    {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
    except Exception as e:
        print(f"✗ 环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # 运行几步测试
    print(f"\n运行测试（10步随机动作）...")
    for step in range(10):
        # 生成随机动作
        with torch.inference_mode():
            action = (torch.rand(env.action_space.shape, device=env.unwrapped.device) - 0.5) * 2.0
        
        # 执行动作
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  步骤 {step+1}: 奖励={reward.item() if isinstance(reward, torch.Tensor) else reward:.4f}")
            
            # 检查指标
            if "log" in info and "metrics" in info["log"]:
                metrics = info["log"]["metrics"]
                print(f"    指标: {metrics}")
            
            if terminated.any() if isinstance(terminated, torch.Tensor) else terminated:
                print(f"  Episode 终止，重置环境...")
                obs, info = env.reset()
                
        except Exception as e:
            print(f"✗ 步骤 {step+1} 执行失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
    
    env.close()
    print("\n环境已关闭")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
