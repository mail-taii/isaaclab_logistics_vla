#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""测试接口：验证 Benchmark 输入输出接口。

这个脚本测试 single_arm_sorting 任务的接口是否正确：
- 输入：动作（action）
- 输出：场景画面（图像）、机器人当前位姿、三种指标

使用随机动作进行测试。

环境变量支持：
- LIVESTREAM=2: 启用流直播（私有网络）
- HEADLESS=1: 无头模式
- ENABLE_CAMERAS=1: 启用相机（必须，因为需要图像输出）

使用示例：
    # 无头模式 + 流直播 + 启用相机（推荐）
    LIVESTREAM=2 HEADLESS=1 ENABLE_CAMERAS=1 ./isaaclab.sh -p source/isaaclab_logistics_vla/method/test_interface.py
    
    # 或者使用命令行参数（推荐）
    LIVESTREAM=2 HEADLESS=1 ./isaaclab.sh -p source/isaaclab_logistics_vla/method/test_interface.py --enable_cameras
    
    # 环境变量方式
    ENABLE_CAMERAS=1 LIVESTREAM=2 HEADLESS=1 ./isaaclab.sh -p source/isaaclab_logistics_vla/method/test_interface.py
"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数支持
parser = argparse.ArgumentParser(description="测试 Benchmark 输入输出接口")
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用 Fabric，使用 USD I/O 操作"
)
# 添加 AppLauncher 的命令行参数（支持 --headless, --livestream 等）
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 Omniverse 应用（会读取环境变量 LIVESTREAM, HEADLESS 等）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下导入需要在应用启动后进行"""

import numpy as np
import torch
import gymnasium as gym

# 导入任务（自动注册环境）
import isaaclab_logistics_vla


def main():
    """测试接口：输入动作，输出场景画面、机器人位姿和指标。"""
    
    print("="*70)
    print("Benchmark 接口测试 - Single Arm Sorting")
    print("="*70)
    
    # 显示环境变量信息
    import os
    livestream = os.environ.get("LIVESTREAM", "未设置")
    headless = os.environ.get("HEADLESS", "未设置")
    enable_cameras = os.environ.get("ENABLE_CAMERAS", "未设置")
    print(f"\n环境变量配置:")
    print(f"  LIVESTREAM={livestream}")
    print(f"  HEADLESS={headless}")
    print(f"  ENABLE_CAMERAS={enable_cameras}")
    print(f"  并行环境数量: {args_cli.num_envs}")
    
    # 检查相机是否启用（通过命令行参数或环境变量）
    enable_cameras_enabled = (
        hasattr(args_cli, "enable_cameras") and args_cli.enable_cameras
    ) or (enable_cameras != "未设置" and int(enable_cameras) == 1)
    
    if not enable_cameras_enabled:
        print("\n⚠️  警告: 相机未启用！环境需要相机来输出图像观测。")
        print("   请使用 --enable_cameras 参数或设置 ENABLE_CAMERAS=1 环境变量。")
    
    # 创建环境
    env_name = "Isaac-Logistics-SingleArmSorting-Realman-v0"
    print(f"\n创建环境: {env_name}")
    
    try:
        # 尝试使用 parse_env_cfg（如果可用）
        try:
            from isaaclab_tasks.utils import parse_env_cfg
            
            env_cfg = parse_env_cfg(
                env_name,
                device=args_cli.device,
                num_envs=args_cli.num_envs,
                use_fabric=not args_cli.disable_fabric,
            )
            env = gym.make(env_name, cfg=env_cfg)
        except ImportError:
            # 如果 parse_env_cfg 不可用，直接使用环境配置类
            from isaaclab_logistics_vla.tasks.single_arm_sorting.config.realman import (
                RealmanSingleArmSortingEnvCfg,
            )
            
            env_cfg = RealmanSingleArmSortingEnvCfg()
            if args_cli.num_envs is not None:
                env_cfg.scene.num_envs = args_cli.num_envs
            
            env = gym.make(env_name, cfg=env_cfg)
        
        print("✓ 环境创建成功")
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
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
    
    # 持续运行直到应用关闭
    print(f"\n开始持续运行（使用随机动作）...")
    print("提示：脚本会持续运行，直到应用关闭（流直播时会一直运行）")
    print("可以通过 WebRTC 流直播查看机器人运动\n")
    
    # 获取机器人初始关节状态（用于对比）
    robot = env.unwrapped.scene["robot"]
    prev_joint_pos = robot.data.joint_pos.clone()
    step_count = 0
    
    # 持续运行，直到应用关闭
    while simulation_app.is_running():
        step_count += 1
        
        # 生成随机动作（使用更大的范围，让动作更明显）
        with torch.inference_mode():
            action = (torch.rand(env.action_space.shape, device=env.unwrapped.device) - 0.5) * 2.0
        
        # 执行动作
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录执行后的关节位置并计算变化
            current_joint_pos = robot.data.joint_pos.clone()
            joint_diff = (current_joint_pos - prev_joint_pos).abs().max()
            prev_joint_pos = current_joint_pos
            
            # 每50步输出一次详细信息，减少输出频率
            if step_count % 50 == 0:
                # 获取奖励值（处理tensor）
                if isinstance(reward, torch.Tensor):
                    reward_val = reward[0].item() if reward.numel() > 1 else reward.item()
                else:
                    reward_val = float(reward)
                
                print(f"\n[步骤 {step_count}] 关节位置变化: {joint_diff.item():.6f}, 奖励: {reward_val:.4f}")
                
                if "log" in info and "metrics" in info["log"]:
                    metrics = info["log"]["metrics"]
                    metrics_str = ", ".join([f"{k}: {v.item():.3f}" if hasattr(v, 'item') else f"{k}: {v}" 
                                           for k, v in metrics.items()])
                    if metrics_str:
                        print(f"  指标: {metrics_str}")
            
            # 如果环境终止，重置
            if terminated.any() if isinstance(terminated, torch.Tensor) else terminated:
                # 只在每50步输出时显示episode终止信息
                if step_count % 50 == 0:
                    print(f"  Episode 终止，重置环境...")
                    if "log" in info:
                        log = info["log"]
                        if "grasping_success_rate" in log and log["grasping_success_rate"]:
                            print(f"    抓取成功率: {log['grasping_success_rate']}")
                        if "intent_accuracy" in log and log["intent_accuracy"]:
                            print(f"    意图正确率: {log['intent_accuracy']}")
                        if "task_success_rate" in log and log["task_success_rate"]:
                            print(f"    任务成功率: {log['task_success_rate']}")
                obs, info = env.reset()
                prev_joint_pos = robot.data.joint_pos.clone()
                
        except KeyboardInterrupt:
            print("\n\n收到中断信号，正在停止...")
            break
        except Exception as e:
            print(f"\n✗ 步骤 {step_count} 执行失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*70)
    print("接口测试完成")
    print("="*70)
    
    # 关闭环境
    env.close()
    print("\n环境已关闭")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # 关闭应用
        simulation_app.close()

