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
    print(f"\n开始持续运行（使用IK控制到目标物品位置）...")
    print("提示：脚本会持续运行，直到应用关闭（流直播时会一直运行）")
    print("可以通过 WebRTC 流直播查看机器人运动\n")
    
    # 获取场景对象
    scene = env.unwrapped.scene
    robot = scene["robot"]
    device = env.unwrapped.device
    
    # 验证环境信息是否可访问（benchmark必须功能）
    print("\n验证环境信息可访问性...")
    try:
        object_rigid = scene["object"]
        print(f"✓ 物品对象可访问: {type(object_rigid).__name__}")
        if hasattr(object_rigid.data, "object_pos_w"):
            print(f"  object_pos_w shape: {object_rigid.data.object_pos_w.shape}")
            print(f"  object_quat_w shape: {object_rigid.data.object_quat_w.shape}")
        else:
            print("  (提示) RigidObjectCollection 没有 root_pos_w，请用 object_pos_w/object_quat_w 或 get_active_object_pose_w")
    except Exception as e:
        print(f"✗ 无法访问物品对象: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        ee_frame = scene["ee_frame"]
        print(f"✓ 末端执行器帧可访问: {type(ee_frame).__name__}")
        # 使用与observations.py相同的方式：target_pos_w[..., 0, :] 获取第一个target frame（left_ee_tcp）
        ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]
        ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # [num_envs, 4]
        print(f"  left_ee_tcp pos_w shape: {ee_pos_w.shape}")
        print(f"  left_ee_tcp quat_w shape: {ee_quat_w.shape}")
    except Exception as e:
        print(f"✗ 无法访问末端执行器帧: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"✓ 环境数量: {scene.num_envs}")
    print(f"✓ 动作空间维度: {env.action_space.shape[0]}")
    print()
    
    # 获取机器人初始关节状态（用于对比）
    prev_joint_pos = robot.data.joint_pos.clone()
    step_count = 0
    
    # 持续运行，直到应用关闭
    while simulation_app.is_running():
        step_count += 1
        
        # 读取目标物品的位置和旋转
        with torch.inference_mode():
            try:
                object_rigid = scene["object"]
                if hasattr(object_rigid.data, "object_pos_w"):
                    # RigidObjectCollection: 使用 collection 数据
                    env_ids = torch.arange(scene.num_envs, device=device)
                    active_ids = getattr(env.unwrapped, "_active_object_indices", None)
                    if active_ids is None or active_ids.numel() != scene.num_envs:
                        active_ids = torch.zeros(scene.num_envs, dtype=torch.long, device=device)
                    object_pos = object_rigid.data.object_pos_w[env_ids, active_ids]
                    object_quat = object_rigid.data.object_quat_w[env_ids, active_ids]
                else:
                    # 单个 RigidObject 兼容路径
                    object_pos = object_rigid.data.root_pos_w[:, :3]
                    object_quat = object_rigid.data.root_quat_w
                
                # shape 校验
                if object_pos.dim() != 2 or object_pos.shape[1] != 3:
                    raise ValueError(f"object_pos shape should be [num_envs, 3], got: {object_pos.shape}")
                num_envs = object_pos.shape[0]
                
                # 获取当前末端执行器位置（用于插值，使运动更平滑）
                # 使用与observations.py相同的方式：target_pos_w[..., 0, :] 获取第一个target frame（left_ee_tcp）
                try:
                    ee_frame = scene["ee_frame"]
                    current_ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]
                    current_ee_rot = ee_frame.data.target_quat_w[..., 0, :]  # [num_envs, 4]
                    
                    # 确保形状正确
                    if current_ee_pos.dim() != 2 or current_ee_pos.shape[1] != 3:
                        raise ValueError(f"current_ee_pos shape should be [num_envs, 3], got: {current_ee_pos.shape}")
                    if current_ee_rot.dim() != 2 or current_ee_rot.shape[1] != 4:
                        raise ValueError(f"current_ee_rot shape should be [num_envs, 4], got: {current_ee_rot.shape}")
                except (KeyError, AttributeError, ValueError) as e:
                    # 如果获取不到当前末端位置，直接使用目标位置
                    if step_count % 50 == 0:
                        print(f"⚠️  无法获取末端位置，使用默认值: {e}")
                    current_ee_pos = object_pos.clone()
                    # 创建一个默认的四元数 [1, 0, 0, 0] (无旋转)
                    current_ee_rot = torch.zeros((num_envs, 4), device=device, dtype=object_pos.dtype)
                    current_ee_rot[:, 0] = 1.0  # w=1
                
                # 设置抓取姿态：末端执行器垂直向下（适合抓取）
                # 四元数 (w, x, y, z) = (1, 0, 0, 0) 表示无旋转（默认姿态）
                grasp_rot = torch.zeros((num_envs, 4), device=device, dtype=object_pos.dtype)
                grasp_rot[:, 0] = 1.0  # w=1, 无旋转
                
                # 目标位置：物品位置上方一点（便于抓取）
                target_pos = object_pos.clone()
                target_pos[:, 2] += 0.05  # 在物品上方5cm
                
                # 小步长插值，使运动更平滑
                alpha = 0.1  # 插值系数，可以调整运动速度
                interp_pos = current_ee_pos + alpha * (target_pos - current_ee_pos)
                interp_rot = current_ee_rot + alpha * (grasp_rot - current_ee_rot)
                # 归一化四元数
                interp_rot = interp_rot / torch.linalg.norm(interp_rot, dim=-1, keepdim=True).clamp(min=1e-6)
                
                # 确保插值后的形状正确
                if interp_pos.dim() != 2 or interp_pos.shape[1] != 3:
                    raise ValueError(f"interp_pos shape should be [num_envs, 3], got: {interp_pos.shape}")
                if interp_rot.dim() != 2 or interp_rot.shape[1] != 4:
                    raise ValueError(f"interp_rot shape should be [num_envs, 4], got: {interp_rot.shape}")
                
                # 构造末端执行器目标位姿：[x, y, z, qw, qx, qy, qz]
                target_pose = torch.cat([interp_pos, interp_rot], dim=-1)  # shape: [num_envs, 7]
                
                # 确保 target_pose 形状正确
                if target_pose.dim() != 2 or target_pose.shape[1] != 7:
                    raise ValueError(f"target_pose shape should be [num_envs, 7], got: {target_pose.shape}")
                
                # 构造完整动作向量
                # 动作空间格式：[left_arm_pose(7), right_arm_pose(7), gripper(?)]
                total_dim = env.action_space.shape[0]
                action = torch.zeros((num_envs, total_dim), device=device, dtype=target_pose.dtype)
                
                # 填充左手臂目标位姿（前7维）
                if total_dim >= 7:
                    action[:, :7] = target_pose
                else:
                    # 如果动作空间小于7维，只填充前total_dim维
                    action[:, :total_dim] = target_pose[:, :total_dim]
                
                # 如果有右手臂，可以设置为相同位置或保持不动（设为0）
                # 如果有夹爪，可以设置为打开状态（根据实际情况调整）
                
            except Exception as e:
                # 如果获取物品位置失败，使用随机动作作为fallback
                import traceback
                if step_count % 50 == 0:  # 只在每50步输出一次错误，避免刷屏
                    print(f"⚠️  无法获取物品位置，使用随机动作: {e}")
                    print(f"    错误类型: {type(e).__name__}")
                    traceback.print_exc()
                # 确保随机动作的形状正确
                num_envs = env.unwrapped.scene.num_envs
                action = (torch.rand((num_envs, env.action_space.shape[0]), device=device) - 0.5) * 2.0
        
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
                
                # 获取物品和末端执行器位置（用于调试）
                try:
                    object_pos = scene["object"].data.root_pos_w[0, :3].cpu().numpy()
                    ee_frame = scene["ee_frame"]
                    left_ee_tcp = ee_frame.get_target_frame("left_ee_tcp")
                    ee_pos = left_ee_tcp.data.pos_w[0].cpu().numpy()
                    distance = torch.norm(left_ee_tcp.data.pos_w[0] - scene["object"].data.root_pos_w[0, :3]).item()
                except:
                    object_pos = None
                    ee_pos = None
                    distance = None
                
                print(f"\n[步骤 {step_count}] 关节位置变化: {joint_diff.item():.6f}, 奖励: {reward_val:.4f}")
                if object_pos is not None:
                    print(f"  物品位置: [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]")
                if ee_pos is not None:
                    print(f"  末端位置: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                if distance is not None:
                    print(f"  末端到物品距离: {distance:.3f}m")
                
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

