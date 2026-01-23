# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""统一入口脚本：使用随机动作代理运行Isaac Lab环境。

此脚本支持所有已注册的Gym环境，包括：
- Isaac-Realman-lift
- Isaac-Logistics-SingleArmSorting-Franka-v0
- 以及其他通过isaaclab_logistics_vla扩展注册的环境

使用方法:
    python random_agent.py --task Isaac-Realman-lift --num_envs 2 --headless --record-video output.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 如果指定了录制视频，自动启用摄像头（必须在解析参数之前）
if "--record-video" in sys.argv and "--enable_cameras" not in sys.argv:
    print("[信息]: 检测到 --record-video 参数，自动添加 --enable_cameras 标志")
    sys.argv.insert(1, "--enable_cameras")

# add argparse arguments
parser = argparse.ArgumentParser(
    description="统一入口：使用随机动作代理运行Isaac Lab物流VLA环境。",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例用法:
  python random_agent.py --task Isaac-Realman-lift --num_envs 2
  python random_agent.py --task Isaac-Realman-lift --num_envs 2 --headless --record-video output.mp4
  
可用任务:
  - Isaac-Realman-lift: Realman机器人抓取任务
  - Isaac-Logistics-SingleArmSorting-Franka-v0: Franka机器人单臂分拣任务
    """
)

parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="禁用fabric并使用USD I/O操作的命令。",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="要模拟的环境数",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    required=True,
    help="任务名称（例如：Isaac-Realman-lift）。",
)
parser.add_argument(
    "--record-video",
    type=str,
    default=None,
    help="录制视频文件路径（例如：output.mp4）。如果指定，将自动启用摄像头。",
)
parser.add_argument(
    "--record-duration",
    type=float,
    default=10.0,
    help="录制时长（秒），默认10秒。",
)
parser.add_argument(
    "--camera-mode",
    type=str,
    default="fixed",
    choices=["fixed", "orbital"],
    help="相机模式：'fixed'=固定位置(-2, -0.49, 2)，'orbital'=旋转轨道视角。",
)
parser.add_argument(
    "--camera-pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="自定义相机位置（三个浮点数：X Y Z）。如果指定，将覆盖--camera-mode的设置。",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont
import math

# 导入isaaclab_tasks以注册标准任务
import isaaclab_tasks  # noqa: F401

# 导入isaaclab_logistics_vla扩展以注册自定义任务
import isaaclab_logistics_vla  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def add_text_overlay(img_array, text_lines):
    """在图像左上角添加文字叠加层"""
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    try:
        # 尝试使用默认字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    y_offset = 10
    for line in text_lines:
        # 画黑色半透明背景（作为阴影效果）
        bbox = draw.textbbox((0, 0), line, font=font) if font else (0, 0, len(line) * 10, 20)
        draw.rectangle([10 - 2, y_offset - 2, 10 + bbox[2] + 2, y_offset + bbox[3] + 2], fill=(0, 0, 0, 200))
        # 画白色文字
        draw.text((10, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += 22
    return np.array(img_pil)


def main():
    """使用随机动作代理运行Isaac Lab环境。"""
    # 检查任务是否已注册
    if args_cli.task not in gym.envs.registry:
        print(f"[错误]: 任务 '{args_cli.task}' 未在Gym注册表中找到。")
        print(f"[提示]: 请确保已正确安装isaaclab_logistics_vla扩展。")
        print(f"[提示]: 可用的任务列表:")
        # 列出所有Isaac相关的任务
        isaac_tasks = [env_id for env_id in gym.envs.registry.keys() if "Isaac" in env_id]
        for task in sorted(isaac_tasks):
            print(f"  - {task}")
        return

    # 创建环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # 如果指定了录制视频，确保启用摄像头
    video_writer = None
    camera_name = "tiled_camera_top"  # 默认使用tiled_camera_top
    if args_cli.record_video:
        # 启用摄像头（如果环境支持）
        if hasattr(env_cfg.scene, 'tiled_camera_top'):
            camera_name = "tiled_camera_top"
        elif hasattr(env_cfg.scene, 'top_camera'):
            camera_name = "top_camera"
        elif hasattr(env_cfg.scene, 'camera'):
            camera_name = "camera"
        
        if camera_name:
            print(f"[信息]: 找到摄像头配置: {camera_name}")
        else:
            print(f"[警告]: 未找到摄像头配置，将尝试从场景中查找摄像头。")
    
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 打印信息（这是向量化环境）
    print(f"[信息]: Gym观察空间: {env.observation_space}")
    print(f"[信息]: Gym动作空间: {env.action_space}")
    print(f"[信息]: 环境数量: {env.unwrapped.num_envs}")
    print(f"[信息]: 设备: {env.unwrapped.device}")
    if args_cli.headless:
        print(f"[信息]: 运行模式: 无头模式")

    # 重置环境
    env.reset()

    # 设置视频录制
    if args_cli.record_video:
        video_path = Path(args_cli.record_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        # 计算FPS（假设环境步长为0.01秒，decimation为2）
        fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation) if hasattr(env_cfg, 'decimation') else 30
        video_writer = imageio.get_writer(str(video_path), fps=int(fps))
        print(f"[信息]: 开始录制视频到: {video_path}")
        print(f"[信息]: 录制时长: {args_cli.record_duration}秒, FPS: {fps}")

    # 计算最大步数（基于录制时长）
    if args_cli.record_video:
        dt = env_cfg.sim.dt * env_cfg.decimation if hasattr(env_cfg, 'decimation') else 0.02
        max_steps = int(args_cli.record_duration / dt)
    else:
        max_steps = None

    # 相机参数设置
    camera_angle = None  # 初始化，仅在orbital模式下使用
    if args_cli.camera_pos:
        # 如果用户指定了自定义位置，使用自定义位置
        camera_x, camera_y, camera_z = args_cli.camera_pos
        camera_mode = "fixed"
        print(f"[信息]: 使用自定义相机位置: ({camera_x}, {camera_y}, {camera_z})")
    elif args_cli.camera_mode == "fixed":
        # 固定位置模式：使用用户推荐的参数（提高高度）
        camera_x, camera_y, camera_z = -2.0, -0.49, 4.0
        camera_mode = "fixed"
        print(f"[信息]: 使用固定相机位置: ({camera_x}, {camera_y}, {camera_z})")
    else:
        # 旋转轨道模式
        camera_mode = "orbital"
        camera_angle = 0.0  # 初始角度
        camera_radius = 2.5  # 相机距离场景中心的距离
        camera_height = 4.0  # 相机高度
        camera_center = np.array([0.5, -0.5, 0.0])  # 场景中心（根据机器人初始位置调整）
        angle_speed = 0.02  # 每步旋转角度（弧度）
        camera_x = camera_center[0] + camera_radius * math.cos(camera_angle)
        camera_y = camera_center[1] + camera_radius * math.sin(camera_angle)
        camera_z = camera_center[2] + camera_height
        print(f"[信息]: 使用旋转轨道相机模式，初始位置: ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f})")

    step_count = 0
    # 模拟环境
    while simulation_app.is_running():
        # 检查是否达到最大步数
        if max_steps and step_count >= max_steps:
            print(f"[信息]: 已达到最大录制步数 {max_steps}，停止录制。")
            break

        # 在推理模式下运行所有操作
        with torch.inference_mode():
            # 从-1到1采样动作
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # 应用动作
            obs, rewards, dones, truncated, info = env.step(actions)
            step_count += 1

            # 录制视频
            if video_writer is not None:
                isaac_env = env.unwrapped
                
                # 更新相机位置（根据模式）
                if camera_mode == "orbital":
                    # 旋转轨道模式：更新角度和位置
                    camera_angle += angle_speed
                    camera_x = camera_center[0] + camera_radius * math.cos(camera_angle)
                    camera_y = camera_center[1] + camera_radius * math.sin(camera_angle)
                    camera_z = camera_center[2] + camera_height
                    look_at = camera_center
                else:
                    # 固定位置模式：位置不变
                    look_at = np.array([0.5, -0.5, 0.0])  # 场景中心
                
                # 更新相机位姿（使用USD API动态更新）
                try:
                    import omni.usd
                    from pxr import Gf, UsdGeom
                    stage = omni.usd.get_context().get_stage()
                    # 更新第一个环境的相机位置
                    camera_prim_path = f"/World/envs/env_0/orbital_camera"
                    camera_prim = stage.GetPrimAtPath(camera_prim_path)
                    if camera_prim.IsValid():
                        xform = UsdGeom.Xformable(camera_prim)
                        # 清除旧的变换操作
                        xform.ClearXformOpOrder()
                        # 设置位置
                        translate_op = xform.AddTranslateOp()
                        translate_op.Set(Gf.Vec3d(float(camera_x), float(camera_y), float(camera_z)))
                        
                        # 计算旋转（使相机看向场景中心）
                        # 使用look-at逻辑：计算yaw和pitch
                        dx = look_at[0] - camera_x
                        dy = look_at[1] - camera_y
                        dz = look_at[2] - camera_z
                        
                        # 计算yaw（水平旋转）
                        yaw = math.atan2(dy, dx)
                        # 计算pitch（垂直旋转）
                        dist_horizontal = math.sqrt(dx*dx + dy*dy)
                        pitch = -math.atan2(dz, dist_horizontal)
                        
                        # 设置旋转（使用RotateZYX顺序）
                        rotate_op = xform.AddRotateZYXOp()
                        rotate_op.Set(Gf.Vec3f(0.0, float(pitch), float(yaw)))
                except Exception as e:
                    # 如果更新失败，继续使用固定相机（不影响录制）
                    if step_count % 100 == 0:
                        print(f"[警告]: 无法更新相机位置: {e}")
                
                # 获取摄像头数据
                camera_data = None
                
                # 方法1: 如果场景中有tiled_camera_top，使用output["rgb"]
                if hasattr(isaac_env.scene, 'tiled_camera_top'):
                    sensor = isaac_env.scene.tiled_camera_top
                    if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                        if isinstance(sensor.data.output, dict) and 'rgb' in sensor.data.output:
                            camera_data = sensor.data.output['rgb']
                
                # 方法2: 从场景的传感器中获取（使用camera_name）
                if camera_data is None and camera_name and hasattr(isaac_env.scene, 'sensors'):
                    if camera_name in isaac_env.scene.sensors:
                        sensor = isaac_env.scene.sensors[camera_name]
                        if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                            if isinstance(sensor.data.output, dict) and 'rgb' in sensor.data.output:
                                camera_data = sensor.data.output['rgb']
                
                # 方法3: 如果场景中有tiled_camera_top（可能是orbital_camera）
                if camera_data is None and hasattr(isaac_env.scene, 'tiled_camera_top'):
                    sensor = isaac_env.scene.tiled_camera_top
                    if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                        if isinstance(sensor.data.output, dict) and 'rgb' in sensor.data.output:
                            camera_data = sensor.data.output['rgb']
                
                if camera_data is not None:
                    # 转换为numpy数组
                    if torch.is_tensor(camera_data):
                        img = camera_data.cpu().numpy()
                    else:
                        img = np.array(camera_data)
                    
                    # 处理多环境情况：取第一个环境
                    if img.ndim == 4:  # (num_envs, H, W, C)
                        img = img[0]
                    elif img.ndim == 5:  # (num_envs, num_cameras, H, W, C)
                        img = img[0, 0]
                    
                    # 归一化到 [0, 255]
                    if img.dtype != np.uint8:
                        img_min = float(img.min())
                        img_max = float(img.max())
                        if img_max > img_min:
                            img_norm = (img - img_min) / (img_max - img_min)
                        else:
                            img_norm = np.zeros_like(img)
                        img_uint8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img_uint8 = img
                    
                    # 获取机器人位姿信息
                    robot_pos = None
                    robot_rot = None
                    if hasattr(isaac_env.scene, 'robot'):
                        robot = isaac_env.scene.robot
                        if hasattr(robot, 'data'):
                            # 获取第一个环境的机器人位姿
                            robot_pos_w = robot.data.root_pos_w[0].cpu().numpy() if torch.is_tensor(robot.data.root_pos_w) else robot.data.root_pos_w[0]
                            robot_quat_w = robot.data.root_quat_w[0].cpu().numpy() if torch.is_tensor(robot.data.root_quat_w) else robot.data.root_quat_w[0]
                            robot_pos = robot_pos_w[:3]
                            robot_rot = robot_quat_w  # quaternion (w, x, y, z)
                    
                    # 准备文本信息
                    text_lines = [
                        f"Step: {step_count}/{max_steps if max_steps else 'inf'}",
                        f"Camera Pos: ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f})",
                        f"Camera Mode: {camera_mode}",
                    ]
                    if camera_mode == "orbital" and camera_angle is not None:
                        text_lines.append(f"Camera Angle: {math.degrees(camera_angle):.1f}°")
                    if robot_pos is not None:
                        text_lines.extend([
                            f"Robot Pos: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})",
                            f"Robot Rot: ({robot_rot[0]:.3f}, {robot_rot[1]:.3f}, {robot_rot[2]:.3f}, {robot_rot[3]:.3f})",
                        ])
                    
                    # 添加文本叠加
                    img_with_text = add_text_overlay(img_uint8, text_lines)
                    video_writer.append_data(img_with_text)
                else:
                    # 如果没有找到摄像头，尝试从观察空间中获取图像
                    if isinstance(obs, dict):
                        for key in ['rgb', 'image', 'camera', 'policy']:
                            if key in obs:
                                img_obs = obs[key]
                                if torch.is_tensor(img_obs):
                                    img = img_obs.cpu().numpy()
                                else:
                                    img = np.array(img_obs)
                                
                                if img.ndim >= 3:  # 至少是 (H, W, C)
                                    if img.ndim == 4:
                                        img = img[0]
                                    # 归一化
                                    if img.dtype != np.uint8:
                                        img_min = img.min()
                                        img_max = img.max()
                                        if img_max > img_min:
                                            img_norm = (img - img_min) / (img_max - img_min)
                                        else:
                                            img_norm = np.zeros_like(img)
                                        img_uint8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)
                                    else:
                                        img_uint8 = img
                                    video_writer.append_data(img_uint8)
                                    break

            # 每100步打印一次统计信息
            if step_count % 100 == 0:
                print(f"[信息]: 步数: {step_count}, 平均奖励: {rewards.mean().item():.4f}")

            # 如果环境结束，重置
            if dones.any() or truncated.any():
                env.reset()

    # 关闭视频录制
    if video_writer is not None:
        video_writer.close()
        print(f"[信息]: 视频已保存到: {args_cli.record_video}")

    # 关闭模拟器
    env.close()
    print(f"[信息]: 总共运行了 {step_count} 步。")


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭sim app
    simulation_app.close()
