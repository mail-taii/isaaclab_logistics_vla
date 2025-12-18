# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for single arm sorting task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

from ..object_randomization import get_active_object_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the end-effector in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # 获取末端执行器在世界坐标系中的位置（取第一个target frame）
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    # 转换到机器人根坐标系
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object_pos_w, _ = get_active_object_pose_w(env, object_cfg)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """The position of the target area in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    target_pos_w = target.data.root_pos_w[:, :3]
    target_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w)
    return target_pos_b


def scene_rgb_image(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("scene_camera"),
) -> torch.Tensor:
    """获取场景RGB图像（场景画面）。
    
    Args:
        env: 环境实例
        camera_cfg: 相机配置
        
    Returns:
        RGB图像，形状为 (num_envs, height, width, 3)
    """
    from isaaclab.sensors import TiledCamera
    
    camera: TiledCamera = env.scene[camera_cfg.name]
    # 获取RGB图像（使用 output 字典）
    rgb_image = camera.data.output["rgb"]  # RGB图像，已经是3通道
    return rgb_image

