# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for your_task_name environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_rel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos_rel


def joint_vel_rel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities relative to default."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel_rel


def ee_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in robot root frame."""
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get end-effector position in world frame
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    
    # Transform to robot root frame
    robot_pos_w = robot.data.root_pos_w
    robot_rot_w = robot.data.root_quat_w
    
    # TODO: 实现坐标变换
    # 这里简化处理，直接返回世界坐标
    return ee_pos_w - robot_pos_w


def last_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Last action applied to the environment."""
    return env.action_manager.action


# TODO: 添加你的任务特定的观测函数
# 例如：
# def object_position_in_robot_root_frame(...):
#     """Object position relative to robot root."""
#     pass
