# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric functions for your_task_name environment.

This module defines evaluation metrics for the benchmark:
- Grasping Success Rate: Whether the object is successfully grasped
- Intent Accuracy: Whether the robot follows the correct intent
- Task Success Rate: Whether the complete task is successfully finished
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_object_grasped(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Check if the object is successfully grasped.
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for considering the object as grasped.
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
        
    Returns:
        A boolean tensor (num_envs,) indicating whether each environment has successfully grasped the object.
    """
    # TODO: 根据你的任务实现抓取检测
    # 示例实现：
    # object: RigidObject = env.scene[object_cfg.name]
    # ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # 
    # object_pos_w = object.data.root_pos_w
    # ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    # distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    # 
    # is_lifted = object.data.root_pos_w[:, 2] > 0.02
    # return (distance < threshold) & is_lifted
    
    # 占位实现
    return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)


def is_intent_correct(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    source_cfg: SceneEntityCfg = SceneEntityCfg("source_area"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """Check if the robot follows the correct intent.
    
    TODO: 根据你的任务定义"意图正确"的含义
    
    Args:
        env: The environment instance.
        object_cfg: Configuration for the object entity.
        source_cfg: Configuration for the source area.
        target_cfg: Configuration for the target area.
        
    Returns:
        A boolean tensor (num_envs,) indicating whether each environment follows the correct intent.
    """
    # TODO: 实现意图正确性检测
    # 示例：检查物体是否从源区域向目标区域移动
    # object = env.scene[object_cfg.name]
    # source = env.scene[source_cfg.name]
    # target = env.scene[target_cfg.name]
    # 
    # object_pos_w = object.data.root_pos_w
    # source_pos_w = source.data.root_pos_w
    # target_pos_w = target.data.root_pos_w
    # 
    # dist_to_source = torch.norm(object_pos_w - source_pos_w, dim=1)
    # dist_to_target = torch.norm(object_pos_w - target_pos_w, dim=1)
    # 
    # return dist_to_target < dist_to_source
    
    # 占位实现
    return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)


def is_task_completed(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.05,
    min_height: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """Check if the task is successfully completed.
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for considering the object as correctly placed.
        min_height: Minimum height for the object to be considered as properly placed.
        object_cfg: Configuration for the object entity.
        target_cfg: Configuration for the target area.
        
    Returns:
        A boolean tensor (num_envs,) indicating whether each environment has completed the task.
    """
    # TODO: 实现任务完成检测
    # 示例：检查物体是否在目标区域内且高度合理
    # object = env.scene[object_cfg.name]
    # target = env.scene[target_cfg.name]
    # 
    # object_pos_w = object.data.root_pos_w
    # target_pos_w = target.data.root_pos_w
    # 
    # distance_to_target = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    # is_at_target = distance_to_target < threshold
    # is_at_height = object_pos_w[:, 2] > min_height
    # 
    # return is_at_target & is_at_height
    
    # 占位实现
    return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)
