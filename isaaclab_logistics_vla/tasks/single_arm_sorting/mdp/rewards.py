# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for single arm sorting task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # Extract the used quantities
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    return 1 - torch.tanh(distance / std)


def object_is_grasped(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for successfully grasping the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Distance between object and end-effector
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    # Check if gripper is closed (simplified check - can be improved)
    # For now, just check distance
    return torch.where(distance < threshold, 1.0, 0.0)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_target_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """Reward the agent for moving the object close to the target area."""
    object: RigidObject = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]  # AssetBase, not RigidObject
    # Distance between object and target
    object_pos_w = object.data.root_pos_w
    # For AssetBase, we use the prim's world position
    # For now, use a fixed target position (can be improved)
    target_pos_w = torch.tensor([0.0, 0.5, 0.0], device=object_pos_w.device).unsqueeze(0).repeat(object_pos_w.shape[0], 1)
    distance = torch.norm(object_pos_w - target_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_is_placed_correctly(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """Reward the agent for placing the object correctly in the target area."""
    object: RigidObject = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]  # AssetBase
    # Distance between object and target
    object_pos_w = object.data.root_pos_w
    # For AssetBase, use fixed target position
    target_pos_w = torch.tensor([0.0, 0.5, 0.0], device=object_pos_w.device).unsqueeze(0).repeat(object_pos_w.shape[0], 1)
    distance = torch.norm(object_pos_w - target_pos_w, dim=1)
    # Check if object is placed correctly (within threshold)
    return torch.where(distance < threshold, 1.0, 0.0)


def task_is_completed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Reward the agent for completing the task (object placed correctly)."""
    object: RigidObject = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]  # AssetBase
    # Distance between object and target
    object_pos_w = object.data.root_pos_w
    # For AssetBase, use fixed target position
    target_pos_w = torch.tensor([0.0, 0.5, 0.0], device=object_pos_w.device).unsqueeze(0).repeat(object_pos_w.shape[0], 1)
    distance = torch.norm(object_pos_w - target_pos_w, dim=1)
    # Check if object is placed correctly and lifted
    is_placed = distance < threshold
    is_lifted = object.data.root_pos_w[:, 2] > 0.05
    return (is_placed & is_lifted).float()

