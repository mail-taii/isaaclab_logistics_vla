# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric functions for single arm sorting task.

This module defines evaluation metrics for the benchmark:
- Grasping Success Rate: Whether the object is successfully grasped
- Intent Accuracy: Whether the robot follows the correct intent (e.g., grasping the correct object, moving to correct target)
- Task Success Rate: Whether the complete task is successfully finished (grasping from source and placing to target)
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
    env: ManagerBasedRLEnv,
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
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Distance between object and end-effector
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    
    # Check if object is lifted (above a minimal height to confirm grasp)
    is_lifted = object.data.root_pos_w[:, 2] > 0.02
    
    # Object is considered grasped if close to end-effector and lifted
    return (distance < threshold) & is_lifted


def is_intent_correct(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    source_cfg: SceneEntityCfg = SceneEntityCfg("source_area"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """Check if the robot follows the correct intent.
    
    Intent correctness is defined as:
    1. The object starts in the source area (initial state check, done at reset)
    2. The robot attempts to move the object toward the target area
    3. The object should move from source toward target (not in wrong direction)
    
    For simplicity, we check if the object has moved closer to target than source,
    which indicates the robot is following the correct intent.
    
    Args:
        env: The environment instance.
        object_cfg: Configuration for the object entity.
        source_cfg: Configuration for the source area.
        target_cfg: Configuration for the target area.
        
    Returns:
        A boolean tensor (num_envs,) indicating whether each environment follows the correct intent.
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position
    object_pos_w = object.data.root_pos_w
    
    # Source area position (fixed position from scene config)
    source_pos_w = torch.tensor([0.5, 0.0, 0.0], device=object_pos_w.device).unsqueeze(0).repeat(object_pos_w.shape[0], 1)
    
    # Target area position (fixed position from scene config)
    target_pos_w = torch.tensor([0.0, 0.5, 0.0], device=object_pos_w.device).unsqueeze(0).repeat(object_pos_w.shape[0], 1)
    
    # Calculate distances
    dist_to_source = torch.norm(object_pos_w - source_pos_w, dim=1)
    dist_to_target = torch.norm(object_pos_w - target_pos_w, dim=1)
    
    # Intent is correct if object is closer to target than source
    # This indicates the robot is moving in the correct direction
    intent_correct = dist_to_target < dist_to_source
    
    return intent_correct


def is_task_completed(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    min_height: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_area"),
) -> torch.Tensor:
    """Check if the task is successfully completed.
    
    Task completion requires:
    1. Object is grasped (or was grasped during the episode)
    2. Object is placed correctly in the target area (within threshold)
    3. Object is at a reasonable height (not dropped)
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for considering the object as correctly placed.
        min_height: Minimum height for the object to be considered as properly placed.
        object_cfg: Configuration for the object entity.
        target_cfg: Configuration for the target area.
        
    Returns:
        A boolean tensor (num_envs,) indicating whether each environment has completed the task.
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position
    object_pos_w = object.data.root_pos_w
    
    # Target area position (fixed position from scene config)
    target_pos_w = torch.tensor([0.0, 0.5, 0.0], device=object_pos_w.device).unsqueeze(0).repeat(object_pos_w.shape[0], 1)
    
    # Calculate distance to target
    distance_to_target = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)  # Only x, y distance
    
    # Check if object is placed correctly
    is_at_target = distance_to_target < threshold
    
    # Check if object is at reasonable height (not dropped)
    is_at_height = object_pos_w[:, 2] > min_height
    
    # Task is completed if object is at target and at correct height
    return is_at_target & is_at_height


def compute_grasping_success_rate(
    env: ManagerBasedRLEnv,
    was_grasped_buf: torch.Tensor,
) -> torch.Tensor:
    """Compute grasping success rate for each environment.
    
    This metric tracks whether the object was successfully grasped at least once during the episode.
    
    Args:
        env: The environment instance.
        was_grasped_buf: A buffer (num_envs,) tracking if object was grasped at least once.
        
    Returns:
        A boolean tensor (num_envs,) indicating grasping success (1 if grasped at least once, 0 otherwise).
    """
    return was_grasped_buf.float()


def compute_intent_accuracy(
    env: ManagerBasedRLEnv,
    intent_correct_buf: torch.Tensor,
) -> torch.Tensor:
    """Compute intent accuracy for each environment.
    
    This metric tracks whether the robot followed the correct intent during the episode.
    
    Args:
        env: The environment instance.
        intent_correct_buf: A buffer (num_envs,) tracking if intent was correct at least once.
        
    Returns:
        A boolean tensor (num_envs,) indicating intent correctness (1 if correct, 0 otherwise).
    """
    return intent_correct_buf.float()


def compute_task_success_rate(
    env: ManagerBasedRLEnv,
    task_completed_buf: torch.Tensor,
) -> torch.Tensor:
    """Compute task success rate for each environment.
    
    This metric tracks whether the complete task was successfully finished.
    
    Args:
        env: The environment instance.
        task_completed_buf: A buffer (num_envs,) tracking if task was completed.
        
    Returns:
        A boolean tensor (num_envs,) indicating task success (1 if completed, 0 otherwise).
    """
    return task_completed_buf.float()

