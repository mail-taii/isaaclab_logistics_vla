# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def goal_specific_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    hand_name: str,  # <--- 新增参数：指定要检查哪只手
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    is_world = False
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_sensor: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    try:
        hand_idx = ee_sensor.data.target_frame_names.index(hand_name)
    except ValueError:
        raise ValueError(f"Name '{hand_name}' not found in EE sensor frames: {ee_sensor.frame_names}")

    current_ee_pos_w = ee_sensor.data.target_pos_w[:, hand_idx, :]    #世界坐标系

    command = env.command_manager.get_command(command_name)
    if not is_world:
        target_pos_b = command[:, :3]
        
        target_pos_w, _ = combine_frame_transforms( 
            robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b    #target_pos_b 是相对于机器人（带旋转，也即正方向改变了）的相对坐标
        )
    else:
        target_pos_w = command[:, :3]

    distance = torch.norm(target_pos_w - current_ee_pos_w, dim=1)    #target_pos_w 目标点的绝对坐标
    print(target_pos_w)

    return 1 - torch.tanh(distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))



def command_term_metric(env: ManagerBasedRLEnv, command_name: str, metric_key: str) -> torch.Tensor:
    """
    通用奖励函数：直接从指定的 CommandTerm 中读取 metric 值作为 reward。
    
    Args:
        env: 环境管理器
        command_name: 在 config 中定义的命令项名称 (例如 'order_task')
        metric_key: metric 字典里的 key (例如 'order_completion_rate')
    """
    # 1. 获取 Command Manager
    # 注意：如果你的环境没有 command manager，这里会报错
    cmd_manager: CommandManager= env.command_manager
    
    # 2. 获取特定的 Term 实例 (即 AssignSSSTCommandTerm 的实例)
    # command_name 必须和你 RLEnvCfg 中定义的名称一致
    term = cmd_manager.get_term(command_name)
    
    if term is None:
        raise ValueError(f"Reward function cannot find command term '{command_name}'")
        
    # 3. 获取 Metric
    # 假设你的 Term 已经在 _update_metrics 里更新了 self.metrics
    # 这里的 val 是 (num_envs,) 的 Tensor
    val = term.metrics.get(metric_key)
    
    if val is None:
        # 如果第一帧还没算出来，返回 0
        return torch.zeros(env.num_envs, device=env.device)
        
    return val