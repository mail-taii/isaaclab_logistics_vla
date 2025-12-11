# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for your_task_name environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for large action rates (smoothness)."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


# TODO: 实现你的任务特定的奖励函数
# 例如：
# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float = 0.1,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward for reaching the object."""
#     object = env.scene[object_cfg.name]
#     ee_frame = env.scene[ee_frame_cfg.name]
#     
#     object_pos_w = object.data.root_pos_w
#     ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
#     
#     distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
#     reward = torch.exp(-distance / std)
#     
#     return reward
