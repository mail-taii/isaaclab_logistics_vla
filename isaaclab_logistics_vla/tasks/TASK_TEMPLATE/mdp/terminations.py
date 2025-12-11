# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for your_task_name environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Terminate when episode length is reached."""
    return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)


# TODO: 实现你的任务特定的终止条件
# 例如：
# def root_height_below_minimum(
#     env: ManagerBasedRLEnv,
#     minimum_height: float = -0.05,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Terminate when object falls below minimum height."""
#     asset = env.scene[asset_cfg.name]
#     return asset.data.root_pos_w[:, 2] < minimum_height
