"""
轨迹回放策略：从 TXT 文件加载 RRT 等规划轨迹，前 N 步保持零动作（如升降柱抬升），之后按帧回放。
原逻辑从 VLA_Evaluator 中拆出，使评估器只负责驱动，策略单独实现。
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from isaaclab_logistics_vla.evaluation.models.policy.base import Policy


def load_and_process_trajectory_txt(
    file_path: str,
    device: torch.device,
) -> torch.Tensor:
    """
    读取带 [ ] 和 , 的 TXT，并将顺序格式转换为交错格式（左臂 7 + 右臂 7 -> 交错 14）。
    要求每行 14 个数。
    """
    raw_data_list = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            clean_line = line.replace("[", "").replace("]", "").replace(",", " ")
            row_values = np.fromstring(clean_line, sep=" ")
            if len(row_values) > 0:
                raw_data_list.append(row_values)
    raw_data = np.array(raw_data_list)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    T, D = raw_data.shape
    if D != 14:
        raise ValueError(f"TXT data must have 14 columns, but got {D}")
    left_arm = raw_data[:, 0:7]
    right_arm = raw_data[:, 7:14]
    interleaved_data = np.zeros_like(raw_data)
    interleaved_data[:, 0::2] = left_arm
    interleaved_data[:, 1::2] = right_arm
    return torch.tensor(interleaved_data, dtype=torch.float32, device=device)


class TrajectoryPlaybackPolicy(Policy):
    """
    前 lift_duration 步输出零动作（除固定夹爪/升降柱），之后按 TXT 轨迹逐帧回放；
    轨迹播完后保持最后一帧。
    """

    def __init__(
        self,
        txt_path: str,
        device: torch.device,
        action_dim: int = 17,
        lift_duration: int = 250,
    ):
        self._device = device
        self._action_dim = action_dim
        self._lift_duration = lift_duration
        self._step_counter = 0
        if os.path.exists(txt_path):
            self._action_trajectory = load_and_process_trajectory_txt(txt_path, device)
        else:
            print(f"[WARNING] Trajectory file {txt_path} not found. Policy will output zeros.")
            self._action_trajectory = None

    @property
    def name(self) -> str:
        return "trajectory_playback"

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        self._step_counter = 0

    def predict(self, obs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        num_envs = 1
        if obs and "robot_state" in obs:
            rs = obs["robot_state"]
            if isinstance(rs, dict) and "qpos" in rs and isinstance(rs["qpos"], torch.Tensor):
                num_envs = rs["qpos"].shape[0]

        actions = torch.zeros((num_envs, self._action_dim), device=self._device)
        actions[:, 16] = 0.5
        actions[:, 14] = 0.0
        actions[:, 15] = 0.0

        if self._step_counter >= self._lift_duration and self._action_trajectory is not None:
            traj_idx = self._step_counter - self._lift_duration
            if traj_idx < len(self._action_trajectory):
                current_pose = self._action_trajectory[traj_idx]
            else:
                current_pose = self._action_trajectory[-1]
            actions[:, 0:14] = current_pose

        self._step_counter += 1
        return actions
