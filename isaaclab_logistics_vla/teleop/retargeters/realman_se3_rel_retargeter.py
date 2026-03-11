from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class RealmanSe3RelRetargeterCfg(RetargeterCfg):
    """面向 Realman 的相对位姿 retargeter（在 Isaac Lab Se3Rel 基础上增加坐标映射/死区）。

    设计意图：
    - lerobot-realman-vla 里的 Vive→Robot 有明显的坐标轴交换/符号翻转（Robot_X=-Vive_Z 等）。
    - AVP(OpenXR) 的手部坐标系也经常与机器人基座坐标系不直观一致。
    - 这里提供 map_pos/map_rot 两个 3x3 矩阵，让你把“手动方向”调成符合 realman 操作直觉。
    """

    bound_hand: OpenXRDevice.TrackingTarget = OpenXRDevice.TrackingTarget.HAND_RIGHT
    zero_out_xy_rotation: bool = False
    use_wrist_rotation: bool = True
    use_wrist_position: bool = True
    delta_pos_scale_factor: float = 3.0
    delta_rot_scale_factor: float = 3.0
    alpha_pos: float = 0.5
    alpha_rot: float = 0.5

    # 3x3 映射矩阵（默认 identity）。用于把 OpenXR 的 delta(pos/rotvec) 映射到机器人/控制坐标系。
    map_pos: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    map_rot: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    # 死区（比 Isaac Lab 默认更可控）
    position_threshold: float = 0.001
    rotation_threshold: float = 0.01


class RealmanSe3RelRetargeter(RetargeterBase):
    """Realman 定制的相对位姿 retargeter。输出 6 维 (dx,dy,dz, rx,ry,rz)。"""

    def __init__(self, cfg: RealmanSe3RelRetargeterCfg):
        super().__init__(cfg)
        self.bound_hand = cfg.bound_hand
        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        self._use_wrist_rotation = cfg.use_wrist_rotation
        self._use_wrist_position = cfg.use_wrist_position
        self._delta_pos_scale_factor = cfg.delta_pos_scale_factor
        self._delta_rot_scale_factor = cfg.delta_rot_scale_factor
        self._alpha_pos = cfg.alpha_pos
        self._alpha_rot = cfg.alpha_rot

        self._map_pos = np.asarray(cfg.map_pos, dtype=np.float32)
        self._map_rot = np.asarray(cfg.map_rot, dtype=np.float32)
        self._position_threshold = float(cfg.position_threshold)
        self._rotation_threshold = float(cfg.rotation_threshold)

        self._smoothed_delta_pos = np.zeros(3, dtype=np.float32)
        self._smoothed_delta_rot = np.zeros(3, dtype=np.float32)

        self._previous_thumb_tip = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._previous_index_tip = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._previous_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def retarget(self, data: dict) -> torch.Tensor:
        hand_data = data[self.bound_hand]
        thumb_tip = hand_data.get("thumb_tip")
        index_tip = hand_data.get("index_tip")
        wrist = hand_data.get("wrist")

        if thumb_tip is None or index_tip is None or wrist is None:
            return torch.zeros((6,), dtype=torch.float32, device=self._sim_device)

        delta_thumb_tip = self._calculate_delta_pose(thumb_tip, self._previous_thumb_tip)
        delta_index_tip = self._calculate_delta_pose(index_tip, self._previous_index_tip)
        delta_wrist = self._calculate_delta_pose(wrist, self._previous_wrist)

        self._previous_thumb_tip = thumb_tip.copy()
        self._previous_index_tip = index_tip.copy()
        self._previous_wrist = wrist.copy()

        cmd = self._retarget_rel(delta_thumb_tip, delta_index_tip, delta_wrist)
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    def _calculate_delta_pose(self, joint_pose: np.ndarray, previous_joint_pose: np.ndarray) -> np.ndarray:
        delta_pos = joint_pose[:3] - previous_joint_pose[:3]
        abs_rotation = Rotation.from_quat([*joint_pose[4:7], joint_pose[3]])
        previous_rot = Rotation.from_quat([*previous_joint_pose[4:7], previous_joint_pose[3]])
        relative_rotation = abs_rotation * previous_rot.inv()
        return np.concatenate([delta_pos, relative_rotation.as_rotvec()]).astype(np.float32)

    def _retarget_rel(self, thumb_tip: np.ndarray, index_tip: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        # position source
        if self._use_wrist_position:
            position = wrist[:3]
        else:
            position = (thumb_tip[:3] + index_tip[:3]) / 2

        # rotation source
        if self._use_wrist_rotation:
            rotation = wrist[3:6].copy()
        else:
            rotation = ((thumb_tip[3:6] + index_tip[3:6]) / 2).copy()

        if self._zero_out_xy_rotation:
            rotation[0] = 0.0
            rotation[1] = 0.0

        # apply mapping (OpenXR delta -> desired control delta)
        position = self._map_pos @ position
        rotation = self._map_rot @ rotation

        # smooth & deadzone & scale
        self._smoothed_delta_pos = self._alpha_pos * position + (1.0 - self._alpha_pos) * self._smoothed_delta_pos
        if np.linalg.norm(self._smoothed_delta_pos) < self._position_threshold:
            self._smoothed_delta_pos = np.zeros(3, dtype=np.float32)
        position = self._smoothed_delta_pos * self._delta_pos_scale_factor

        self._smoothed_delta_rot = self._alpha_rot * rotation + (1.0 - self._alpha_rot) * self._smoothed_delta_rot
        if np.linalg.norm(self._smoothed_delta_rot) < self._rotation_threshold:
            self._smoothed_delta_rot = np.zeros(3, dtype=np.float32)
        rotation = self._smoothed_delta_rot * self._delta_rot_scale_factor

        return np.concatenate([position, rotation]).astype(np.float32)

