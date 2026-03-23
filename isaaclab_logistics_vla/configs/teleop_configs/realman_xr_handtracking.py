from __future__ import annotations

import os

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg

def _teleop_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def build_realman_xr_handtracking_devices_cfg(
    *,
    sim_device: str,
    xr_cfg,
    pos_scale: float | None = None,
    rot_scale: float | None = None,
) -> DevicesCfg:
    """Realman 专用 XR(handtracking) 设备配置。

    说明：
    - 保持与 Isaac Lab `teleop_se3_agent.py` 一致：OpenXRDevice + retargeters。
    - 动作幅度：若传入 pos_scale/rot_scale 则优先使用，否则读 TELEOP_POS_SCALE / TELEOP_ROT_SCALE（默认 8.0）。
    """
    if pos_scale is None:
        pos_scale = _teleop_float_env("TELEOP_POS_SCALE", 8.0)
    if rot_scale is None:
        rot_scale = _teleop_float_env("TELEOP_ROT_SCALE", 8.0)

    left = Se3RelRetargeterCfg(
        bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
        delta_pos_scale_factor=pos_scale,
        delta_rot_scale_factor=rot_scale,
        alpha_pos=0.5,
        alpha_rot=0.5,
        use_wrist_rotation=True,
        use_wrist_position=True,
        zero_out_xy_rotation=False,
        sim_device=sim_device,
    )

    right = Se3RelRetargeterCfg(
        bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
        delta_pos_scale_factor=pos_scale,
        delta_rot_scale_factor=rot_scale,
        alpha_pos=0.5,
        alpha_rot=0.5,
        use_wrist_rotation=True,
        use_wrist_position=True,
        zero_out_xy_rotation=False,
        sim_device=sim_device,
    )

    # 顺序必须与 env actions 一致：left_arm_ik, right_arm_ik, left_gripper, right_gripper（否则双臂/夹爪会错位）
    return DevicesCfg(
        devices={
            "handtracking": OpenXRDeviceCfg(
                retargeters=[
                    left,   # 左臂 6D
                    right,  # 右臂 6D
                    GripperRetargeterCfg(bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT, sim_device=sim_device),
                    GripperRetargeterCfg(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, sim_device=sim_device),
                ],
                sim_device=sim_device,
                xr_cfg=xr_cfg,
            )
        }
    )

