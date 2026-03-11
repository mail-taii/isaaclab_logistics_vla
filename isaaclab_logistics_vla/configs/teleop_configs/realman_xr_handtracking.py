from __future__ import annotations

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg

from isaaclab_logistics_vla.teleop.retargeters.realman_se3_rel_retargeter import (
    RealmanSe3RelRetargeterCfg,
)


def build_realman_xr_handtracking_devices_cfg(*, sim_device: str, xr_cfg) -> DevicesCfg:
    """Realman 专用 XR(handtracking) 设备配置。

    说明：
    - 保持与 Isaac Lab `teleop_se3_agent.py` 一致：OpenXRDevice + retargeters。
    - 这里用 RealmanSe3RelRetargeter 作为可选替代（带坐标轴映射/死区/缩放），默认不改变轴向（identity）。
    """

    left = RealmanSe3RelRetargeterCfg(
        bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
        delta_pos_scale_factor=3.0,
        delta_rot_scale_factor=3.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
        # 默认不做轴交换；若你想按 lerobot 的 Vive→Robot 习惯，可改为 map_pos=[[0,0,-1],[-1,0,0],[0,1,0]]
        map_pos=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        map_rot=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        use_wrist_rotation=True,
        use_wrist_position=True,
        zero_out_xy_rotation=False,
        sim_device=sim_device,
    )

    right = RealmanSe3RelRetargeterCfg(
        bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
        delta_pos_scale_factor=3.0,
        delta_rot_scale_factor=3.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
        map_pos=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        map_rot=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        use_wrist_rotation=True,
        use_wrist_position=True,
        zero_out_xy_rotation=False,
        sim_device=sim_device,
    )

    return DevicesCfg(
        devices={
            "handtracking": OpenXRDeviceCfg(
                retargeters=[
                    # 左手 → 左臂末端增量 + 左夹爪
                    left,
                    GripperRetargeterCfg(bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT, sim_device=sim_device),
                    # 右手 → 右臂末端增量 + 右夹爪
                    right,
                    GripperRetargeterCfg(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, sim_device=sim_device),
                ],
                sim_device=sim_device,
                xr_cfg=xr_cfg,
            )
        }
    )

