from __future__ import annotations

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg

def build_realman_xr_handtracking_devices_cfg(*, sim_device: str, xr_cfg) -> DevicesCfg:
    """Realman 专用 XR(handtracking) 设备配置。

    说明：
    - 保持与 Isaac Lab `teleop_se3_agent.py` 一致：OpenXRDevice + retargeters。
    - 仍使用 Isaac Lab 内置 retargeter（避免黑屏/未知 cfg type），Realman 的轴向/符号映射在运行时后处理。
    """

    left = Se3RelRetargeterCfg(
        bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
        delta_pos_scale_factor=3.0,
        delta_rot_scale_factor=3.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
        use_wrist_rotation=True,
        use_wrist_position=True,
        zero_out_xy_rotation=False,
        sim_device=sim_device,
    )

    right = Se3RelRetargeterCfg(
        bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
        delta_pos_scale_factor=3.0,
        delta_rot_scale_factor=3.0,
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

