from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

from .env_cfg import Spawn_ds_st_sparse_EnvCfg


@configclass
class RealmanFrankaEE_XrTeleopActionsCfg:
    """XR 遥操作专用动作：双臂 Differential IK + 双夹爪。

    说明：
    - 原 benchmark 的 `realman_franka_ee_actionscfg` 是 17 维 joint-position（14 臂 + 2 夹爪 + 平台）。
    - XR handtracking 更适配 Se3RelRetargeter → Differential IK，因此这里单独提供一个 teleop 动作配置，
      不影响评估与训练。
    """

    left_arm_ik = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["l_joint[1-7]"],
        body_name="panda_left_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )

    right_arm_ik = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["r_joint[1-7]"],
        body_name="panda_right_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )

    left_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_left_joint", "left_right_joint"],
        open_command_expr={"left_.*_joint": 0.04},
        close_command_expr={"left_.*_joint": 0.0},
    )

    right_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_left_joint", "right_right_joint"],
        open_command_expr={"right_.*_joint": 0.04},
        close_command_expr={"right_.*_joint": 0.0},
    )

    # 升降柱：与 RealmanFrankaEE_ActionsCfg 一致，必须给出目标位置否则会因无控制而下坠
    platform = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["platform_joint"],
        scale=1.0,
        use_default_offset=False,
    )


@register.add_env_configs("Spawn_ds_st_sparse_XRTeleop_EnvCfg")
@configclass
class Spawn_ds_st_sparse_XRTeleop_EnvCfg(Spawn_ds_st_sparse_EnvCfg):
    """在同一场景上启用 XR(handtracking) 遥操作的 EnvCfg 变体。"""

    # 覆盖动作配置
    actions = RealmanFrankaEE_XrTeleopActionsCfg()

    # XR 视角：AVP 正前方即顶视（与 realman top_camera 一致：俯视工作区）
    # anchor_rot: 绕 X 轴 180°，使场景「上」(sim Z) 对准头显「前方」(-Z)，正对头显即俯视
    # 四元数 (w,x,y,z) 绕 X 轴 180° = (0, 1, 0, 0)
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.2),
        anchor_rot=(0.0, 1.0, 0.0, 0.0),
        near_plane=0.15,
    )

    def __post_init__(self):
        super().__post_init__()

        # XR 建议的仿真/渲染节奏：90Hz 物理 + 45Hz 立体渲染（按官方建议）
        self.sim.dt = 1.0 / 90.0
        self.sim.render_interval = 2

        # XR 下不建议外置相机；这里不强制移除（你的 benchmark 可能依赖相机观测），先保持原观测配置。
        # 若后续出现 XR + camera 冲突，再在脚本层面按需 remove_camera_configs。

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        # 左手 → 左臂末端增量
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
                            sim_device=self.sim.device,
                        ),
                        # 右手 → 右臂末端增量
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                )
            }
        )

