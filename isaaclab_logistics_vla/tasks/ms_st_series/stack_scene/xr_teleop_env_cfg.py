import os

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac
from isaaclab_logistics_vla.configs.teleop_configs.realman_xr_handtracking import (
    build_realman_xr_handtracking_devices_cfg,
)

from .env_cfg import Spawn_ds_st_sparse_EnvCfg


@configclass
class RealmanFrankaEE_XrTeleopActionsCfg:
    """XR 遥操作专用动作：双臂 Differential IK + 双夹爪。

    说明：
    - 原 benchmark 的 `realman_franka_ee_actionscfg` 是 17 维 joint-position（14 臂 + 2 夹爪 + 平台）。
    - XR handtracking 更适配 Se3RelRetargeter → Differential IK，因此这里单独提供一个 teleop 动作配置，
      不影响评估与训练。
    """

    # scale 越大末端跟随幅度越大（0.5 偏小，改为 1.0 便于够到箱子）
    left_arm_ik = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["l_joint[1-7]"],
        body_name="panda_left_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )

    right_arm_ik = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["r_joint[1-7]"],
        body_name="panda_right_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
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

    # XR 视角：固定为你在 UI 里调好的 /XRAnchor（更直觉）
    # 来自你的截图：Translate=(1.0, 2.2, 0.0), Orient(XYZ degrees)=(-90, 30, 0)
    xr: XrCfg = XrCfg(
        anchor_pos=(1.0, 2.2, 0.0),
        anchor_rot=euler_to_quat_isaac(r=-90, p=30, y=0, return_tensor=False),
        near_plane=0.15,
    )

    def __post_init__(self):
        super().__post_init__()

        # 动作幅度可配：TELEOP_IK_SCALE（默认 1.0），越大末端跟随幅度越大
        try:
            ik_scale = float(os.environ.get("TELEOP_IK_SCALE", "1.0"))
        except (TypeError, ValueError):
            ik_scale = 1.0
        self.actions.left_arm_ik.scale = ik_scale
        self.actions.right_arm_ik.scale = ik_scale

        # XR 建议的仿真/渲染节奏：90Hz 物理 + 45Hz 立体渲染（按官方建议）
        self.sim.dt = 1.0 / 90.0
        self.sim.render_interval = 2

        # XR 下不建议外置相机；这里不强制移除（你的 benchmark 可能依赖相机观测），先保持原观测配置。
        # 若后续出现 XR + camera 冲突，再在脚本层面按需 remove_camera_configs。

        # Teleop devices：使用 realman 专属配置（与场景解耦，便于复用到其它任务场景）
        self.teleop_devices = build_realman_xr_handtracking_devices_cfg(sim_device=self.sim.device, xr_cfg=self.xr)

