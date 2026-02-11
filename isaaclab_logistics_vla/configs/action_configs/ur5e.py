"""
UR5e 机器人动作配置：6 关节臂。
与 configs/robot_configs 中的机器人 asset、eeframe 分离，便于单独修改动作空间。
"""
from isaaclab.utils import configclass

from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks import mdp


@register.add_action_configs("ur5e_actionscfg")
@configclass
class UR5eActionsCfg:
    arm_joints = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )
