"""
Realman 机器人动作配置：双臂关节、左右夹爪、平台。
与 configs/robot_configs 中的机器人 asset、eeframe 分离，便于单独修改动作空间。
"""
from isaaclab.utils import configclass

from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks import mdp


@register.add_action_configs("realman_franka_ee_actionscfg")
@configclass
class RealmanFrankaEE_ActionsCfg:
    arm_joints = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_joint[1-7]", "r_joint[1-7]"],
        scale=1.0,
        use_default_offset=False,
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

    platform = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["platform_joint"],
        scale=1.0,
        use_default_offset=True,
    )
