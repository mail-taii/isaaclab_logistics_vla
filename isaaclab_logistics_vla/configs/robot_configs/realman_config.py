import pybullet as p
import numpy as np
import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import  OffsetCfg
from isaaclab_logistics_vla.tasks import mdp


from isaaclab_logistics_vla import ISAACLAB_LOGISTICS_VLA_EXT_DIR
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.utils.util import euler2quat

USD_PATH = os.path.join(
    ISAACLAB_LOGISTICS_VLA_EXT_DIR,
    "isaaclab_logistics_vla",
    "assets",
    "robots",
    "realman",
    "realman_franka_ee.usd"
)

@register.add_robot('realman_franka_ee')
@configclass
class RealmanFrankaEE(ArticulationCfg):
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=0,fix_root_link=True,
        ),
    )

    init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"l_joint[1-7]": 0.0,
                       "r_joint[1-7]": 0.0,
                    "left_left_joint":0.04,
                    "left_right_joint":0.04,
                    "right_right_joint":0.04,
                    "right_left_joint":0.04}, 
            pos=(-0.71, 0, 0.216),
            rot= euler2quat('z',-90)    #正对x轴正方向
        )

    actuators={
        "left_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["l_joint[1-7]"], 
            effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "left_hand_gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=["(left_left_joint|left_right_joint)"], 
            effort_limit_sim=20.0,
            stiffness=500.0,
            damping=10.0,
        ),
        "right_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["r_joint[1-7]"], 
            effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "right_hand_gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=["(right_left_joint|right_right_joint)"], 
            effort_limit_sim=20.0,
            stiffness=500.0,
            damping=10.0,
        ),
        'platform_joint': ImplicitActuatorCfg(
            joint_names_expr=["platform_joint"], 
            effort_limit_sim=1000,
            stiffness=5000,
            damping=100,
        ),
    }

@register.add_eeframe_configs('realman_franka_ee_eeframe')
@configclass
class RealmanFrankaEE_FrameTransformerCfg(FrameTransformerCfg):
    prim_path=f"{{ENV_REGEX_NS}}/Robot/base_link_underpan"
    debug_vis=False

    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/panda_left_hand",
            name="left_ee_tcp", 
            # 偏移量：从“手掌”原点向前 10.34cm (标准 Franka TCP)
            offset=OffsetCfg(
                pos=(0.0, 0.0, 0.1034), 
            ),
        ),
        FrameTransformerCfg.FrameCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/panda_right_hand",
            name="right_ee_tcp", 
            # 偏移量：从“手掌”原点向前 10.34cm (标准 Franka TCP)
            offset=OffsetCfg(
                pos=(0.0, 0.0, 0.1034), 
            ),
        )
    ]

@register.add_action_configs('realman_franka_ee_actionscfg')
@configclass
class RealmanFrankaEE_ActionsCfg:
    arm_joints = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_joint[1-7]","r_joint[1-7]"], 
        scale=1.0,
        use_default_offset=False,
    )

    left_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        # 只写左手的两个手指关节
        joint_names=["left_left_joint", "left_right_joint"], 
        # 正则表达式需要匹配上面这俩名字
        open_command_expr={"left_.*_joint": 0.04}, 
        close_command_expr={"left_.*_joint": 0.0},
    )

    right_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        # 只写右手的两个手指关节
        joint_names=["right_left_joint", "right_right_joint"], 
        # 正则表达式需要匹配上面这俩名字
        open_command_expr={"right_.*_joint": 0.04},
        close_command_expr={"right_.*_joint": 0.0},
    )

    platform = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["platform_joint"], 
        scale=1.0,
        use_default_offset=False,
    )