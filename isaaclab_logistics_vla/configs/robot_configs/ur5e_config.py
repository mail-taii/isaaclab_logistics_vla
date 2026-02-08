"""
UR5e 机器人配置：场景加载、EE frame、动作。

资源来自 curobo content（ur_description/ur5e.urdf + meshes），已复制到 assets/robots/ur5e/。
Isaac Lab 仅支持 USD，需先用 scripts/convert_ur5e_urdf_to_usd.py 将 ur5e.urdf 转为 ur5e.usd 后再运行评估。
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

from isaaclab_logistics_vla import ISAACLAB_LOGISTICS_VLA_EXT_DIR
from isaaclab_logistics_vla.utils.register import register

# UR5e 资源路径；需先运行 scripts/convert_ur5e_urdf_to_usd.py 生成 ur5e.usd
UR5E_ASSET_DIR = os.path.join(
    ISAACLAB_LOGISTICS_VLA_EXT_DIR,
    "isaaclab_logistics_vla",
    "assets",
    "robots",
    "ur5e",
)
UR5E_USD_PATH = os.path.join(UR5E_ASSET_DIR, "ur5e.usd")

_spawn_cfg = sim_utils.UsdFileCfg(
    usd_path=UR5E_USD_PATH,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=0,
        fix_root_link=True,
    ),
)


@register.add_robot("ur5e")
@configclass
class UR5eCfg(ArticulationCfg):
    spawn = _spawn_cfg
    init_state = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -0.5,
            "elbow_joint": 0.5,
            "wrist_1_joint": -0.5,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
        pos=(0.96781, 2.28535, 0.216),
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    actuators = {
        # 悬空臂需较大刚度/阻尼才能抗重力，否则在策略控制频率有限时会持续下沉
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.14,

            # --- 参数调优核心区域 ---
            
            # 1. 刚度：依然需要保持较高，以对抗重力
            stiffness={
                "shoulder_.*": 3000.0, 
                "elbow_.*": 2500.0,
                "wrist_.*": 2000.0,
            },
            
            # 2. 阻尼：设为刚度的 1/10 左右，消除弹簧感
            damping={
                "shoulder_.*": 300.0,  # 3000 * 0.1
                "elbow_.*": 250.0,
                "wrist_.*": 200.0,
            },

            # 3. 【神器】电枢惯量 (Armature)：
            # 这是一个“低通滤波器”。值越大，运动越平滑，越没有“数码味”。
            # UR5e 这种级别的臂，0.1 到 10.0 都可以试。建议从 5.0 开始。
            armature=1.0, 

            # 4. 【神器】摩擦力 (Friction)：
            # 模拟关节生锈/咬合的感觉。
            # 这东西能帮刚度“省力”，让它不动的时候更稳。
            friction=10.0, 
        ),
    }


@register.add_eeframe_configs("ur5e_eeframe")
@configclass
class UR5eFrameTransformerCfg(FrameTransformerCfg):
    prim_path = "{ENV_REGEX_NS}/Robot/base_link"
    debug_vis = False
    # UrdfConverter 可能合并 fixed joint，tool0 未必单独存在；用 wrist_3_link 作为末端参考
    target_frames = [
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
            name="ee_tcp",
            offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ),
    ]
