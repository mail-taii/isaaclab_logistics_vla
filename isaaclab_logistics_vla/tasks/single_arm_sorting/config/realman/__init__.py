# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Realman robot configuration for single arm sorting task with custom scene.

场景配置：
- 操作台（从 simple_01.usd 加载，包含源箱子和目标箱子）
- Realman 机器人
- 相机（输出场景画面）
"""

import os
from dataclasses import MISSING

import gymnasium as gym

from ...single_arm_sorting_env_cfg import SingleArmSortingEnvCfg
from ... import mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_logistics_vla import ISAACLAB_LOGISTICS_VLA_EXT_DIR

##
# Robot Configuration
##

# 获取机器人USD文件的路径
REALMAN_USD_PATH = os.path.join(
    ISAACLAB_LOGISTICS_VLA_EXT_DIR,
    "isaaclab_logistics_vla",
    "assets",
    "robots",
    "realman",
    "realman_no_wheel.usd"
)

# Realman 机器人配置
REALMAN_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=REALMAN_USD_PATH,
        # 关闭重力以降低基座漂移风险；当前版本不支持 fixed_base 参数
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "l_joint[1-7]": 0.0,
            "left_right_joint": 0.04,
            "right_right_joint": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(-1, 0, 0, 0)
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["l_joint[1-7]"],
            effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*right_joint"],
            effort_limit_sim=20.0,
            stiffness=500.0,
            damping=10.0,
        )
    },
)

# 获取场景USD文件的路径
SCENE_USD_PATH = os.path.join(
    ISAACLAB_LOGISTICS_VLA_EXT_DIR,
    "isaaclab_logistics_vla",
    "assets",
    "sence",
    "simple_01.usd"
)

##
# Scene Configuration
##

@configclass
class RealmanSingleArmSortingSceneCfg(InteractiveSceneCfg):
    """Realman 机器人的单臂分拣场景配置，使用自定义场景USD文件。
    
    场景包含：
    - 操作台（从 simple_01.usd 加载，包含源箱子和目标箱子）
    - Realman机器人
    - 相机（用于输出场景画面）
    """
    
    # 加载自定义场景USD（包含操作台和两个箱子）
    base_scene = AssetBaseCfg(
        prim_path="/World/BaseScene",
        spawn=UsdFileCfg(usd_path=SCENE_USD_PATH),
    )
    
    # 机器人（会在环境配置中设置）
    robot: ArticulationCfg = MISSING
    
    # 物体（会在环境配置中设置）
    object: RigidObjectCfg = MISSING
    
    # 末端执行器帧（会在环境配置中设置）
    ee_frame: FrameTransformerCfg = MISSING
    
    # 源区域（source area）- 用于指标计算（使用 RigidObjectCfg 以支持观测函数）
    source_area = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SourceArea",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.04, 1.48, 0.91], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.01),  # 小的标记区域
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),  # 无质量，仅作标记
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 红色标记
        ),
    )
    
    # 目标区域（target area）- 用于指标计算（使用 RigidObjectCfg 以支持观测函数）
    target_area = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetArea",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 1.48, 0.91], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.01),  # 小的标记区域
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),  # 无质量，仅作标记
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # 绿色标记
        ),
    )
    
    # 地面
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    
    # 光源
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    # 相机配置：用于输出场景画面（benchmark输出要求）
    scene_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-2.0, -0.49, 4.0),  # 相机位置：从上方观察场景
            rot=(0.707, -0.707, 0.0, 0.0),  # 相机旋转：俯视角度
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1000.0),
        ),
        height=720,
        width=1280,
        data_types=["rgb"],
        debug_vis=False,
    )


##
# Observations Configuration (with image)
##

@configclass
class RealmanObservationsCfg:
    """观测配置，包含图像和机器人位姿。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组：包含图像和状态信息。"""

        # 图像观测（场景画面）
        rgb_image = ObsTerm(
            func=mdp.scene_rgb_image,
            params={"camera_cfg": SceneEntityCfg("scene_camera")},
        )
        
        # 机器人状态观测（位姿信息）
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pos = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        object_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_pos = ObsTerm(func=mdp.target_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            # 图像观测无法与其他标量观测拼接，设置为 False 以字典形式返回
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Environment Configuration
##

@configclass
class RealmanSingleArmSortingEnvCfg(SingleArmSortingEnvCfg):
    """Realman 机器人配置的单臂分拣环境。"""

    def __post_init__(self):
        """后初始化，设置机器人、物体、相机和动作配置。"""
        # 保存原始场景配置的参数
        num_envs = self.scene.num_envs
        env_spacing = self.scene.env_spacing
        
        # 使用新的场景配置（包含场景USD和相机）
        self.scene = RealmanSingleArmSortingSceneCfg(
            num_envs=num_envs,
            env_spacing=env_spacing,
        )
        
        # 使用外部机器人配置（从realman_no_wheel.usd加载）
        # 这样更方便操作和控制机器人，不受USD场景中机器人的限制
        self.scene.robot = REALMAN_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 设置物体配置
        # 注意：物体初始位置需要与场景USD中箱子的位置匹配
        # 
        # 物体位置说明：
        # - pos=[x, y, z]: 物体的中心位置（单位：米）
        # - x, y: 水平位置，应该位于箱子内部
        # - z: 垂直位置，应该是：箱子底部高度 + 箱子内部深度/2 + 物体高度/2
        #     例如：如果箱子底部在z=0.1，内部深度0.15m，物体高度0.05m
        #     则 z ≈ 0.1 + 0.15/2 + 0.05/2 = 0.175
        #
        # 当前配置的位置 [0.4, 0.0, 0.05] 需要根据实际场景调整
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.05]),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),  # 小型SKU包裹 (5cm立方体)
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 轻量包裹
            ),
        )
        
        # 设置末端执行器帧（Realman机器人）
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link_underpan",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_left_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),  # TCP偏移
                    ),
                )
            ]
        )
        
        # 使用包含图像的观测配置
        self.observations = RealmanObservationsCfg()
        
        # 设置动作配置：使用差分 IK，让末端执行器朝向 source 区域
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["l_joint[1-7]"],
            body_name="panda_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",  # 阻尼最小二乘，稳定性较好
                ik_params={"lambda_val": 0.05},
            ),
            body_offset=None,
            scale=0.5,  # 缩小一步位移，避免大幅摆动
        )
        
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*right_joint"],
            open_command_expr={".*right_joint": 0.04},
            close_command_expr={".*right_joint": 0.0},
        )
        
        # 调用父类初始化（这会设置其他配置，如decimation、episode_length等）
        super().__post_init__()


##
# Register Gym environments
##

gym.register(
    id="Isaac-Logistics-SingleArmSorting-Realman-v0",
    entry_point="isaaclab_logistics_vla.tasks.single_arm_sorting.single_arm_sorting_env:SingleArmSortingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:RealmanSingleArmSortingEnvCfg",
    },
)
