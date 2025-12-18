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

from ...single_arm_sorting_env_cfg import SingleArmSortingEnvCfg, ActionsCfg
from ... import mdp
from ...object_randomization import (
    TargetObjectRandomizationCfg,
    TargetObjectSpec,
    build_object_collection_cfg,
    randomize_target_objects,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
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
from isaaclab.managers import EventTermCfg as EventTerm

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
        # 固定底座：真正固定机器人底座，防止飞起来
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,  # 固定底座
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,  # 增加速度迭代，有助于prismatic关节的稳定性
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "l_joint[1-7]": 0.0,  # 左手臂初始位置
            "r_joint[1-7]": 0.0,  # 右手臂初始位置
            "left_right_joint": 0.04,  # 左手夹爪初始位置
            "right_right_joint": 0.04,  # 右手夹爪初始位置
            "platform_joint": 0.9,  # 升降关节：控制身体在底座上的高度（单位：米）
            # 增加这个值可以抬高身体部分，例如 0.3 表示身体抬高 30cm
        },
        # 机器人位置和朝向
        # pos=(x, y, z): 机器人底座中心位置（底座的位置，不是身体的位置）
        # z坐标是底座的位置，身体高度由 platform_joint 控制
        # rot=(w, x, y, z): 四元数旋转，控制机器人朝向
        # (1, 0, 0, 0) = 无旋转
        # (0, 0, 0, 1) = 180度绕z轴（水平旋转180度，前后颠倒）
        # (0, 0, 1, 0) = 180度绕y轴（左右颠倒）
        # (0, 1, 0, 0) = 180度绕x轴（上下颠倒）
        pos=(0.2, 1.0, 0.0),  # 底座位置，z=0.0 让底座在地面上
        rot=(0, 0, 0, 1)  # 绕z轴旋转180度，修正朝向（如果还是反的，可以尝试 (0, 0, 1, 0)）
    ),
    actuators={
        "left_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["l_joint[1-7]"],
            effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "right_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["r_joint[1-7]"],
            effort_limit_sim=60.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*right_joint|.*left_joint"],
            effort_limit_sim=20.0,
            stiffness=500.0,
            damping=10.0,
        ),
        "platform_joint": ImplicitActuatorCfg(
            joint_names_expr=["platform_joint"],
            effort_limit_sim=1000.0,  # 更大的力限制，支撑身体重量（增加到1000N）
            stiffness=10000.0,  # 更高的刚度，保持位置（prismatic关节需要更高的刚度，增加到10000）
            damping=1000.0,  # 更高的阻尼，防止震荡（增加到1000）
        ),
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
    # 注意：确保场景中的物体有碰撞属性，防止物体穿透
    # 重要：USD文件中的物体需要有碰撞网格才能检测碰撞
    # 如果USD文件本身没有碰撞网格，这里的设置可能无效
    base_scene = AssetBaseCfg(
        prim_path="/World/BaseScene",
        # 调整场景位置，使其靠近机器人
        # pos=[x, y, z]: 
        #   x: 左右方向（正数向右，负数向左）
        #   y: 前后方向（正数向前/远离机器人，负数向后/靠近机器人）
        #   z: 上下方向（正数向上，负数向下）
        # 机器人位置：(0.2, 1.0, 0.0)
        # 如果场景前移了（远离机器人），需要减小y值（设为负数）让它靠近机器人
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.4, 0.0],  # y=-0.5 向后移动50cm，让场景靠近机器人
            rot=[0.0, 0.0, 0.0, 0.0]  # 场景旋转（四元数，1.0表示无旋转）
        ),
        spawn=UsdFileCfg(
            usd_path=SCENE_USD_PATH,
            # 尝试确保场景中的所有刚体都有碰撞检测
            # 注意：这只会应用到有RigidBodyAPI的prim
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # 非kinematic，确保有碰撞
                rigid_body_enabled=True,  # 启用刚体
                solver_position_iteration_count=32,  # 增加位置迭代次数，提高碰撞精度
                solver_velocity_iteration_count=1,  # 增加速度迭代次数
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,  # 启用碰撞检测
                contact_offset=0.005,  # 接触偏移量（5mm），确保碰撞检测更敏感
                rest_offset=0.0,  # 静止偏移量
            ),
            # 设置场景物体的质量，防止一碰就飞起来
            # 较大的质量可以让场景更稳定
            mass_props=sim_utils.MassPropertiesCfg(
                mass=50.0,  # 设置较大的质量（50kg），让场景更稳定，不容易被推动
                # 如果还是太轻，可以增加到 100.0 或更大
            ),
        ),
    )
    
    # 机器人（会在环境配置中设置）
    robot: ArticulationCfg = MISSING
    
    # 物体（会在环境配置中设置）
    object: RigidObjectCfg | RigidObjectCollectionCfg = MISSING
    
    # 末端执行器帧（会在环境配置中设置）
    ee_frame: FrameTransformerCfg = MISSING
    
    # 源区域（source area）- 用于指标计算（使用 RigidObjectCfg 以支持观测函数）
    # 注意：设置为kinematic=False，确保有碰撞检测，防止物体穿透
    source_area = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SourceArea",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.04, 1.8, 0.75], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.CuboidCfg(
            # 放大 XY 尺寸以更好地覆盖箱子底部
            size=(0.5, 0.35, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=False,  # 非kinematic，确保有碰撞
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # 设置质量，确保有碰撞
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 红色标记
        ),
    )
    
    # 目标区域（target area）- 用于指标计算（使用 RigidObjectCfg 以支持观测函数）
    # 注意：设置为kinematic=False，确保有碰撞检测，防止物体穿透
    target_area = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetArea",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 1.8, 0.75], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.CuboidCfg(
            # 同样放大 XY 尺寸，保证目标区域能完全覆盖箱子底部
            size=(0.5, 0.35, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=False,  # 非kinematic，确保有碰撞
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # 设置质量，确保有碰撞
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
    
    # 相机配置：用于输出场景画面（benchmark输出或 livestream）
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
# Actions Configuration (with dual arms)
##

@configclass
class RealmanActionsCfg(ActionsCfg):
    """动作配置，包含两只手臂的动作。"""
    
    # 右手臂动作（在__post_init__中设置）
    right_arm_action: MISSING = MISSING


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


@configclass
class RealmanEventsCfg:
    """事件配置：在reset时进行目标物体随机化。"""

    randomize_targets: EventTerm = MISSING


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
        
        # 目标物体随机化：默认提供三种尺寸/颜色的立方体，用户可以通过
        # `object_randomization.asset_pool` 直接替换为任意 USD 或外部资源
        if not hasattr(self, "object_randomization") or self.object_randomization is None:
            # 使用 grasp 资产库中的真实物体（YCB + Mugs）
            # 使用绝对路径，确保 IsaacLab 能正确加载 USD 文件
            import os
            from pathlib import Path

            # 获取包的 assets 目录的绝对路径
            # __file__ = .../tasks/single_arm_sorting/config/realman/__init__.py
            # parent(0): realman
            # parent(1): config
            # parent(2): single_arm_sorting
            # parent(3): tasks
            # parent(4): isaaclab_logistics_vla  <-- 包根目录
            package_root = Path(__file__).resolve().parents[4]
            assets_dir = package_root / "assets" / "grasp"
            base_path = str(assets_dir)
            
            self.object_randomization = TargetObjectRandomizationCfg(
                asset_pool=[
                    # 使用自带刚体物理的 YCB 资产（Axis_Aligned_Physics）
                    TargetObjectSpec(
                        name="cracker_box",
                        usd_path=os.path.join(base_path, "YCB", "Axis_Aligned_Physics", "003_cracker_box.usd"),
                        mass=0.25,
                        scale=(1.0, 1.0, 1.0),
                        num_instances=2,
                    ),
                    TargetObjectSpec(
                        name="sugar_box",
                        usd_path=os.path.join(base_path, "YCB", "Axis_Aligned_Physics", "004_sugar_box.usd"),
                        mass=0.25,
                        scale=(1.0, 1.0, 1.0),
                        num_instances=2,
                    ),
                    TargetObjectSpec(
                        name="tomato_soup_can",
                        usd_path=os.path.join(base_path, "YCB", "Axis_Aligned_Physics", "005_tomato_soup_can.usd"),
                        mass=0.25,
                        scale=(1.0, 1.0, 1.0),
                        num_instances=2,
                    ),
                    TargetObjectSpec(
                        name="mustard_bottle",
                        usd_path=os.path.join(base_path, "YCB", "Axis_Aligned_Physics", "006_mustard_bottle.usd"),
                        mass=0.25,
                        scale=(1.0, 1.0, 1.0),
                        num_instances=2,
                    ),
                ],
                max_spawned_objects=4,
                # 位姿采样：严格模式下，XY 由 source_area 决定，这里只影响高度偏移和 yaw
                pose_range={
                    "x": (-0.08, 0.08),   # 严格模式下会被忽略
                    "y": (1.40, 1.56),    # 严格模式下会被忽略
                    "z": (0.92, 1.02),    # 严格模式下会被忽略
                    "z_offset": (0.01, 0.05),  # 相对 source_area 表面的高度偏移
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.3, 0.3),
                },
                min_separation=0.04,
                offstage_pose=(4.0, 4.0, 4.0),
            )

        # 将物体构建为集合，支持按需随机选择/生成多个目标
        self.scene.object = build_object_collection_cfg(
            self.object_randomization,
            prim_prefix="{ENV_REGEX_NS}/Object",
        )
        
        # 设置末端执行器帧（Realman机器人 - 两只手）
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link_underpan",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_left_hand",
                    name="left_ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),  # 左手TCP偏移
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_right_hand",
                    name="right_ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),  # 右手TCP偏移
                    ),
                ),
            ]
        )
        
        # 使用包含图像的观测配置
        self.observations = RealmanObservationsCfg()
        
        # 使用包含两只手臂的动作配置
        self.actions = RealmanActionsCfg()
        
        # 设置动作配置：使用差分 IK，让两只手的末端执行器都朝向 source 区域
        # 左手动作
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
        
        # 右手动作（添加右手臂控制）
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["r_joint[1-7]"],
            body_name="panda_right_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            body_offset=None,
            scale=0.5,
        )
        
        # 夹爪动作（控制两只手的夹爪）
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*right_joint|.*left_joint"],
            open_command_expr={".*right_joint": 0.04, ".*left_joint": 0.04},
            close_command_expr={".*right_joint": 0.0, ".*left_joint": 0.0},
        )

        # 重置阶段执行目标物随机化：严格在源区域内部采样目标物体的 XY 位置
        self.events = RealmanEventsCfg()
        self.events.randomize_targets = EventTerm(
            func=randomize_target_objects,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("object"),
                "randomization_cfg": self.object_randomization,
                "source_cfg": SceneEntityCfg("source_area"),
                # 与上方 source_area.spawn.size 对应：X=0.5, Y=0.35
                "source_size_xy": (0.5, 0.35),
            },
        )
        
        # 先设置episode长度，避免频繁重启（默认5秒太短）
        # 注意：必须在super().__post_init__()之前设置，否则会被父类覆盖
        self.episode_length_s = 60.0  # 60秒，让环境运行更长时间
        
        # 禁用或调整termination条件，避免提前结束
        # 覆盖terminations配置，禁用object_dropping（物体掉落）终止条件
        from ...single_arm_sorting_env_cfg import TerminationsCfg
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from ... import mdp as task_mdp
        
        self.terminations = TerminationsCfg()
        # 只保留time_out，移除object_dropping（避免物体掉落导致提前结束）
        self.terminations.object_dropping = DoneTerm(
            func=task_mdp.active_object_height_below_minimum,
            params={"minimum_height": -10.0, "object_cfg": SceneEntityCfg("object")},  # 设置极低阈值，实际不会触发
        )
        
        # 调用父类初始化（这会设置其他配置，如decimation等）
        super().__post_init__()
        
        # 再次确保episode_length_s被设置（防止被父类覆盖）
        self.episode_length_s = 60.0


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
