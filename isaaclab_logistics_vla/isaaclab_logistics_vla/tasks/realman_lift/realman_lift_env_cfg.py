# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_logistics_vla import ISAACLAB_LOGISTICS_VLA_EXT_DIR

from . import mdp

# 获取机器人USD文件的路径
REALMAN_USD_PATH = os.path.join(
    ISAACLAB_LOGISTICS_VLA_EXT_DIR,
    "isaaclab_logistics_vla",
    "assets",
    "robots",
    "realman",
    "realman_no_wheel.usd"
)

RM_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=REALMAN_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"l_joint[1-7]": 0.0,
                   "left_right_joint":0.04,
                   "right_right_joint":0.04}, 
        pos=(0.25, -0.25, 0.3),
        rot=(-1,0,0,0)
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

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import  OffsetCfg
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = RM_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # end-effector sensor: will be populated by agent env cfg
    ee_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/base_link_underpan", # <-- 正确路径
        debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/panda_left_hand",
                name="ee_tcp", #
                    # 偏移量：从“手掌”原点向前 10.34cm (标准 Franka TCP)
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034), #
                    ),
            )
        ]
        #target_frame_prim_path="{ENV_REGEX_NS}/LiftObject",
    )
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LiftObject",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd",
            scale=(0.05, 0.05, 0.05), # 5cm 方块
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 1.5, 1.5)),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, -2, 1], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Tiled相机：旋转轨道视角，用于录制视频
    # 注意：TiledCameraCfg 会自动创建相机，不需要单独的 CameraCfg
    tiled_camera_top = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/orbital_camera",
        update_period=0.1,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-2.0, -0.49, 4.0),  # 固定位置：用户推荐的视角（提高高度）
            rot=(0.707, -0.707, 0.0, 0.0),  # 初始朝向：看向场景中心
            convention="world",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link_underpan",  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(1.25, 1.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_joints = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_joint[1-7]"], 
        scale=1.0,
        use_default_offset=False,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        # 匹配我（Gemini）在 RM_CONFIG 中添加的执行器
        joint_names=[".*right_joint"], 
        open_command_expr={".*right_joint": 0.04},
        close_command_expr={".*right_joint": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.25, 0.25)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_joints_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_joint[1-7]"]),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=2, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
