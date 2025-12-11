# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the your_task_name environment.

TODO: 描述你的任务
例如：This implements task X.X.X-X: Description of your task.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class YourTaskNameSceneCfg(InteractiveSceneCfg):
    """Configuration for the your_task_name scene.
    
    TODO: 描述场景中包含哪些对象
    """

    # Robot (will be set by agent env cfg)
    robot: ArticulationCfg = MISSING

    # Object (will be set by agent env cfg)
    object: RigidObjectCfg = MISSING

    # TODO: 添加场景中的其他对象
    # 例如：源区域、目标区域、障碍物等

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # TODO: 定义你的观测项
        # State observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pos = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        # object_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_pos = ObsTerm(func=mdp.target_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Will be set by agent env cfg
    arm_action: MISSING = MISSING
    gripper_action: MISSING = MISSING


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # TODO: 定义你的奖励函数
    # 示例：
    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    # grasping_success = RewTerm(func=mdp.object_is_grasped, params={"threshold": 0.02}, weight=10.0)

    # Action smoothness
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # TODO: 添加其他终止条件
    # 例如：
    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    # )


##
# Environment configuration
##


@configclass
class YourTaskNameEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the your_task_name environment.
    
    TODO: 描述任务目标
    """

    # Scene settings
    scene: YourTaskNameSceneCfg = YourTaskNameSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 5.0  # TODO: 根据任务调整

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
