# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the single arm sorting environment.

This implements task 3.1.1-1: Single arm sorting of light random small SKU packages
to category-divided staging compartments.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class SingleArmSortingSceneCfg(InteractiveSceneCfg):
    """Configuration for the single arm sorting scene with a robot and a small SKU package."""

    # Robot (single arm - Franka Panda)
    robot: ArticulationCfg = MISSING  # Will be set by agent env cfg

    # Small SKU package (light random small package)
    object: RigidObjectCfg | RigidObjectCollectionCfg = MISSING  # Will be set by agent env cfg

    # End-effector frame (used by observations / rewards / metrics)
    # Concrete agents (e.g. Franka, Realman) must set this to a valid FrameTransformerCfg.
    ee_frame: FrameTransformerCfg = MISSING

    # Source area (where packages start)
    source_area = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SourceArea",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.04, 2.0, 0.60],
            rot=[0.707, 0, 0, 0.707],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.5, 1.5, 1.0),  # ← 在 X/Y 方向放大 1.5 倍
        ),
    )

    # Target area (staging compartments)
    target_area = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetArea",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.4, 2.0, 0.60],
            rot=[0.707, 0, 0, 0.707],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.5, 1.5, 1.0),  # ← 同样放大
        ),
    )

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

        # State observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pos = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        object_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_pos = ObsTerm(func=mdp.target_position_in_robot_root_frame)
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

    # Reaching object
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # Grasping success
    grasping_success = RewTerm(
        func=mdp.object_is_grasped, params={"threshold": 0.02}, weight=10.0
    )

    # Lifting object
    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.05}, weight=5.0
    )

    # Reaching target
    reaching_target = RewTerm(
        func=mdp.object_target_distance, params={"std": 0.1}, weight=15.0
    )

    # Placing success
    placing_success = RewTerm(
        func=mdp.object_is_placed_correctly, params={"threshold": 0.05}, weight=20.0
    )

    # Task completion
    task_completion = RewTerm(func=mdp.task_is_completed, weight=50.0)

    # Action smoothness
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.active_object_height_below_minimum,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("object")},
    )


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Randomize base object position around source area on each reset
    randomize_object_position = EventTerm(
        func=mdp.randomize_object_positions,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "source_cfg": SceneEntityCfg("source_area"),
            "pos_range": ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)),
        },
    )

    # Placeholder for mass / scale randomization (no-op but keeps API)
    randomize_object_properties = EventTerm(
        func=mdp.randomize_object_properties,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "mass_range": (0.05, 0.2),
            "scale_range": (0.8, 1.2),
        },
    )


##
# Environment configuration
##


@configclass
class SingleArmSortingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the single arm sorting environment.

    Task: Sort light random small SKU packages with one arm to category-divided staging compartments.
    """

    # Scene settings
    scene: SingleArmSortingSceneCfg = SingleArmSortingSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Re-enable generic randomization events for object position / properties
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 5.0

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

