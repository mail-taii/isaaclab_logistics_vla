# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka robot configuration for single arm sorting task."""

import gymnasium as gym

from ...single_arm_sorting_env_cfg import SingleArmSortingEnvCfg, SingleArmSortingSceneCfg
from ... import mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils

##
# Robot Configuration
##

# Franka Panda robot configuration for logistics tasks
FRANKA_LOGISTICS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
            "panda_finger_joint.*": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)

# End-effector frame
# NOTE:
# Isaac Lab's FrameTransformerCfg requires `target_frames` to be specified.
# For this single-arm Franka setup, we simply track the TCP frame of the panda hand itself.
FRANKA_EE_FRAME_CFG = FrameTransformerCfg(
    # Source frame (we use the hand link as the source frame)
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
    # Target frames to track w.r.t. the source frame.
    # Here we track the same hand link, with an optional small TCP offset if needed in the future.
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
            name="ee_tcp",
            offset=OffsetCfg(),  # identity offset; can be tuned if a TCP offset is desired
        )
    ],
    debug_vis=False,
)

##
# Environment Configuration
##


@configclass
class FrankaSingleArmSortingEnvCfg(SingleArmSortingEnvCfg):
    """Franka robot configuration for single arm sorting."""

    def __post_init__(self):
        """Post initialization."""
        # Call parent post_init
        super().__post_init__()

        # --------------------------------------------------------------------- #
        # Scene: robot and task objects
        # --------------------------------------------------------------------- #
        # Override scene to add robot and object
        self.scene.robot = FRANKA_LOGISTICS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.05]),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),  # Small SKU package (5cm cube)
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # Light package
            ),
        )

        # Override source / target areas to be rigid objects (not just XForm prims)
        # so that metrics / observations can use `.data` safely.
        self.scene.source_area = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SourceArea",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.04, 2.0, 0.60], rot=[0.707, 0.0, 0.0, 0.707]),
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.35, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        )
        self.scene.target_area = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetArea",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 2.0, 0.60], rot=[0.707, 0.0, 0.0, 0.707]),
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.35, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        )

        # --------------------------------------------------------------------- #
        # Sensors / frames
        # --------------------------------------------------------------------- #
        # Add end-effector frame
        self.scene.ee_frame = FRANKA_EE_FRAME_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/panda_hand")

        # --------------------------------------------------------------------- #
        # Actions
        # --------------------------------------------------------------------- #
        # Override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0},
        )


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Logistics-SingleArmSorting-Franka-v0",
    entry_point="isaaclab_logistics_vla.tasks.single_arm_sorting.single_arm_sorting_env:SingleArmSortingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:FrankaSingleArmSortingEnvCfg",
    },
)

