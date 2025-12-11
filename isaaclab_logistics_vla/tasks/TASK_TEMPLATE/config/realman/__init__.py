# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Realman robot configuration for your_task_name task.

TODO: 根据你的任务配置机器人
参考 single_arm_sorting/config/realman/__init__.py
"""

import os
from dataclasses import MISSING

import gymnasium as gym

# TODO: 导入基础环境配置
# from ...your_task_name_env_cfg import YourTaskNameEnvCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=0
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

# TODO: 获取场景USD文件的路径（如果有）
# SCENE_USD_PATH = os.path.join(...)

##
# Scene Configuration
##

# TODO: 实现场景配置
# @configclass
# class RealmanYourTaskNameSceneCfg(InteractiveSceneCfg):
#     """Realman 机器人的任务场景配置"""
#     pass

##
# Environment Configuration
##

# TODO: 实现环境配置
# @configclass
# class RealmanYourTaskNameEnvCfg(YourTaskNameEnvCfg):
#     """Realman 机器人配置的任务环境"""
#     pass

##
# Register Gym environments
##

# TODO: 注册环境
# gym.register(
#     id="Isaac-Logistics-YourTaskName-Realman-v0",
#     entry_point="isaaclab_logistics_vla.tasks.your_task_name.your_task_name_env:YourTaskNameEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}:RealmanYourTaskNameEnvCfg",
#     },
# )
