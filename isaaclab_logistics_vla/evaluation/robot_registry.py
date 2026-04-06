"""
评估侧机器人注册表：多机器人接入时，按 robot_id 提供 arm_dof、平台关节、臂基偏置等。
``arm_base_offset_in_root`` + ``platform_joint`` 用于
``CuRoboPlanPolicy`` 中 ``combine_frame_transforms`` 计算臂基世界位姿。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RobotEvalConfig:
    """单机器人在评估 / cuRobo 目标系侧需要的配置。"""

    robot_id: str

    arm_dof: int
    """用于 IK 的手臂关节数（如左臂 7）。"""

    platform_joint_name: Optional[str] = None
    """可动平台关节名；用于臂基高度 ``offset_z + platform_joint_value``。"""

    arm_base_offset_in_root: Optional[Tuple[float, float, float]] = None
    """根 link 到臂基（CuRobo ``base_link``，如 ``platform_base_link``）的平移，在 **根 link 坐标系** 下。
    臂基世界位置 = ``root_pos + R_root @ (offset + [0, 0, platform_joint])``。"""

    curobo_yml_name: Optional[str] = None
    curobo_asset_folder: Optional[str] = None
    curobo_urdf_name: Optional[str] = None

    left_arm_joint_names: Optional[Tuple[str, ...]] = None

    unnorm_key: Optional[str] = None
    camera_config_key: str = "realman"
    scene_robot_key: Optional[str] = None
    action_config_key: Optional[str] = None
    init_pos: Optional[Tuple[float, float, float]] = None
    init_rot: Optional[Tuple[float, float, float, float]] = None


REGISTRY: dict[str, RobotEvalConfig] = {
    # 双臂 Realman + Franka 手：臂基偏置（平台系共用）
    "realman_dual_left_arm": RobotEvalConfig(
        robot_id="realman_dual_left_arm",
        arm_dof=7,
        platform_joint_name="platform_joint",
        arm_base_offset_in_root=(0.0, -0.11663, 0.271),
        left_arm_joint_names=(
            "l_joint1",
            "l_joint2",
            "l_joint3",
            "l_joint4",
            "l_joint5",
            "l_joint6",
            "l_joint7",
        ),
        curobo_yml_name="realman_left_arm.yml",
        curobo_asset_folder="realman",
        curobo_urdf_name="realman_no_wheel.urdf",
        unnorm_key="bridge_orig",
        camera_config_key="realman",
        scene_robot_key="realman_franka_ee",
        action_config_key="realman_franka_ee_actionscfg",
    ),
    # 别名：便于 ``CUROBO_PLAN_ROBOT_ID`` 语义化（双臂场景）
    "realman_franka_dual": RobotEvalConfig(
        robot_id="realman_franka_dual",
        arm_dof=7,
        platform_joint_name="platform_joint",
        arm_base_offset_in_root=(0.0, -0.11663, 0.271),
        left_arm_joint_names=(
            "l_joint1",
            "l_joint2",
            "l_joint3",
            "l_joint4",
            "l_joint5",
            "l_joint6",
            "l_joint7",
        ),
        curobo_yml_name="realman_left_arm.yml",
        curobo_asset_folder="realman",
        curobo_urdf_name="realman_no_wheel.urdf",
        unnorm_key="bridge_orig",
        camera_config_key="realman",
        scene_robot_key="realman_franka_ee",
        action_config_key="realman_franka_ee_actionscfg",
    ),
    "ur5e": RobotEvalConfig(
        robot_id="ur5e",
        arm_dof=6,
        platform_joint_name=None,
        arm_base_offset_in_root=None,
        curobo_yml_name="ur5e.yml",
        curobo_asset_folder="ur5e",
        curobo_urdf_name="ur5e.urdf",
        unnorm_key="berkeley_autolab_ur5",
        camera_config_key="ur5e",
        scene_robot_key="ur5e",
        action_config_key="ur5e_actionscfg",
        init_pos=(1.0, 2.0, 1.0),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    ),
}


def get_robot_eval_config(robot_id: str) -> RobotEvalConfig:
    if robot_id not in REGISTRY:
        raise KeyError(
            f"未注册的 robot_id: {robot_id!r}. 已注册: {list(REGISTRY.keys())}. "
            "请在 evaluation/robot_registry.py 的 REGISTRY 中新增条目。"
        )
    return REGISTRY[robot_id]


def list_registered_robots() -> list[str]:
    return list(REGISTRY.keys())
