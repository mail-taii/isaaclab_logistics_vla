"""
评估侧机器人注册表：多机器人接入时，按 robot_id 提供 arm_dof、平台关节、Curobo IK 配置等。
新机器人只需在此注册一条，并在 configs/robot_configs/ 下提供可选 Curobo yml（EE 模式需要）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RobotEvalConfig:
    """单机器人在评估/IK 侧需要的配置，与场景中的 asset 名称可一致或映射。"""

    robot_id: str
    """注册表键，与 --robot_id 或 env 中机器人类型对应，如 realman_dual_left_arm。"""

    arm_dof: int
    """用于 IK 的手臂关节数（如左臂 7）。"""

    platform_joint_name: Optional[str] = None
    """可动平台关节名，若有则用于臂基高度变换；无则 None。"""

    # Curobo IK（EE 模式用）
    curobo_yml_name: Optional[str] = None
    """configs/robot_configs/ 下的 Curobo 左臂/单臂 yml 文件名，如 realman_left_arm.yml。None 表示不初始化 IK。"""

    curobo_asset_folder: Optional[str] = None
    """assets/robots/ 下机器人资源目录名，用于覆盖 Curobo yml 中的 urdf_path/asset_root。"""

    curobo_urdf_name: Optional[str] = None
    """Curobo 使用的 URDF 文件名，如 realman_no_wheel.urdf。"""

    unnorm_key: Optional[str] = None
    """OpenVLA 等策略用的 dataset_statistics key，与机器人动作维度/归一化对应；None 时策略用默认（如 bridge_orig）。"""

    camera_config_key: str = "realman"
    """相机配置键，对应 configs/camera_configs 中的 key；每个机器人一套相机（prim_path 绑定的 link 不同），任务通用。"""

    scene_robot_key: Optional[str] = None
    """场景中加载的机器人 asset 键，用于 register.load_robot(scene_robot_key)；None 表示沿用任务默认（如 realman_franka_ee）。"""


# 注册表：robot_id -> RobotEvalConfig
REGISTRY: dict[str, RobotEvalConfig] = {
    "realman_dual_left_arm": RobotEvalConfig(
        robot_id="realman_dual_left_arm",
        arm_dof=7,
        platform_joint_name="platform_joint",
        curobo_yml_name="realman_left_arm.yml",
        curobo_asset_folder="realman",
        curobo_urdf_name="realman_no_wheel.urdf",
        unnorm_key="bridge_orig",
        camera_config_key="realman",
        scene_robot_key="realman_franka_ee",
    ),
    # 示例：
    # "xarm7": RobotEvalConfig(
    #     robot_id="xarm7",
    #     arm_dof=7,
    #     platform_joint_name=None,
    #     curobo_yml_name=None,
    #     curobo_asset_folder=None,
    #     curobo_urdf_name=None,
    #     unnorm_key="xarm7_orig",
    #     camera_config_key="xarm7",   # 对应 configs/camera_configs/xarm7.py（prim_path 绑定 xarm7 的 link）
    #     scene_robot_key="xarm7",     # register.load_robot("xarm7")，需在 configs/robot_configs 中注册
    # ),
}


def get_robot_eval_config(robot_id: str) -> RobotEvalConfig:
    """根据 robot_id 取评估配置；未注册则抛 KeyError。"""
    if robot_id not in REGISTRY:
        raise KeyError(
            f"未注册的 robot_id: {robot_id!r}. "
            f"已注册: {list(REGISTRY.keys())}. "
            "请在 evaluation/robot_registry.py 的 REGISTRY 中新增条目。"
        )
    return REGISTRY[robot_id]


def list_registered_robots() -> list[str]:
    """返回已注册的 robot_id 列表。"""
    return list(REGISTRY.keys())


# -----------------------------------------------------------------------------
# 接入新机器人：在 REGISTRY 中新增一条 RobotEvalConfig，并视需要添加 Curobo yml。
# 1. 必填：robot_id（唯一）、arm_dof（用于 IK 的关节数）。
# 2. 若有可动平台：填 platform_joint_name，否则 None。
# 3. 若需 EE 模式（末端控制 + IK）：填 curobo_yml_name、curobo_asset_folder、curobo_urdf_name，
#    并在 configs/robot_configs/ 下提供对应 yml（及 spheres/ 下的碰撞球配置）。
# 4. 若用 OpenVLA 等需 dataset_statistics 的策略：填 unnorm_key（与动作维度对应），否则 None 用策略默认。
# 5. 相机：填 camera_config_key，与 configs/camera_configs 中该机器人的 key 一致（任务通用，每个机器人一套相机）。
# 6. 场景机器人：填 scene_robot_key（register.load_robot 的键）；None 时沿用任务默认。评估时用 --robot_id 与注册表对应即可。
