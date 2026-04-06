# Copyright (c) 2025, Logistics VLA extension contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""机器人规划配置：显式描述 URDF、可选 kinematics 缓存与末端链参数，供 ``CuroboPlanner`` 使用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RobotSpec:
    """与 ``load_realman_config`` / ``generate_robot_config_from_urdf`` 对齐的对外合同。

    - ``cache_path`` 指向由本包从 URDF 生成并写入的 kinematics YAML（``RobotConfig.from_dict`` 可读）。
    - 若文件不存在，则用 ``urdf_path`` 与 ``base_link`` / ``*_ee_link`` 重新生成并可写入 ``cache_path``。
    """

    urdf_path: str
    cache_path: Optional[str] = None
    base_link: str = "dual_rm_75b_description_platform_base_link"
    left_ee_link: str = "panda_left_hand"
    right_ee_link: str = "panda_right_hand"

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        *,
        cache_path: Optional[str] = None,
        base_link: str = "dual_rm_75b_description_platform_base_link",
        left_ee_link: str = "panda_left_hand",
        right_ee_link: str = "panda_right_hand",
    ) -> RobotSpec:
        """从 URDF 声明机器人；可选指定缓存 YAML 路径（命中则直接加载）。"""
        return cls(
            urdf_path=urdf_path,
            cache_path=cache_path,
            base_link=base_link,
            left_ee_link=left_ee_link,
            right_ee_link=right_ee_link,
        )

    @classmethod
    def from_robot_config_yaml(
        cls,
        config_yaml: str,
        *,
        urdf_path: str,
        base_link: str = "dual_rm_75b_description_platform_base_link",
        left_ee_link: str = "panda_left_hand",
        right_ee_link: str = "panda_right_hand",
    ) -> RobotSpec:
        """显式以「机器人配置文件」为主入口：``config_yaml`` 作为 ``cache_path`` 传给加载逻辑。

        当 ``config_yaml`` 不存在时，用 ``urdf_path`` 生成并写入该路径。
        """
        return cls(
            urdf_path=urdf_path,
            cache_path=config_yaml,
            base_link=base_link,
            left_ee_link=left_ee_link,
            right_ee_link=right_ee_link,
        )
