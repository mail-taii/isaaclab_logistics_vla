# Copyright (c) 2025, Logistics VLA extension contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""机器人描述：指向 **已生成** 的 kinematics YAML，供 ``CuroboPlanner`` 加载。

YAML 须由使用方用仓库 ``scripts/generate_curobo_robot_kinematics_yaml.py`` 或等价流程从 URDF 生成；
封装包内 **不** 负责 URDF → YAML。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RobotSpec:
    """``cache_path`` 为 **必须已存在** 的 kinematics YAML（``RobotConfig.from_dict`` 可读）。

    ``urdf_path`` / ``base_link`` / ``*_ee_link`` 仅作文档与业务侧记录，**不参与** ``CuroboPlanner`` 加载逻辑
    （真实链名已固化在 YAML 中）。
    """

    cache_path: str
    urdf_path: Optional[str] = None
    base_link: str = "dual_rm_75b_description_platform_base_link"
    left_ee_link: str = "panda_left_hand"
    right_ee_link: str = "panda_right_hand"

    @classmethod
    def from_kinematics_yaml(
        cls,
        kinematics_yaml: str,
        *,
        urdf_path: Optional[str] = None,
        base_link: str = "dual_rm_75b_description_platform_base_link",
        left_ee_link: str = "panda_left_hand",
        right_ee_link: str = "panda_right_hand",
    ) -> RobotSpec:
        """主入口：``kinematics_yaml`` 为已生成的缓存文件路径。"""
        return cls(
            cache_path=kinematics_yaml,
            urdf_path=urdf_path,
            base_link=base_link,
            left_ee_link=left_ee_link,
            right_ee_link=right_ee_link,
        )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        *,
        cache_path: str,
        base_link: str = "dual_rm_75b_description_platform_base_link",
        left_ee_link: str = "panda_left_hand",
        right_ee_link: str = "panda_right_hand",
    ) -> RobotSpec:
        """兼容旧名：``urdf_path`` 仅作记录；**必须**提供已存在的 ``cache_path`` YAML。"""
        return cls(
            cache_path=cache_path,
            urdf_path=urdf_path,
            base_link=base_link,
            left_ee_link=left_ee_link,
            right_ee_link=right_ee_link,
        )

    @classmethod
    def from_robot_config_yaml(
        cls,
        config_yaml: str,
        *,
        urdf_path: Optional[str] = None,
        base_link: str = "dual_rm_75b_description_platform_base_link",
        left_ee_link: str = "panda_left_hand",
        right_ee_link: str = "panda_right_hand",
    ) -> RobotSpec:
        """与 ``from_kinematics_yaml`` 同义（历史命名）。"""
        return cls(
            cache_path=config_yaml,
            urdf_path=urdf_path,
            base_link=base_link,
            left_ee_link=left_ee_link,
            right_ee_link=right_ee_link,
        )
