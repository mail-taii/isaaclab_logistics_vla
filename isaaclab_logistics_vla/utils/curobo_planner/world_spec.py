# Copyright (c) 2025, Logistics VLA extension contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""世界障碍描述：与 ``CuroboPlanner.set_world`` 同一语义（机器人约定坐标系下的长方体）。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping

import numpy as np
import yaml


def _to_vec3(x: Any, *, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"{name} 须为长度 3 的序列，得到 shape {a.shape}")
    return a


def _to_quat_wxyz(x: Any) -> np.ndarray:
    if x is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 4:
        raise ValueError(f"quaternion 须为长度 4 (wxyz)，得到 shape {a.shape}")
    return a


@dataclass
class WorldSpec:
    """长方体障碍列表。通过 ``from_yaml`` / ``from_cuboids`` 构造，再交给 ``CuroboPlanner.apply_world``。"""

    obstacles: List[MutableMapping[str, Any]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> WorldSpec:
        return cls(obstacles=[])

    @classmethod
    def from_cuboids(cls, obstacles: List[Dict[str, Any]]) -> WorldSpec:
        """与 ``set_world`` 相同字段：``position``、``size`` 或 ``dims``、可选 ``quaternion`` (wxyz)、``name``。"""
        if not obstacles:
            return cls.empty()
        out: List[MutableMapping[str, Any]] = []
        for i, raw in enumerate(obstacles):
            out.append(cls._normalize_cuboid_dict(raw, index=i))
        return cls(obstacles=out)

    @classmethod
    def from_yaml(cls, path: str) -> WorldSpec:
        """YAML 格式示例::

            version: 1
            cuboids:
              - name: table
                position: [0.0, 0.0, 0.5]
                dims: [2.0, 1.0, 0.05]
                quaternion: [1.0, 0.0, 0.0, 0.0]

        根键可为 ``cuboids`` 或 ``obstacles``。
        """
        p = os.path.expanduser(path)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            return cls.empty()
        raw_list = data.get("cuboids") or data.get("obstacles") or []
        if not isinstance(raw_list, list):
            raise ValueError("YAML 中 cuboids/obstacles 须为列表")
        obs: List[Dict[str, Any]] = []
        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                raise ValueError(f"cuboids[{i}] 须为字典")
            obs.append(dict(item))
        return cls.from_cuboids(obs)

    @staticmethod
    def _normalize_cuboid_dict(raw: Dict[str, Any], *, index: int) -> MutableMapping[str, Any]:
        if "position" not in raw:
            raise KeyError(f"障碍[{index}] 缺少 position")
        size = raw.get("size", raw.get("dims"))
        if size is None:
            raise KeyError(f"障碍[{index}] 需要 size 或 dims")
        name = raw.get("name", f"obs_{index}")
        return {
            "name": name,
            "position": _to_vec3(raw["position"], name="position"),
            "size": _to_vec3(size, name="size/dims"),
            "quaternion": _to_quat_wxyz(raw.get("quaternion")),
        }

    def to_planner_obstacles(self) -> List[Dict[str, Any]]:
        """转为 ``CuroboPlanner.set_world`` 所需列表。"""
        planner_obs: List[Dict[str, Any]] = []
        for o in self.obstacles:
            planner_obs.append(
                {
                    "name": str(o.get("name", "obs")),
                    "position": np.asarray(o["position"], dtype=np.float64).reshape(3),
                    "size": np.asarray(o["size"], dtype=np.float64).reshape(3),
                    "quaternion": np.asarray(o["quaternion"], dtype=np.float64).reshape(4),
                }
            )
        return planner_obs
