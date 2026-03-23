"""
环境碰撞箱/障碍物的统一描述。

目前主要用于：
- 调试 IK（例如 Curobo）：知道哪些地方“理论上不该去”；
- 以后若要接 Curobo 的 world 配置，可以直接从这里读出所有静态 box。

坐标系：Isaac 世界系，单位：米。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EnvBox:
    name: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]  # (Lx, Ly, Lz)

    @property
    def half_size(self) -> Tuple[float, float, float]:
        return tuple(s * 0.5 for s in self.size)  # type: ignore[return-value]


def get_env_collision_boxes() -> List[EnvBox]:
    """
    返回当前 ss_st_sparse 系列任务中，用作“障碍物/工作区”的静态箱体列表。

    目前包括：
    - 前方三只源箱：s_box_1/2/3
    - 后方三只目标箱：t_box_1/2/3
    - 桌子、传送带先不做近似 box，避免和真实几何差太多（仍由 PhysX 负责）。
    """
    # 与 BaseOrderSceneCfg 中 WORK_BOX_PARAMS / s_box_* / t_box_* 保持一致
    box_size = (0.36, 0.56, 0.23)  # (X_LENGTH, Y_LENGTH, Z_LENGTH)

    boxes = [
        # 源箱（机器人前方）
        EnvBox("s_box_1", (1.57989, 1.33474, 0.750), box_size),
        EnvBox("s_box_2", (1.02500, 1.33614, 0.725), box_size),
        EnvBox("s_box_3", (0.51025, 1.33614, 0.750), box_size),
        # 目标箱（机器人后方）
        EnvBox("t_box_1", (1.57989, 3.44290, 0.820), box_size),
        EnvBox("t_box_2", (1.02500, 3.44290, 0.820), box_size),
        EnvBox("t_box_3", (0.51000, 3.44290, 0.820), box_size),
    ]
    return boxes


__all__ = ["EnvBox", "get_env_collision_boxes"]

