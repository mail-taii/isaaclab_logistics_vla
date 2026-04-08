"""
Realman → cuRobo 坐标辅助（不依赖 Isaac 类型，仅 numpy）。

来源：参考历史 `cjx` 分支中 Realman 使用 cuRobo 的做法：
- Realman 机器人在场景中的 root link（如 `base_link_underpan`）与 cuRobo 运动学 `base_link`
  （如平台/臂安装基座 link）不一定重合，且可能存在 `platform_joint` 升降导致的 z 偏移。
- cuRobo 规划目标与障碍需要表达在 **运动学 base_link 固连系** 下；因此需要提供 World→base_link 的变换。

与 URDF 对齐示例：`Benchmark/robot/realman/realman_franka_ee.urdf` 中 `platform_joint` 父连杆为
`dual_rm_75b_description`（与底盘/underpan 几何一致）、子连杆为
`dual_rm_75b_description_platform_base_link`，`origin xyz="0. -0.11663 0.271"`、`axis` 沿 **z**；
`arm_base_offset_in_root_xyz` 与 `platform_joint_value` 加到 z 的用法与此一致（`rpy=0` 时）。

注意：
- 本模块只解决 “World → base_link” 的刚体变换（位置/姿态）。
- `CuroboPlanner` 自身还可能额外做轴约定对齐（`apply_robot_to_curobo_frame_transform` 的绕 z -90°）。
  因此推荐的流程是：先用本模块把 World 数据变到 base_link 系，再交给 `CuroboPlanner`（保持同一套约定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _as_vec3(x: np.ndarray | Tuple[float, float, float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"期望 vec3，得到 shape={a.shape}")
    return a


def _as_quat_wxyz(x: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != 4:
        raise ValueError(f"期望 quat(wxyz)，得到 shape={a.shape}")
    return a


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = _as_quat_wxyz(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 乘法，wxyz。"""
    w1, x1, y1, z1 = _as_quat_wxyz(q1)
    w2, x2, y2, z2 = _as_quat_wxyz(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_rotate_vec3_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数旋转向量，wxyz。"""
    q = _as_quat_wxyz(q)
    v = _as_vec3(v)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul_wxyz(quat_mul_wxyz(q, qv), quat_conjugate_wxyz(q))[1:]


@dataclass(frozen=True)
class RealmanBaseLinkKinematics:
    """Realman 场景 root → 运动学 base_link 的偏移参数（近似合同）。

    - `arm_base_offset_in_root_xyz`：root 系下，从 root 原点到臂基（cuRobo base_link）的平移。
      该值应来自 URDF 中 platform_joint（或等价关节）的 origin。
    - `platform_joint_name`：升降平台关节名（若有），其值将加到 offset 的 z 分量上。
    """

    platform_joint_name: Optional[str] = "platform_joint"
    arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271)


def compute_realman_base_link_pose_w(
    *,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    platform_joint_value: float = 0.0,
    arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
    assume_base_quat_equals_root: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """由 root 位姿 + 偏移（含平台 z）估算运动学 base_link 在 World 下的位姿。

    Returns:
        base_pos_w, base_quat_wxyz
    """
    root_pos_w = _as_vec3(root_pos_w)
    root_quat = _as_quat_wxyz(root_quat_wxyz)

    off = np.array(
        [arm_base_offset_in_root_xyz[0], arm_base_offset_in_root_xyz[1], arm_base_offset_in_root_xyz[2] + float(platform_joint_value)],
        dtype=np.float64,
    )
    base_pos_w = root_pos_w + quat_rotate_vec3_wxyz(root_quat, off)
    base_quat = root_quat if assume_base_quat_equals_root else root_quat
    return base_pos_w.astype(np.float64), base_quat.astype(np.float64)


def world_pos_to_base_pos(
    *,
    pos_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """World 位置点 → base_link 位置（表达在 base_link 固连系）。"""
    pos_w = _as_vec3(pos_w)
    base_pos_w = _as_vec3(base_pos_w)
    base_quat = _as_quat_wxyz(base_quat_wxyz)
    dp_w = pos_w - base_pos_w
    # v_b = R_bw * v_w = q_wb^{-1} ⊗ v_w ⊗ q_wb
    return quat_rotate_vec3_wxyz(quat_conjugate_wxyz(base_quat), dp_w).astype(np.float64)


def world_pose_to_base_pose(
    *,
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """World 位姿 → base_link 位姿（表达在 base_link 固连系）。"""
    p_b = world_pos_to_base_pos(pos_w=pos_w, base_pos_w=base_pos_w, base_quat_wxyz=base_quat_wxyz)
    q_w = _as_quat_wxyz(quat_wxyz)
    q_base = _as_quat_wxyz(base_quat_wxyz)
    # q_b = q_base^{-1} ⊗ q_w
    q_b = quat_mul_wxyz(quat_conjugate_wxyz(q_base), q_w)
    return p_b.astype(np.float64), q_b.astype(np.float64)

