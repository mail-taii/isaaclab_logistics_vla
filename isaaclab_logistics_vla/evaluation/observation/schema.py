from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, MutableMapping, Optional, Sequence, TypedDict, Union

import torch


#
# 顶层 Observation / Action / Metrics 结构定义
# 这些是“评估模块”和“策略/模型”之间约定好的数据格式（schema），不包含任何环境细节。
#


class MetaInfo(TypedDict, total=False):
    """与当前观测相关的元信息（完全可选，但推荐统一字段名）。"""

    task_name: str
    episode_id: int
    step_id: int
    num_envs: int
    timestamp: float
    # 单位、坐标系等信息（例如 depth 单位、pcd frame、extrinsic 方向）
    units: Dict[str, str]


class RobotState(TypedDict):
    """机器人状态观测。"""

    # 关节位置 / 速度 / 加速度，均为 tensor，shape = (num_envs, num_joints)
    qpos: torch.Tensor
    qvel: torch.Tensor
    # qacc 可以全 0 或真实加速度，由环境侧决定；缺失时可以不填（由 builder 补）
    qacc: torch.Tensor


class VisionData(TypedDict, total=False):
    """视觉相关观测（多相机、多模态，可选）。"""

    # 相机名称列表，定义了后续张量的第 0 维顺序
    cameras: List[str]

    # RGB / Depth / Seg 等通常来自 IsaacLab Camera/TiledCamera 传感器
    # 约定 shape:
    #   rgb:    (C, num_envs, H, W, 3)
    #   depth:  (C, num_envs, H, W)
    #   seg:    (C, num_envs, H, W)
    rgb: torch.Tensor
    depth: torch.Tensor
    segmentation: torch.Tensor
    robot_mask: torch.Tensor

    # 内参与外参：
    #   intrinsic: (C, num_envs, 3, 3)
    #   extrinsic: (C, num_envs, 4, 4)
    intrinsic: torch.Tensor
    extrinsic: torch.Tensor


class PointCloudData(TypedDict, total=False):
    """点云相关观测（可选）。"""

    # masked_point_cloud: (num_envs, N, 3)
    masked_point_cloud: torch.Tensor
    # 点云坐标系：'world' 或 'camera'
    frame: Literal["world", "camera"]


class InstructionData(TypedDict, total=False):
    """指令相关观测（可选）。"""

    # 可以是单条指令，也可以是每个 env 一条指令
    text: Union[str, List[str]]


class ObservationDict(TypedDict, total=False):
    """策略/模型看到的一次完整观测。

    - 顶层只做“分组”；真正的 tensor 都放在嵌套结构里。
    - 所有字段都是可选的，由 ObservationBuilder 按需填充。
    """

    meta: MetaInfo
    robot_state: RobotState
    instruction: InstructionData
    vision: VisionData
    point_cloud: PointCloudData


class ActionDict(TypedDict, total=False):
    """策略/模型输出给环境的一次动作（或一组动作）。"""

    # 单步动作：shape = (num_envs, action_dim)
    action: torch.Tensor

    # 动作空间类型：目前推荐统一为 'joint'
    action_space: Literal["joint", "ee"]

    # 可选：一次返回多步动作 chunk，用于远程推理降低往返次数
    #   action_chunk: (K, num_envs, action_dim)
    action_chunk: torch.Tensor
    chunk_horizon: int


#
# Metrics 相关结构（只读）
#

MetricScalar = Union[float, int, bool]
MetricVector = Sequence[MetricScalar]


class MetricsReadout(TypedDict):
    """从环境读取到的一次 metrics 快照（通常在 episode 结束时调用）。"""

    # metrics: 例如
    # {
    #   "object_success_rate": 0.9,
    #   "order_success_rate": [0.8, 0.9, 1.0, 0.7],  # 多 env
    #   "success": True,
    #   ...
    # }
    metrics: Dict[str, Union[MetricScalar, MetricVector]]

    # reduce 策略说明（多 env 时的对齐方式）
    # - 'none'：metrics 中保留每个 env 的值（推荐 episode 级别）
    # - 'mean'：已在读取时做均值（不再保留逐 env）
    # - 'first'：只取 env0 的值（调试用）
    reduce: Literal["none", "mean", "first"]

    # 当前 env 个数（便于下游做检查）
    num_envs: int


#
# Observation 构建选项（给 ObservationBuilder 使用）
#

@dataclass
class ObservationRequire:
    """控制 ObservationBuilder 需要构建哪些模态."""

    require_rgb: bool = True
    require_depth: bool = True
    require_seg: bool = True
    require_pcd: bool = False

    # 点云坐标系：'world' / 'camera'
    pcd_frame: Literal["world", "camera"] = "world"

