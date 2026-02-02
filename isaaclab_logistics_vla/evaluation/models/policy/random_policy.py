from __future__ import annotations

"""
一个最简单的 RandomPolicy，用来验证：

1. 评估侧可以通过 ObservationBuilder 拿到完整 ObservationDict（meta / robot_state / vision / point_cloud）。
2. 这个 ObservationDict 可以作为入参传给任意策略（这里用随机策略代替），策略内部可以看到同样的结构。

真正做策略的时候，只需要沿用这里的接口约定即可：
    - policy(obs: ObservationDict) -> ActionDict
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch

from isaaclab_logistics_vla.evaluation.observation.schema import ActionDict, ObservationDict


def _summarize_tensor(t: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }


def summarize_observation(obs: ObservationDict) -> Dict[str, Any]:
    """把 ObservationDict 转成易读结构（策略侧看到的视角）。"""
    summary: Dict[str, Any] = {}

    meta = obs.get("meta")
    if meta is not None:
        summary["meta"] = dict(meta)

    rs = obs.get("robot_state")
    if rs is not None:
        rs_sum: Dict[str, Any] = {}
        for k, v in rs.items():
            if isinstance(v, torch.Tensor):
                rs_sum[k] = _summarize_tensor(v)
        summary["robot_state"] = rs_sum

    vision = obs.get("vision")
    if vision is not None:
        v_sum: Dict[str, Any] = {}
        cams = vision.get("cameras")
        if cams is not None:
            v_sum["cameras"] = list(cams)
        for key in ["rgb", "depth", "segmentation", "intrinsic", "extrinsic"]:
            val = vision.get(key)
            if isinstance(val, torch.Tensor):
                v_sum[key] = _summarize_tensor(val)
        summary["vision"] = v_sum

    pcd = obs.get("point_cloud")
    if pcd is not None:
        p_sum: Dict[str, Any] = {}
        mpc = pcd.get("masked_point_cloud")
        if isinstance(mpc, torch.Tensor) or torch.is_tensor(mpc):
            p_sum["masked_point_cloud"] = _summarize_tensor(mpc)
        elif mpc is not None:
            p_sum["masked_point_cloud"] = {"type": type(mpc).__name__}
        frame = pcd.get("frame")
        if frame is not None:
            p_sum["frame"] = frame
        summary["point_cloud"] = p_sum

    return summary


@dataclass
class RandomPolicy:
    """一个最小可用的策略实现，用于验证数据通路。

    - 初始化时只需要知道动作维度和 device。
    - __call__ 接受 ObservationDict，返回 ActionDict。
    """

    action_dim: int
    device: torch.device

    # 只在第一次调用时打印观测结构，避免刷屏
    _printed_summary: bool = False

    def __call__(self, obs: ObservationDict) -> ActionDict:
        """根据 ObservationDict 生成一个随机动作，并在首次调用时打印观测结构。"""
        if not self._printed_summary:
            from pprint import pprint

            print("=== [RandomPolicy] Received ObservationDict (summary) ===")
            pprint(summarize_observation(obs), width=120)
            self._printed_summary = True

        # 这里为了简单，假设 num_envs 可以从 robot_state.qpos 推出来
        num_envs = 1
        if "robot_state" in obs:
            qpos = obs["robot_state"].get("qpos")
            if isinstance(qpos, torch.Tensor):
                num_envs = qpos.shape[0]

        # 生成 [-1, 1] 区间的随机动作
        action = 2.0 * torch.rand((num_envs, self.action_dim), device=self.device) - 1.0

        out: ActionDict = {
            "action": action,
            "action_space": "joint",
        }
        return out


__all__ = ["RandomPolicy", "summarize_observation"]

