"""
OpenVLA 策略（本地 stub 版）：

- 只做两件事：
  1) 从 ObservationBuilder 生成的 ObservationDict 中按 OpenVLA 需求挑选数据并组装 payload
     （单张 RGB + prompt + ee_state 等，占位）
  2) 暂时不做远程推理/不做真实动作映射，返回零 joint 动作用于先跑通闭环

你后续确定远程协议后，只需要把 `predict()` 里 payload 发到服务器并把返回的 delta_action
通过后处理映射为 env 可执行的动作即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from isaaclab_logistics_vla.evaluation.models.policy.base import Policy


def _pick_single_rgb(
    obs: Dict[str, Any],
    prefer_camera: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    从 ObservationDict['vision']['rgb'] 里选一张 RGB。
    约定 rgb shape: (C, num_envs, H, W, 3)
    返回：rgb_single shape (num_envs, H, W, 3) + debug 信息
    """
    vision = obs.get("vision") or {}
    rgb = vision.get("rgb")
    cams = vision.get("cameras") or []

    if not isinstance(rgb, torch.Tensor):
        raise ValueError("ObservationDict.vision.rgb is missing; please enable cameras and require_rgb.")

    cam_idx = 0
    if prefer_camera and cams and prefer_camera in cams:
        cam_idx = cams.index(prefer_camera)

    rgb_single = rgb[cam_idx]  # (num_envs, H, W, 3)
    debug = {"camera_names": list(cams), "picked_camera": cams[cam_idx] if cams else None, "picked_index": cam_idx}
    return rgb_single, debug


@dataclass
class OpenVLAStubPolicy(Policy):
    """
    OpenVLA stub 策略：
    - instruction 目前写死（你后续从 env/task 获取到指令后再替换）
    - 输出先返回零 joint 动作（action_space='joint'）保证环境能跑
    """

    action_dim: int
    device: torch.device
    instruction: str = "pick up the object and place it to the target"
    prefer_camera: Optional[str] = "head_camera"

    _printed_once: bool = False

    @property
    def name(self) -> str:
        return "openvla_stub"

    @property
    def control_mode(self) -> str:
        # OpenVLA 模型级别输出通常是 EE delta，这里先标注为 ee（仅语义），实际先返回 joint 零动作
        return "ee"

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        self._printed_once = False

    def predict(self, obs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # 1) 组装模型输入（payload）
        rgb_single, rgb_debug = _pick_single_rgb(obs, prefer_camera=self.prefer_camera)

        # OpenVLA prompt 模板（按你给的说明）
        prompt = f"In: What action should the robot take to {self.instruction.lower()}?\\nOut: "

        # ee_state：你后续真正做动作后处理时会用到。现在环境侧接口拿不到，就先占位。
        # 这里给一个零向量 placeholder（num_envs, 6），分别代表 pos/euler。
        num_envs = int(obs.get("meta", {}).get("num_envs", rgb_single.shape[0]))
        ee_state = torch.zeros((num_envs, 6), device=self.device, dtype=torch.float32)

        payload = {
            "rgb": rgb_single,  # (num_envs, H, W, 3) on device
            "prompt": prompt,
            "ee_state": ee_state,
            "debug": rgb_debug,
        }

        if not self._printed_once:
            # 只打印一次，避免刷屏：确认数据通路/格式
            rgb_shape = tuple(payload["rgb"].shape)
            print("=== [OpenVLAStubPolicy] Built payload (no remote call) ===")
            print(f"- prompt: {payload['prompt']!r}")
            print(f"- rgb shape: {rgb_shape}, dtype={payload['rgb'].dtype}, device={payload['rgb'].device}")
            print(f"- camera picked: {payload['debug']}")
            self._printed_once = True

        # 2) 暂时不做远程推理：返回零 joint 动作，保证 env.step 能继续跑
        return torch.zeros((num_envs, self.action_dim), device=self.device, dtype=torch.float32)


__all__ = ["OpenVLAStubPolicy"]

