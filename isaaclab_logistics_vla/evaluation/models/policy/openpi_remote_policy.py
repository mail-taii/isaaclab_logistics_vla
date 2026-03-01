"""
OpenPI 远程策略（复用 /home/junzhe/openpi 的 websocket+msgpack 推理服务）。

设计目标：
- bench 侧遵循本项目的 Policy 接口：predict(ObservationDict) -> action_tensor
- 网络与序列化复用 openpi-client 的 WebsocketClientPolicy（最小依赖、已实现 msgpack-numpy）
- 输入侧把 ObservationDict 按 openpi policy server 的约定组装成 dict：
    - "observation/image": uint8 RGB (H,W,3)（客户端侧 resize_with_pad 到 224）
    - "observation/state": proprio state（这里先用 qpos/qvel 拼接，具体可按你的模型训练口径调整）
    - "prompt": 指令文本（优先取 obs["instruction"]["text"]，否则使用默认/写死）

注意：
- 该策略假设远端服务器返回 {"actions": (horizon, action_dim)}，并取第 0 步作为当前动作。
- 如果你的 openpi 模型输出的是 EE delta / chunk 等，需要在这里做后处理与动作映射。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from isaaclab_logistics_vla.evaluation.models.policy.base import Policy


def _require_openpi_client():
    try:
        from openpi_client import image_tools  # noqa: F401
        from openpi_client import websocket_client_policy  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未找到 openpi-client。请先在你的运行环境安装：\n"
            "  cd /home/junzhe/openpi/packages/openpi-client && pip install -e .\n"
            f"原始错误: {e}"
        )


def _pick_rgb_np(
    obs: Dict[str, Any],
    env_id: int,
    prefer_camera: Optional[str],
    resize_hw: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """从 ObservationDict 中取单 env 的单张 RGB，并 resize/uint8 化为 numpy。"""
    from openpi_client import image_tools

    vision = obs.get("vision") or {}
    rgb = vision.get("rgb")
    cams = vision.get("cameras") or []
    if not isinstance(rgb, torch.Tensor):
        raise ValueError("ObservationDict.vision.rgb 缺失；请确保 enable_cameras=True 且 require_rgb=True")

    cam_idx = 0
    if prefer_camera and cams and prefer_camera in cams:
        cam_idx = cams.index(prefer_camera)

    # rgb: (C, num_envs, H, W, 3) -> (H, W, 3)
    rgb_t = rgb[cam_idx, env_id]
    rgb_np = rgb_t.detach().to("cpu").numpy()
    rgb_np = image_tools.convert_to_uint8(rgb_np)
    rgb_np = image_tools.resize_with_pad(rgb_np, resize_hw[0], resize_hw[1])
    return rgb_np


def _build_state_np(obs: Dict[str, Any], env_id: int) -> np.ndarray:
    """构造 openpi 的 proprio state（这里先简单拼 qpos/qvel）。"""
    rs = obs.get("robot_state") or {}
    qpos = rs.get("qpos")
    qvel = rs.get("qvel")
    if not (isinstance(qpos, torch.Tensor) and isinstance(qvel, torch.Tensor)):
        # fallback：至少返回一个空向量，避免服务端崩溃；你后续可改成硬错误
        return np.zeros((0,), dtype=np.float32)
    qpos_np = qpos[env_id].detach().to("cpu").numpy().astype(np.float32)
    qvel_np = qvel[env_id].detach().to("cpu").numpy().astype(np.float32)
    return np.concatenate([qpos_np, qvel_np], axis=0)


def _pick_prompt(obs: Dict[str, Any], default_instruction: str) -> str:
    inst = obs.get("instruction") or {}
    text = inst.get("text")
    if isinstance(text, str) and text.strip():
        return text
    # 多 env 时也可能是 list[str]
    if isinstance(text, list) and text and isinstance(text[0], str):
        return text[0]
    return default_instruction


@dataclass
class OpenPIRemotePolicy(Policy):
    """通过 openpi websocket policy server 做远程推理。"""

    action_dim: int
    device: torch.device

    # 连接参数：默认优先读环境变量，便于 bench 侧不改 CLI 也能配置
    host: str = ""
    port: int = 0
    api_key: Optional[str] = None

    # 输入选择
    prefer_camera: Optional[str] = "head_camera"
    image_size: int = 224
    default_instruction: str = "pick up the object and place it to the target"

    _client: Any = None
    _printed_once: bool = False

    @property
    def name(self) -> str:
        return "openpi_remote"

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        self._printed_once = False

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        _require_openpi_client()
        from openpi_client import websocket_client_policy

        host = self.host or os.environ.get("OPENPI_HOST", "localhost")
        port = self.port or int(os.environ.get("OPENPI_PORT", "8000"))
        api_key = self.api_key or os.environ.get("OPENPI_API_KEY")
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=api_key)

    def predict(self, obs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self._ensure_client()

        num_envs = int((obs.get("meta") or {}).get("num_envs", 1))
        prompt = _pick_prompt(obs, self.default_instruction)

        actions_out = torch.zeros((num_envs, self.action_dim), device=self.device, dtype=torch.float32)

        for env_id in range(num_envs):
            rgb_np = _pick_rgb_np(
                obs,
                env_id=env_id,
                prefer_camera=self.prefer_camera,
                resize_hw=(self.image_size, self.image_size),
            )
            state_np = _build_state_np(obs, env_id=env_id)

            request = {
                "observation/image": rgb_np,
                "observation/state": state_np,
                "prompt": prompt,
            }
            response = self._client.infer(request)
            if "actions" not in response:
                raise RuntimeError(f"openpi server response missing 'actions': keys={list(response.keys())}")

            act = np.asarray(response["actions"])
            # 兼容 chunk: (H, D) 或单步: (D,)
            if act.ndim == 2:
                act0 = act[0]
            elif act.ndim == 1:
                act0 = act
            else:
                raise RuntimeError(f"Unexpected actions shape from server: {act.shape}")

            if act0.shape[-1] != self.action_dim:
                raise RuntimeError(
                    f"action_dim mismatch: server={act0.shape[-1]} vs env={self.action_dim}. "
                    "你需要在这里做动作映射/投影，或确保服务器使用匹配的动作空间训练。"
                )

            actions_out[env_id] = torch.from_numpy(act0.astype(np.float32)).to(self.device)

        if not self._printed_once:
            print("=== [OpenPIRemotePolicy] Connected and inferred once ===")
            try:
                meta = (obs.get("meta") or {})
                print(f"- meta: {dict(meta)}")
            except Exception:
                pass
            print(f"- prompt: {prompt!r}")
            self._printed_once = True

        return actions_out


__all__ = ["OpenPIRemotePolicy"]

