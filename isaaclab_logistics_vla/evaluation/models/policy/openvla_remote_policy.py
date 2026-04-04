"""
OpenVLA 远程策略：调用 /home/junzhe/openvla 里的 REST 部署脚本 `vla-scripts/deploy.py` 提供的接口。

Server 侧（OpenVLA 仓库）：
- 运行 `python vla-scripts/deploy.py`，会启动一个 FastAPI + Uvicorn 服务：
  - 路由: POST /act
  - 请求: {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
  - 响应: {"action": np.ndarray}

Client 侧（本 bench 仓库）：
- 通过 HTTP POST 调用该接口，使用 json-numpy 对 numpy array 编码。
- 本策略类负责：
  1) 从 ObservationDict 中取一张 RGB 图像和指令文本
  2) 将它们打包成 OpenVLA Server 期望的 payload
  3) 解析返回的 action，并转换成 IsaacLab 环境所需的动作 tensor

注意：
- 当前默认只用第 0 个 env 的观测发请求；得到的动作会广播到所有 env。
- 如果 OpenVLA 输出的 action 维度和 env.action_dim 不一致，会抛出错误，提示需要在此处做映射。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import json_numpy
import numpy as np
import requests
import torch

from isaaclab_logistics_vla.evaluation.models.policy.base import Policy


def _pick_single_rgb_np(
    obs: Dict[str, Any],
    env_id: int = 0,
    prefer_camera: Optional[str] = None,
) -> np.ndarray:
    """
    从 ObservationDict['vision']['rgb'] 中选出一个 env 的一张 RGB 图像，返回 numpy.uint8 的 (H, W, 3)。
    约定 rgb shape: (C, num_envs, H, W, 3)；依赖 ObservationBuilder 提供的标准 vision.rgb。
    """
    vision = obs.get("vision") or {}
    rgb = vision.get("rgb")
    cams = vision.get("cameras") or []

    if not isinstance(rgb, torch.Tensor):
        raise ValueError(
            "ObservationDict.vision.rgb 缺失或非 Tensor；"
            "请确保 enable_cameras=True 且 ObservationBuilder 提供 require_rgb=True。"
        )

    cam_idx = 0
    if prefer_camera and cams and prefer_camera in cams:
        cam_idx = cams.index(prefer_camera)

    rgb_t = rgb[cam_idx, env_id]  # (H, W, 3)
    rgb_np = rgb_t.detach().to("cpu").numpy()

    if not np.issubdtype(rgb_np.dtype, np.uint8):
        rgb_np = (255.0 * np.clip(rgb_np, 0.0, 1.0)).astype(np.uint8)

    return rgb_np


def _action_dim_hint(dim: int) -> str:
    """按动作维度给出常见解读（不硬编码单一维度）。"""
    hints = {
        7: "常见为单臂 EE/关节 7 维（如 3 位移+姿态/夹爪）",
        14: "常见为双臂各 7 维",
        16: "常见为双臂 7+7 + 双夹爪 1+1",
        17: "常见为双臂 7+7 + 夹爪/基座等",
        21: "常见为双臂 7+7 + 夹爪等 7 维",
        31: "常见为双臂 7+7 + 双夹爪 1+1 + 基座/其他 15 维",
    }
    if dim in hints:
        return hints[dim]
    if dim <= 7:
        return f"可能为单臂/EE 等 {dim} 维"
    if dim <= 14:
        return f"可能为双臂或单臂+夹爪等 {dim} 维"
    return f"多自由度配置，共 {dim} 维（前 3 维通常为 EE 位移）"


def _pick_instruction(obs: Dict[str, Any], default_instruction: str) -> str:
    inst = obs.get("instruction") or {}
    text = inst.get("text")
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(text, list) and text and isinstance(text[0], str):
        return text[0]
    return default_instruction


@dataclass
class OpenVLARemotePolicy(Policy):
    """
    利用 openvla 仓库中的 deploy.py 暴露的 REST 接口进行远程推理。

    - action_dim: IsaacLab 动作维度（用来做 shape 校验）
    - device: 动作输出所在 device
    - host/port: OpenVLA Server 地址，默认从环境变量 OPENVLA_HOST/OPENVLA_PORT 读取，否则用 localhost:8000
    - unnorm_key: 可选，对应 OpenVLA server 中的 dataset_statistics key
    """

    action_dim: int
    device: torch.device

    host: str = "10.60.45.81"
    port: int = 8001
    unnorm_key: Optional[str] = "bridge_orig"  # 使用 bridge_orig，这是最常用的 unnorm_key

    default_instruction: str = "pick up the object from the white box"
    prefer_camera: Optional[str] = "head_camera"
    timeout: float = 10.0

    _session: Any = None
    _printed_once: bool = False

    @property
    def name(self) -> str:
        return "openvla_remote"

    @property
    def control_mode(self) -> str:
        """
        OpenVLA 模型输出末端执行器位移（EE delta），
        需要在内部实现逆运动学转换
        """
        return "ee"

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        self._printed_once = False

    def _ensure_client(self):
        if self._session is not None:
            return
        json_numpy.patch()
        self._session = requests.Session()

        self._host = self.host or os.environ.get("OPENVLA_HOST", "localhost")
        self._port = self.port or int(os.environ.get("OPENVLA_PORT", "8000"))

    def predict(self, obs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self._ensure_client()

        num_envs = int((obs.get("meta") or {}).get("num_envs", 1))
        if num_envs <= 0:
            num_envs = 1

        # 先只用 env0 发送请求；得到的动作广播给所有 env
        env_id = 0
        image_np = _pick_single_rgb_np(obs, env_id=env_id, prefer_camera=self.prefer_camera)
        instruction = _pick_instruction(obs, self.default_instruction)

        payload: Dict[str, Any] = {
            "image": image_np,
            "instruction": instruction,
        }
        if self.unnorm_key is not None:
            payload["unnorm_key"] = self.unnorm_key

        url = f"http://{self._host}:{self._port}/act"
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        
        # 检查响应状态，提供更详细的错误信息
        if resp.status_code != 200:
            print(f"=== OpenVLA 服务器错误 ===")
            print(f"状态码: {resp.status_code}")
            print(f"响应头: {dict(resp.headers)}")
            try:
                error_data = resp.json()
                print(f"错误响应: {error_data}")
            except:
                print(f"错误内容: {resp.text[:500]}")  # 显示前500字符
            resp.raise_for_status()
        
        data = resp.json()

        if isinstance(data, str) and data == "error":
            raise RuntimeError("OpenVLA server 返回 'error'，请检查 server 端日志和请求格式。")

        if isinstance(data, dict) and "action" in data:
            action_arr = np.asarray(data["action"])
        else:
            raise RuntimeError(f"OpenVLA server 响应中未找到 'action' 字段，实际 keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        # 打印动作数据详细信息
        print(f"=== OpenVLA 动作数据 ===")
        print(f"动作数组形状: {action_arr.shape}")
        print(f"动作数据类型: {action_arr.dtype}")
        print(f"动作数据范围: [{action_arr.min():.3f}, {action_arr.max():.3f}]")
        print(f"动作数据均值: {action_arr.mean():.3f}, 标准差: {action_arr.std():.3f}")
        
        # 打印前10个动作值
        if action_arr.ndim == 1:
            print(f"动作值 (前10个): {action_arr[:10]}")
        elif action_arr.ndim == 2:
            print(f"动作值 (第一帧前10个): {action_arr[0, :10]}")
        
        # 兼容 (D,) 或 (H, D)
        if action_arr.ndim == 2:
            act0 = action_arr[0]
        elif action_arr.ndim == 1:
            act0 = action_arr
        else:
            raise RuntimeError(f"从 OpenVLA server 得到的 action 形状异常: {action_arr.shape}")

        out_dim = int(act0.shape[-1])
        print(f"OpenVLA 输出维度: {out_dim}, 环境需要维度: {self.action_dim}")

        # 按输出维度给出常见解释（不硬编码具体数字）
        _desc = _action_dim_hint(out_dim)
        if _desc:
            print(f"维度解读: {_desc}")
        print(f"动作范围: [{act0.min():.4f}, {act0.max():.4f}], 均值: {act0.mean():.4f}")

        if out_dim != self.action_dim:
            # 将 OpenVLA 前若干维填入 env action，前 3 维供 evaluator 做 EE 位移并 IK
            copy_len = min(out_dim, self.action_dim)
            zero_action = np.zeros(self.action_dim, dtype=act0.dtype)
            zero_action[:copy_len] = act0[:copy_len]
            print(f"INFO: 将 OpenVLA 前 {copy_len} 维填入 action（前 3 维供 EE 位移 + IK）")
            act0 = zero_action

        action_tensor = torch.from_numpy(act0.astype(np.float32)).to(self.device)
        actions = action_tensor.repeat(num_envs, 1)

        if not self._printed_once:
            print("=== [OpenVLARemotePolicy] First call succeeded ===")
            try:
                meta = dict(obs.get("meta") or {})
                print(f"- meta: {meta}")
            except Exception:
                pass
            print(f"- url: {url}")
            print(f"- instruction: {instruction!r}")
            print(f"- image shape: {image_np.shape}, dtype={image_np.dtype}")
            print(f"- action shape: {actions.shape}")
            self._printed_once = True

        return actions


__all__ = ["OpenVLARemotePolicy"]