"""
OpenPI 远程策略（复用 /home/junzhe/openpi 的 websocket+msgpack 推理服务）。

设计目标：
- bench 侧遵循本项目的 Policy 接口：predict(ObservationDict) -> action_tensor
- 网络与序列化复用 openpi-client 的 WebsocketClientPolicy（最小依赖、已实现 msgpack-numpy）
- 默认按 openpi 的 DROID / π0.5-DROID 配置组装请求，适配 `serve_policy.py --env DROID`：
    - "observation/exterior_image_1_left": uint8 RGB (H,W,3) —— 桌面大视角
    - "observation/wrist_image_left":      uint8 RGB (H,W,3) —— 腕部/近景视角
    - "observation/joint_position":        关节角向量（这里默认从 qpos 中切分）
    - "observation/gripper_position":      夹爪开合标量（从 qpos 中切分或回退为 0）
    - "prompt":                            指令文本（优先取 obs["instruction"]["text"]，否则使用默认/写死）

注意：
- 该策略假设远端服务器返回 {"actions": (horizon, action_dim)}，并取第 0 步作为当前动作。
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
        # fallback：至少返回一个空向量，避免服务端崩溃；后续可改成硬错误
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
    prefer_camera: Optional[str] = "top_camera"
    image_size: int = 224
    default_instruction: str = "pick up the object and place it to the target"

    _client: Any = None
    _printed_once: bool = False
    _debug_calls: int = 0

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
            # 从多相机里选两路图像：一张作为“大视角”（base），一张作为“腕部”（wrist）。
            # 默认：prefer_camera 对应 base，相机列表中的下一路作为 wrist；如只存在一路，则 wrist 复用 base。
            vision = obs.get("vision") or {}
            cams = vision.get("cameras") or []

            # base 相机索引
            base_cam_idx = 0
            if self.prefer_camera and cams and self.prefer_camera in cams:
                base_cam_idx = cams.index(self.prefer_camera)
            base_cam_name = cams[base_cam_idx] if cams else None

            # wrist 相机索引：优先选 base 之后的下一路，否则回环或复用 base
            if len(cams) >= 2:
                wrist_cam_idx = (base_cam_idx + 1) % len(cams)
            else:
                wrist_cam_idx = base_cam_idx
            wrist_cam_name = cams[wrist_cam_idx] if cams else None

            rgb_base = _pick_rgb_np(
                obs,
                env_id=env_id,
                prefer_camera=base_cam_name,
                resize_hw=(self.image_size, self.image_size),
            )
            rgb_wrist = _pick_rgb_np(
                obs,
                env_id=env_id,
                prefer_camera=wrist_cam_name,
                resize_hw=(self.image_size, self.image_size),
            )

            state_np = _build_state_np(obs, env_id=env_id)

            # 简单约定：如果 state_np 维度足够，把前 7 维作为关节角，后 1 维作为夹爪；
            # 否则回退为零向量（后续可根据具体机器人手动改映射）。
            if state_np.shape[0] >= 8:
                joint_pos = state_np[:7]
                gripper_pos = state_np[7:8]
            else:
                joint_pos = np.zeros((7,), dtype=np.float32)
                gripper_pos = np.zeros((1,), dtype=np.float32)

            # 按 DROIDInputs 期望的键构造请求，适配 pi0/pi0.5 DROID 配置。
            request = {
                "observation/exterior_image_1_left": rgb_base,
                "observation/wrist_image_left": rgb_wrist,
                "observation/joint_position": joint_pos,
                "observation/gripper_position": gripper_pos,
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

            # openpi π0.5‑DROID 的输出维度为 8：7 个关节 + 1 个夹爪。
            # 这里将其映射到 realman_dual_left_arm 的 17 维动作空间：
            #   [0:7]   左臂 7 关节
            #   [7:14]  右臂 7 关节（此处置零，不动）
            #   [14]    左手夹爪（二值开/闭指令）
            #   [15]    右手夹爪（置零，不动）
            #   [16]    平台关节（置零，不动）
            if act0.shape[-1] < 8:
                raise RuntimeError(f"Expected at least 8-dim actions from openpi server, got {act0.shape[-1]}")

            env_action = np.zeros((self.action_dim,), dtype=np.float32)

            # 左臂 7 关节：直接拷贝 openpi 的前 7 维
            env_action[:7] = act0[:7].astype(np.float32)

            # 右臂 7 关节：保持 0（表示“不动”），如需双臂控制可在此扩展映射。
            # env_action[7:14] 默认已为 0

            # 左手夹爪：openpi 的第 8 维为连续 gripper position，这里用符号映射到二值开/闭指令。
            g_raw = float(act0[7])
            # g_cmd > 0 → 打开，g_cmd <= 0 → 闭合；幅值用 1.0 即可，由 BinaryJointPositionActionCfg 解释。
            env_action[14] = 1.0 if g_raw > 0.0 else -1.0

            # 右手夹爪 / 平台：保持 0（不动）。
            # env_action[15] = 0.0
            # env_action[16] = 0.0

            # 调试打印：观测与动作（只打印前若干次、env0，避免刷屏）
            if env_id == 0 and self._debug_calls < 10:
                print("[OpenPIRemotePolicy][debug]")
                print(f"  cameras: {cams}")
                print(f"  base_cam: {base_cam_name}, wrist_cam: {wrist_cam_name}")
                print(f"  joint_pos (sent to openpi): {joint_pos}")
                print(f"  gripper_pos (sent to openpi): {gripper_pos}")
                print(f"  openpi raw act0 (8-dim): {act0}")
                print(f"  mapped env_action (first 8 dims): {env_action[:8]}")
                self._debug_calls += 1

            actions_out[env_id] = torch.from_numpy(env_action).to(self.device)

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

