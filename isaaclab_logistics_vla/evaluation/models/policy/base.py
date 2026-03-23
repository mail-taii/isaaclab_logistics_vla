"""
Base policy interface for the IsaacLab Logistics VLA benchmark.
All local or remote policies must inherit from this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import torch


class Policy(ABC):
    """统一策略接口：本地或远程模型都要实现"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称，用于日志和结果标识"""
        raise NotImplementedError

    @property
    def control_mode(self) -> str:
        """
        控制模式：'joint'（默认，IsaacLab 用关节空间）
        如果模型输出末端位姿，请在内部做 IK
        """
        return "joint"

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """
        每个 episode 开始前重置内部状态
        Args:
            env_ids: 需要重置的环境 ID（并行环境用）
        """
        pass

    @abstractmethod
    def predict(self, obs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        核心推理入口
        Args:
            obs: ObservationDict（由 ObservationBuilder 生成）
            kwargs: 可选扩展，如 timestep/max_episode_length
        Returns:
            action: shape=(num_envs, action_dim)，直接给 env.step()
        """
        raise NotImplementedError


class RemotePolicyBase(Policy, ABC):
    """远程策略的统一基类：负责网络、重试、序列化"""

    def __init__(
        self,
        server_url: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connect()

    @abstractmethod
    def _connect(self) -> None:
        """建立网络连接（TCP/WebSocket/HTTP/ZMQ）"""
        raise NotImplementedError

    @abstractmethod
    def _disconnect(self) -> None:
        """断开连接"""
        raise NotImplementedError

    @abstractmethod
    def _send_request(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """发送请求并接收动作"""
        raise NotImplementedError

    # 公共辅助：序列化观测
    def _serialize_observation(self, obs: Dict[str, torch.Tensor]) -> Dict:
        """Tensor → numpy list，可 JSON/msgpack 序列化"""
        serialized = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = value.cpu().numpy().tolist()
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                serialized[key] = [v.cpu().numpy().tolist() for v in value]
            else:
                serialized[key] = value
        return serialized

    # 公共辅助：反序列化动作
    def _deserialize_action(self, action_data) -> torch.Tensor:
        """numpy list → Tensor"""
        if isinstance(action_data, list):
            return torch.tensor(action_data, dtype=torch.float32)
        if isinstance(action_data, dict) and "action" in action_data:
            return torch.tensor(action_data["action"], dtype=torch.float32)
        raise ValueError(f"Unknown action format: {type(action_data)}")

    def predict(self, obs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """带重试的发送/接收"""
        for attempt in range(self.max_retries):
            try:
                return self._send_request(obs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    import time
                    print(f"[{self.name}] 请求失败，{self.retry_delay}s 后重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    try:
                        self._disconnect()
                        self._connect()
                    except Exception:
                        pass
                else:
                    raise RuntimeError(f"远程策略请求失败（已重试 {self.max_retries} 次）: {e}")
        raise RuntimeError("无法连接到远程策略服务器")

    def __del__(self):
        """析构时断链"""
        try:
            self._disconnect()
        except Exception:
            pass