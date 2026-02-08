"""
按机器人区分的相机配置（任务通用）：每个机器人一套相机，prim_path 绑定该机器人的 link。
接入新机器人时在本目录新增模块（如 ur10e.py），并在 CAMERA_CONFIG_REGISTRY 中注册。
"""
from __future__ import annotations

from typing import Any

from . import realman
from . import ur5e

# key → 提供 .head_camera / .ee_camera / .top_camera 的类或对象
CAMERA_CONFIG_REGISTRY: dict[str, Any] = {
    "realman": realman.RealmanCameraConfig,
    "ur5e": ur5e.UR5eCameraConfig,
}


def get_camera_config(key: str) -> Any:
    """根据 camera_config_key 返回该机器人的相机配置（含 head_camera、ee_camera、top_camera）。"""
    if key not in CAMERA_CONFIG_REGISTRY:
        raise KeyError(
            f"未注册的相机配置 key: {key!r}. "
            f"已注册: {list(CAMERA_CONFIG_REGISTRY.keys())}. "
            "请在 configs/camera_configs/ 中新增模块并在 CAMERA_CONFIG_REGISTRY 中注册。"
        )
    return CAMERA_CONFIG_REGISTRY[key]


def list_camera_config_keys() -> list[str]:
    """返回已注册的 camera_config_key 列表。"""
    return list(CAMERA_CONFIG_REGISTRY.keys())
