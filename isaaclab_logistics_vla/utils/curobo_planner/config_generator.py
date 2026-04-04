"""
从 URDF 生成 cuRobo RobotConfig（双臂：主 ee_link + link_names 包含左右末端）。
缓存 YAML 仅保存可重建的 ``CudaRobotGeneratorConfig`` 字段（与 ``RobotConfig.from_dict`` 一致）；
勿使用 ``RobotConfig.write_config``，其对 ``CudaRobotModelConfig`` 做 ``vars()`` 会混入不可 YAML 序列化的对象。
"""
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import yaml

from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig


def _generator_config_to_cache_dict(gen_cfg: CudaRobotGeneratorConfig) -> Dict[str, Any]:
    """序列化生成器配置，供 ``RobotConfig.from_dict`` 从缓存恢复。"""
    d = asdict(gen_cfg)
    d.pop("tensor_args", None)
    # 去掉 None，减少体积；加载时用调用方 ``tensor_args`` 与默认值补全
    return {k: v for k, v in d.items() if v is not None}


def _write_robot_config_yaml(gen_cfg: CudaRobotGeneratorConfig, output_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"kinematics": _generator_config_to_cache_dict(gen_cfg)}
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def generate_robot_config_from_urdf(
    urdf_path: str,
    output_path: Optional[str] = None,
    base_link: str = "dual_rm_75b_description_platform_base_link",
    left_ee_link: str = "panda_left_hand",
    right_ee_link: str = "panda_right_hand",
    tensor_args: Optional[TensorDeviceType] = None,
) -> RobotConfig:
    """
    从 URDF 生成双臂 RobotConfig（与 MotionGen 的 batch 双末端 Pose 一致）。

    使用 ``CudaRobotGeneratorConfig``：``ee_link`` 为主链末端，``link_names`` 同时包含左右末端。
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()

    link_names = list(dict.fromkeys([left_ee_link, right_ee_link]))

    gen_cfg = CudaRobotGeneratorConfig(
        base_link=base_link,
        ee_link=left_ee_link,
        tensor_args=tensor_args,
        urdf_path=urdf_path,
        link_names=link_names,
    )
    kin = CudaRobotModelConfig.from_config(gen_cfg)
    robot_config = RobotConfig(kinematics=kin, tensor_args=tensor_args)

    if output_path is not None:
        _write_robot_config_yaml(gen_cfg, output_path)

    return robot_config


def load_realman_config(
    urdf_path: str = "/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
    cache_path: Optional[str] = None,
    device: str = "cuda:0",
) -> RobotConfig:
    """
    加载Realman双臂机器人配置
    
    参数:
        urdf_path: URDF文件路径
        cache_path: 缓存配置路径，如果存在则直接加载，否则生成后保存
        device: 计算设备
    
    返回:
        RobotConfig对象
    """
    tensor_args = TensorDeviceType(device=device)
    
    # 如果缓存存在，直接加载
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        robot_config = RobotConfig.from_dict(config_dict, tensor_args)
        return robot_config
    
    # 否则从URDF生成
    robot_config = generate_robot_config_from_urdf(
        urdf_path=urdf_path,
        output_path=cache_path,
        tensor_args=tensor_args,
    )
    
    return robot_config


if __name__ == "__main__":
    # 测试生成Realman配置
    cache_dir = os.path.expanduser("~/.cache/curobo_realman")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "realman_config_v2.yaml")
    
    print(f"Generating Realman robot config from URDF...")
    robot_config = load_realman_config(
        urdf_path="/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
        cache_path=cache_path
    )
    
    print(f"Config generated successfully!")
    print(f"Saved to: {cache_path}")
    jl = robot_config.kinematics.joint_limits
    n_j = getattr(jl, "position", jl)
    n_j = n_j.shape[-1] if hasattr(n_j, "shape") else "?"
    print(f"Joint limits dim: {n_j}")
