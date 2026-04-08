#!/usr/bin/env python3
# Copyright (c) 2025, Logistics VLA extension contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""从 URDF 生成供 ``CuroboPlanner`` / ``RobotConfig.from_dict`` 使用的 kinematics YAML。

本脚本**不属于** ``utils.curobo_planner`` 算法封装；在部署前或更新 URDF 后由使用方自行运行。
勿使用 cuRobo 的 ``RobotConfig.write_config`` 写入与本格式混用的缓存（会混入不可 YAML 序列化对象）。
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import yaml

from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig


def _generator_config_to_cache_dict(gen_cfg: CudaRobotGeneratorConfig) -> Dict[str, Any]:
    d = asdict(gen_cfg)
    d.pop("tensor_args", None)
    return {k: v for k, v in d.items() if v is not None}


def generate_and_write(
    urdf_path: str,
    output_path: str,
    *,
    base_link: str = "dual_rm_75b_description_platform_base_link",
    ee_link: Optional[str] = None,
    left_ee_link: str = "panda_left_hand",
    right_ee_link: str = "panda_right_hand",
) -> RobotConfig:
    """构建 ``RobotConfig`` 并将可 ``from_dict`` 的 kinematics 写入 ``output_path``。"""
    tensor_args = TensorDeviceType()
    if ee_link is not None:
        link_names = [ee_link]
        main_ee = ee_link
    else:
        link_names = list(dict.fromkeys([left_ee_link, right_ee_link]))
        main_ee = left_ee_link

    gen_cfg = CudaRobotGeneratorConfig(
        base_link=base_link,
        ee_link=main_ee,
        tensor_args=tensor_args,
        urdf_path=urdf_path,
        link_names=link_names,
    )
    kin = CudaRobotModelConfig.from_config(gen_cfg)
    robot_config = RobotConfig(kinematics=kin, tensor_args=tensor_args)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"kinematics": _generator_config_to_cache_dict(gen_cfg)}
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return robot_config


def main() -> None:
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _default_out = os.path.join(
        _repo,
        "isaaclab_logistics_vla",
        "assets",
        "curobo",
        "realman_kinematics.yaml",
    )
    p = argparse.ArgumentParser(description="URDF → cuRobo kinematics YAML（供 CuroboPlanner 加载）")
    p.add_argument("--urdf", required=True, help="输入 URDF 路径")
    p.add_argument(
        "--output",
        default=_default_out,
        help=f"输出 YAML 路径（默认：包内固定位置 {_default_out}）",
    )
    p.add_argument("--base-link", default="dual_rm_75b_description_platform_base_link")
    p.add_argument("--left-ee-link", default="panda_left_hand")
    p.add_argument("--right-ee-link", default="panda_right_hand")
    p.add_argument(
        "--ee-link",
        default=None,
        help="若指定则生成单臂模型（仅该末端）；不指定则为双臂（left + right）",
    )
    args = p.parse_args()

    rc = generate_and_write(
        args.urdf,
        args.output,
        base_link=args.base_link,
        ee_link=args.ee_link,
        left_ee_link=args.left_ee_link,
        right_ee_link=args.right_ee_link,
    )
    print(f"已写入: {args.output}")
    # 兼容不同 cuRobo 版本：CudaRobotModelConfig 可能没有 joint_limits 字段
    dim = "?"
    try:
        if hasattr(rc.kinematics, "get_joint_limits"):
            jl = rc.kinematics.get_joint_limits()
            n_j = getattr(jl, "position", jl)
            dim = n_j.shape[-1] if hasattr(n_j, "shape") else "?"
        elif hasattr(rc.kinematics, "joint_limits"):
            jl = rc.kinematics.joint_limits
            n_j = getattr(jl, "position", jl)
            dim = n_j.shape[-1] if hasattr(n_j, "shape") else "?"
    except Exception:
        dim = "?"
    print(f"关节维数（参考）: {dim}")


if __name__ == "__main__":
    main()
