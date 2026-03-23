from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch

import isaaclab_logistics_vla
from isaaclab_logistics_vla.evaluation.robot_registry import RobotEvalConfig

try:
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig, JointState
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )
except ImportError:
    # 在未安装 Curobo 的环境下允许导入该模块，但实际调用会失败
    CollisionCheckerType = None  # type: ignore
    WorldConfig = None  # type: ignore
    TensorDeviceType = None  # type: ignore
    Pose = None  # type: ignore
    RobotConfig = None  # type: ignore
    JointState = None  # type: ignore
    load_yaml = None  # type: ignore
    MotionGen = None  # type: ignore
    MotionGenConfig = None  # type: ignore
    MotionGenPlanConfig = None  # type: ignore


WorldMode = Literal["boxes_mesh", "boxes_hollow", "boxes_cuboid", "table_only"]


def _load_robot_cfg(
    robot_eval_cfg: RobotEvalConfig,
    tensor_args: "TensorDeviceType",
) -> Tuple["RobotConfig", Path]:
    """根据 RobotEvalConfig 加载 Curobo RobotConfig，并返回 robot_cfg 与 pkg 根路径。"""
    pkg_dir = Path(isaaclab_logistics_vla.__file__).resolve().parent
    robot_configs_dir = pkg_dir / "configs" / "robot_configs"
    assets_dir = pkg_dir / "assets" / "robots" / robot_eval_cfg.curobo_asset_folder
    robot_yml = robot_configs_dir / robot_eval_cfg.curobo_yml_name

    config_file = load_yaml(str(robot_yml))
    kin = config_file["robot_cfg"].get("kinematics", {})
    print(
        f"[curobo_planner] CuRobo yaml: base_link={kin.get('base_link')!r}, "
        f"ee_link={kin.get('ee_link')!r}"
    )
    config_file["robot_cfg"]["kinematics"]["urdf_path"] = str(
        assets_dir / robot_eval_cfg.curobo_urdf_name
    )
    config_file["robot_cfg"]["kinematics"]["asset_root_path"] = str(assets_dir)
    config_file["robot_cfg"]["kinematics"]["collision_spheres"] = str(
        robot_configs_dir / "spheres" / robot_eval_cfg.curobo_yml_name
    )
    robot_cfg = RobotConfig.from_dict(config_file["robot_cfg"], tensor_args)
    return robot_cfg, pkg_dir


def _build_world_cfg(
    world_mode: WorldMode,
    pkg_dir: Path,
) -> Tuple["WorldConfig", "CollisionCheckerType", Optional[dict], int, int]:
    """根据模式构建 WorldConfig 及碰撞检查配置。"""
    from isaaclab_logistics_vla.utils.curobo_mesh_world import (
        get_hollow_box_world_for_curobo,
        get_mesh_world_for_curobo,
    )

    if world_mode == "boxes_mesh":
        asset_root = os.environ.get("ASSET_ROOT_PATH") or str(pkg_dir.parent / "Benchmark")
        world_cfg = get_mesh_world_for_curobo(asset_root=asset_root)
        n_meshes = len(world_cfg.mesh or [])
        n_cuboids = 0
        checker_type = CollisionCheckerType.MESH
        collision_cache = {"obb": 0, "mesh": n_meshes} if n_meshes > 0 else None
        print(f"[curobo_planner] 🌍 已加载箱子 mesh 障碍物: {n_meshes} 个")
        return world_cfg, checker_type, collision_cache, n_cuboids, n_meshes

    if world_mode == "boxes_hollow":
        world_cfg = get_hollow_box_world_for_curobo()
        n_cuboids = len(world_cfg.cuboid or [])
        n_meshes = 0
        checker_type = CollisionCheckerType.PRIMITIVE
        collision_cache = {"obb": n_cuboids, "mesh": 0} if n_cuboids > 0 else None
        print(f"[curobo_planner] 🌍 已加载空心箱障碍物: {n_cuboids} 个 cuboid (5 板/箱)")
        return world_cfg, checker_type, collision_cache, n_cuboids, n_meshes

    if world_mode == "boxes_cuboid":
        # policy 用的 scene_obstacles.yml（箱子近似为 cuboid）
        world_yaml = pkg_dir / "configs" / "worlds" / "scene_obstacles.yml"
        if not world_yaml.exists():
            world_yaml = Path("scene_obstacles.yml").resolve()
        if not world_yaml.exists():
            raise FileNotFoundError(f"[curobo_planner] 障碍物配置不存在: {world_yaml}")
        data = load_yaml(str(world_yaml))
        if "cuboids" in data:
            cuboid_dict = {c["name"]: {"pose": c["pose"], "dims": c["dims"]} for c in data["cuboids"]}
            world_cfg = WorldConfig.from_dict({"cuboid": cuboid_dict})
        elif "cuboid" in data:
            world_cfg = WorldConfig.from_dict(data)
        else:
            raise ValueError(f"[curobo_planner] 障碍物配置缺少 cuboid/cuboids: {world_yaml}")
        n_cuboids = len(world_cfg.cuboid or [])
        n_meshes = 0
        checker_type = CollisionCheckerType.PRIMITIVE
        collision_cache = {"obb": n_cuboids, "mesh": 0} if n_cuboids > 0 else None
        print(f"[curobo_planner] 🌍 已加载箱子 cuboid 障碍物: {n_cuboids} 个")
        return world_cfg, checker_type, collision_cache, n_cuboids, n_meshes

    if world_mode == "table_only":
        # 仅桌子，用于 EE 模式
        world_yaml = pkg_dir / "configs" / "worlds" / "scene_obstacles_table_only.yml"
        if not world_yaml.exists():
            raise FileNotFoundError(f"[curobo_planner] EE 模式障碍物配置不存在: {world_yaml}")
        data = load_yaml(str(world_yaml))
        if "cuboids" in data:
            cuboid_dict = {c["name"]: {"pose": c["pose"], "dims": c["dims"]} for c in data["cuboids"]}
            world_cfg = WorldConfig.from_dict({"cuboid": cuboid_dict})
        elif "cuboid" in data:
            world_cfg = WorldConfig.from_dict(data)
        else:
            raise ValueError(f"[curobo_planner] 障碍物配置缺少 cuboid/cuboids: {world_yaml}")
        n_cuboids = len(world_cfg.cuboid or [])
        n_meshes = 0
        checker_type = CollisionCheckerType.PRIMITIVE
        collision_cache = {"obb": n_cuboids, "mesh": 0} if n_cuboids > 0 else None
        print(f"[curobo_planner] 🌍 已加载桌子 cuboid 障碍物: {n_cuboids} 个")
        return world_cfg, checker_type, collision_cache, n_cuboids, n_meshes

    raise ValueError(f"[curobo_planner] 不支持的 world_mode: {world_mode}")


def build_motion_gen(
    robot_eval_cfg: RobotEvalConfig,
    curobo_device: torch.device,
    world_mode: WorldMode,
    logger_name: str = "curobo_planner",
) -> "MotionGen":
    """统一构建并 warmup Curobo MotionGen，供 policy / evaluator 复用。"""
    if MotionGen is None:
        raise RuntimeError("Curobo 未安装或导入失败，无法构建 MotionGen。")

    tensor_args = TensorDeviceType(device=curobo_device, dtype=torch.float32)
    robot_cfg, pkg_dir = _load_robot_cfg(robot_eval_cfg, tensor_args)
    world_cfg, checker_type, collision_cache, n_cuboids, n_meshes = _build_world_cfg(
        world_mode, pkg_dir
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args=tensor_args,
        collision_checker_type=checker_type,
        collision_cache=collision_cache,
        interpolation_dt=0.01,
        use_cuda_graph=False,
        collision_activation_distance=0.025,
    )
    motion_gen = MotionGen(motion_gen_config)

    print(f"[{logger_name}] 🔥 正在预热 MotionGen...")
    motion_gen.warmup()
    if world_mode == "boxes_mesh":
        print(f"[{logger_name}] ✅ MotionGen 初始化完成（障碍物: {n_meshes} 个箱子 mesh）")
    elif world_mode == "boxes_hollow":
        print(f"[{logger_name}] ✅ MotionGen 初始化完成（障碍物: 空心箱 {n_cuboids} 个 cuboid）")
    elif world_mode == "boxes_cuboid":
        print(f"[{logger_name}] ✅ MotionGen 初始化完成（障碍物: 箱子 cuboid {n_cuboids} 个）")
    elif world_mode == "table_only":
        print(f"[{logger_name}] ✅ MotionGen 初始化完成（障碍物: 仅桌子 {n_cuboids} 个 cuboid）")

    return motion_gen


def plan_single_ee_motion(
    motion_gen: "MotionGen",
    q_start: torch.Tensor,
    target_pos_b: torch.Tensor,
    target_quat_b: torch.Tensor,
    max_attempts: int = 10,
    timeout: float = 2.0,
    enable_graph: bool = True,
    enable_opt: bool = False,
) -> "MotionGenPlanConfig":
    """统一的单次 EE 规划调用：关节 q_start + 目标臂基系 EE 位姿 → MotionGen result。

    Args:
        motion_gen: 已初始化好的 MotionGen 实例。
        q_start: 当前关节角，shape=(dof,) 或 (1, dof)。
        target_pos_b: 目标 EE 位置（臂基系），shape=(3,) 或 (1, 3)。
        target_quat_b: 目标 EE 姿态（臂基系，wxyz），shape=(4,) 或 (1, 4)。
    """
    if MotionGen is None or JointState is None or Pose is None:
        raise RuntimeError("Curobo 未正确安装，无法进行 MotionGen 规划。")

    # 统一形状到 (1, dof) / (1, 1, 3) / (1, 1, 4)，保持原有 device/dtype
    if q_start.dim() == 1:
        q_start_b = q_start.view(1, -1)
    else:
        q_start_b = q_start[:, :].contiguous()

    if target_pos_b.dim() == 1:
        pos_b = target_pos_b.view(1, 1, 3)
    elif target_pos_b.dim() == 2:
        pos_b = target_pos_b.view(1, 1, 3)
    else:
        pos_b = target_pos_b

    if target_quat_b.dim() == 1:
        quat_b = target_quat_b.view(1, 1, 4)
    elif target_quat_b.dim() == 2:
        quat_b = target_quat_b.view(1, 1, 4)
    else:
        quat_b = target_quat_b

    start_state = JointState.from_position(q_start_b)
    goal_pose = Pose(pos_b, quat_b)
    plan_cfg = MotionGenPlanConfig(
        max_attempts=max_attempts,
        timeout=timeout,
        enable_graph=enable_graph,
        enable_opt=enable_opt,
    )
    return motion_gen.plan_single(start_state, goal_pose, plan_cfg)

