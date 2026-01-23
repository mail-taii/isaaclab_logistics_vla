from __future__ import annotations

import random
from dataclasses import field, MISSING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import (
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


#这个地方LJY加了一行注释
@configclass
class TargetObjectSpec:
    """Definition of a single spawnable target物体."""

    name: str = MISSING
    """Unique name for this物体类型."""

    usd_path: str | None = None
    """Path to an external USD资产. Optional, can stay None for primitive shapes."""

    num_instances: int = 1
    """How many拷贝 of this资产 should be预先实例化 (per environment)."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Uniform/xyz scale applied when using USD资产."""

    size: tuple[float, float, float] | None = None
    """Primitive box尺寸 (m). If None, defaults to 0.1m cube."""

    mass: float = 0.1
    """Mass of each实例 (kg)."""

    color: tuple[float, float, float] | None = None
    """Optional RGB颜色 (0-1) for primitive几何体."""

    spawn_cfg: (
        sim_utils.UsdFileCfg
        | sim_utils.CuboidCfg
        | sim_utils.SphereCfg
        | sim_utils.CylinderCfg
        | None
    ) = None
    """Custom spawn配置. If provided, usd_path/size will be ignored."""


@configclass
class TargetObjectRandomizationCfg:
    """High-level configuration describing how to随机化目标物体."""

    asset_pool: list[TargetObjectSpec] = field(default_factory=list)
    """可被随机选择的资产集合."""

    max_spawned_objects: int = 1
    """每个环境本次reset最多放多少个物体 (<= 总实例数)."""

    pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.1),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.2, 0.2),
        }
    )
    """采样姿态范围 (以场景原点为中心，单位m / rad)。"""

    min_separation: float = 0.05
    """两物体中心的最小距离 (m)。"""

    offstage_pose: tuple[float, float, float] = (10.0, 10.0, 10.0)
    """未选中物体的停放位置 (移出工作区即可)。"""


def _build_spawn_cfg(spec: TargetObjectSpec):
    """Create a spawner config from a TargetObjectSpec."""
    if spec.spawn_cfg is not None:
        return spec.spawn_cfg

    # Common rigid / collision defaults
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        rigid_body_enabled=True,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.003,
        rest_offset=0.0,
    )

    if spec.usd_path:
        return sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            scale=spec.scale,
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            rigid_props=rigid_props,
            collision_props=collision_props,
        )

    # 默认立方体
    size = spec.size if spec.size is not None else (0.1, 0.1, 0.1)
    visual_material = None
    if spec.color is not None:
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color)
    return sim_utils.CuboidCfg(
        size=size,
        mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
        rigid_props=rigid_props,
        collision_props=collision_props,
        visual_material=visual_material,
    )


def build_object_collection_cfg(
    randomization_cfg: TargetObjectRandomizationCfg,
    prim_prefix: str = "{ENV_REGEX_NS}/Object",
) -> RigidObjectCollectionCfg:
    """Construct a RigidObjectCollectionCfg with all候选资产实例化.

    Note:
        Use flat prim paths (no extra subfolder) to avoid missing parent prim errors
        during spawning. For example: `{ENV_REGEX_NS}/Object_red_cube_0`.
    """
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for spec in randomization_cfg.asset_pool:
        spawn_cfg = _build_spawn_cfg(spec)
        for idx in range(spec.num_instances):
            prim_path = f"{prim_prefix}_{spec.name}_{idx}"
            rigid_objects[f"{spec.name}_{idx}"] = RigidObjectCfg(
                prim_path=prim_path,
                init_state=RigidObjectCfg.InitialStateCfg(pos=randomization_cfg.offstage_pose, rot=[1.0, 0.0, 0.0, 0.0]),
                spawn=spawn_cfg,
            )
    if not rigid_objects:
        raise ValueError("asset_pool is empty; add at least one TargetObjectSpec.")
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def _sample_object_poses(
    num_objects: int,
    pose_range: dict[str, tuple[float, float]],
    min_separation: float,
    max_tries: int = 500,
) -> list[list[float]]:
    """Sample object poses with简单的最小间距约束."""
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    ranges = [pose_range.get(k, (0.0, 0.0)) for k in keys]
    poses: list[list[float]] = []
    for _ in range(num_objects):
        for attempt in range(max_tries):
            sample = [random.uniform(r[0], r[1]) for r in ranges]
            if not poses:
                poses.append(sample)
                break
            if attempt == max_tries - 1:
                poses.append(sample)
                break
            too_close = False
            for p in poses:
                dx = sample[0] - p[0]
                dy = sample[1] - p[1]
                dz = sample[2] - p[2]
                if (dx * dx + dy * dy + dz * dz) ** 0.5 < min_separation:
                    too_close = True
                    break
            if not too_close:
                poses.append(sample)
                break
    return poses


def randomize_target_objects(
    env,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    randomization_cfg: TargetObjectRandomizationCfg,
    source_cfg: SceneEntityCfg | None = None,
    source_size_xy: tuple[float, float] | None = None,
):
    """Event函数：在reset时随机选择物体并放置到工作区."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    asset = env.scene[object_cfg.name]
    if isinstance(asset, RigidObjectCollection):
        max_spawn = min(randomization_cfg.max_spawned_objects, asset.num_objects)
        active_ids = getattr(env, "_active_object_indices", None)
        if active_ids is None or active_ids.numel() != env.scene.num_envs:
            active_ids = torch.zeros(env.scene.num_envs, dtype=torch.long, device=env.device)

        # 如果提供了 source_cfg + source_size_xy，则严格在红色源区域内部采样 XY 和 Z，
        # 否则退回到基于 pose_range 的通用采样。
        if source_cfg is not None and source_size_xy is not None:
            source = env.scene[source_cfg.name]
            source_centers = source.data.root_pos_w  # (num_envs, 3)
            half_extent_x = 0.5 * float(source_size_xy[0])
            half_extent_y = 0.5 * float(source_size_xy[1])
            # Z 坐标也基于 source_area 的中心，只添加一个小的偏移范围（物体放在源区域表面上）
            # 默认偏移范围：0.01-0.05m（源区域表面上方）
            z_offset_range = randomization_cfg.pose_range.get("z_offset", (0.01, 0.05))
            yaw_range = randomization_cfg.pose_range["yaw"]
        else:
            source_centers = None
            half_extent_x = half_extent_y = None
            z_offset_range = None
            yaw_range = None

        for env_id in env_ids.tolist():
            num_to_spawn = max(1, random.randint(1, max_spawn))
            chosen = torch.randperm(asset.num_objects, device=env.device)[:num_to_spawn]

            if source_centers is not None:
                # 严格在源区域内部采样：以 SourceArea 中心为基准，在 XY 半尺寸内均匀采样
                # Z 坐标也基于源区域中心，只添加小的表面偏移
                center = source_centers[env_id]
                poses = []
                for _ in range(num_to_spawn):
                    local_x = random.uniform(-half_extent_x, half_extent_x)
                    local_y = random.uniform(-half_extent_y, half_extent_y)
                    # Z 坐标 = 源区域中心 Z + 表面偏移（让物体放在源区域表面上）
                    z_offset = random.uniform(z_offset_range[0], z_offset_range[1])
                    z = center[2].item() + z_offset
                    yaw = random.uniform(yaw_range[0], yaw_range[1])
                    # roll / pitch 固定为 0
                    poses.append(
                        [
                            center[0].item() + local_x,
                            center[1].item() + local_y,
                            z,
                            0.0,
                            0.0,
                            yaw,
                        ]
                    )
            else:
                # 退回到原先的通用采样逻辑
                poses = _sample_object_poses(
                    num_objects=num_to_spawn,
                    pose_range=randomization_cfg.pose_range,
                    min_separation=randomization_cfg.min_separation,
                )

            # 初始化全部移出场景
            # state layout expected by RigidObjectCollection: [pos(3), quat(4), lin_vel(3), ang_vel(3)] -> 13 values
            out_pose = torch.tensor(
                [
                    randomization_cfg.offstage_pose[0],
                    randomization_cfg.offstage_pose[1],
                    randomization_cfg.offstage_pose[2],
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device=env.device,
            )
            object_state = out_pose.repeat(asset.num_objects, 1)

            # 填入被选中的物体
            # 注意：严格模式下，poses 已经是世界坐标（基于 source.root_pos_w），不需要再加 env_origin
            # 通用模式下，poses 是相对于环境原点的局部坐标，需要加 env_origin
            use_world_coords = source_centers is not None
            env_origin = env.scene.env_origins[env_id] if not use_world_coords else None
            
            for idx, pose in zip(chosen.tolist(), poses):
                if use_world_coords:
                    # 严格模式：poses 已经是世界坐标，直接使用
                    pos = torch.tensor(pose[:3], device=env.device)
                else:
                    # 通用模式：poses 是局部坐标，需要加 env_origin
                    pos = torch.tensor(pose[:3], device=env.device) + env_origin
                quat = math_utils.quat_from_euler_xyz(
                    torch.tensor([pose[3]], device=env.device),
                    torch.tensor([pose[4]], device=env.device),
                    torch.tensor([pose[5]], device=env.device),
                )[0]
                object_state[idx, :3] = pos
                object_state[idx, 3:7] = quat

            asset.write_object_state_to_sim(
                object_state=object_state.unsqueeze(0),
                env_ids=torch.tensor([env_id], device=env.device),
            )

            # 记录当前焦点物体
            active_ids[env_id] = chosen[0]

        env._active_object_indices = active_ids

    elif isinstance(asset, RigidObject):
        poses = _sample_object_poses(
            num_objects=1,
            pose_range=randomization_cfg.pose_range,
            min_separation=randomization_cfg.min_separation,
        )[0]
        positions = torch.tensor(poses[:3], device=env.device) + env.scene.env_origins[env_ids][:, :3]
        quats = math_utils.quat_from_euler_xyz(
            torch.tensor([poses[3]], device=env.device),
            torch.tensor([poses[4]], device=env.device),
            torch.tensor([poses[5]], device=env.device),
        )
        asset.write_root_pose_to_sim(
            torch.cat([positions, quats.repeat(env_ids.shape[0], 1)], dim=-1),
            env_ids=env_ids,
        )
        if not hasattr(env, "_active_object_indices"):
            env._active_object_indices = torch.zeros(env.scene.num_envs, dtype=torch.long, device=env.device)


def get_active_object_pose_w(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前任务关注的物体位姿 (pos_w, quat_w)."""
    asset = env.scene[object_cfg.name]
    if isinstance(asset, RigidObjectCollection):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
        active_ids = getattr(env, "_active_object_indices", None)
        if active_ids is None or active_ids.numel() != env.scene.num_envs:
            active_ids = torch.zeros(env.scene.num_envs, dtype=torch.long, device=env.device)
        pos = asset.data.object_pos_w[env_ids, active_ids]
        quat = asset.data.object_quat_w[env_ids, active_ids]
    else:
        pos = asset.data.root_pos_w
        quat = asset.data.root_quat_w
    return pos, quat

