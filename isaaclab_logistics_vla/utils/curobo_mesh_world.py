"""
从 USD/OBJ 构建 Curobo 可用的 mesh 障碍物世界。

将场景中的 s_box/t_box 用 mesh 表示（与 Isaac 碰撞体一致），
并转换到机器人基座系，供 CollisionCheckerType.MESH 使用。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import trimesh

# 与 generate_curobo_yaml / base_scene_cfg 一致
BOX_SIZE = (0.36, 0.56, 0.23)  # (X_LENGTH, Y_LENGTH, Z_LENGTH)
WORLD_BOXES = [
    ("s_box_1", (1.57989, 1.33474, 0.750)),
    ("s_box_2", (1.02500, 1.33614, 0.725)),
    ("s_box_3", (0.51025, 1.33614, 0.750)),
    ("t_box_1", (1.57989, 3.44290, 0.820)),
    ("t_box_2", (1.02500, 3.44290, 0.820)),
    ("t_box_3", (0.51000, 3.44290, 0.820)),
]
# sparse 任务：与 Policy 一致。根 (0.96781, 2.1, 0.216) + offset (0, -0.11663, 0.271+0.65) → 臂基
ROBOT_BASE_WORLD = (0.96781, 1.98337, 1.137)


def world_to_robot_frame(center: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """世界系 → 臂基系。Isaac 与 Realman 轴一致，纯减法。"""
    wx, wy, wz = center
    rbx, rby, rbz = ROBOT_BASE_WORLD
    rx = wx - rbx
    ry = wy - rby
    rz = wz - rbz
    return (rx, ry, rz)


def _ensure_box_obj(asset_root: str) -> str:
    """确保 Box.obj 存在，若不存在则创建长方体 mesh。返回 OBJ 绝对路径。"""
    asset_path = Path(asset_root)
    box_usd = asset_path / "env" / "Box.usd"
    box_obj = asset_path / "env" / "Box.obj"
    pkg_dir = Path(__file__).resolve().parent.parent
    fallback_obj = pkg_dir / "assets" / "meshes" / "Box.obj"

    if box_obj.exists():
        return str(box_obj.resolve())
    if fallback_obj.exists():
        return str(fallback_obj.resolve())

    # 尝试从 USD 提取（与 tools/extract_mesh_from_usd 一致）
    if box_usd.exists():
        try:
            import sys
            tools_dir = pkg_dir.parent / "tools"
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            from extract_mesh_from_usd import extract_mesh

            vertices, faces = extract_mesh(box_usd, BOX_SIZE)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            box_obj.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(box_obj))
            return str(box_obj.resolve())
        except Exception:
            pass

    # 回退：trimesh 创建长方体（与 WORK_BOX_PARAMS 尺寸一致）
    mesh = trimesh.creation.box(extents=list(BOX_SIZE))
    out_path = fallback_obj
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    return str(out_path.resolve())


# 空心箱子壁厚（米），用于方案 B：5 块薄板拼成敞口箱，避免凸包把内部封死
HOLLOW_BOX_WALL_THICKNESS = 0.01


def _hollow_box_cuboids(
    name_prefix: str,
    center_arm: Tuple[float, float, float],
    size_xyz: Tuple[float, float, float],
    wall_thickness: float = HOLLOW_BOX_WALL_THICKNESS,
) -> List[Tuple[str, List[float], List[float]]]:
    """
    单个箱子的 5 块薄板：1 底 + 4 侧。pose/dims 为 Curobo 格式 [x,y,z,qw,qx,qy,qz] 与 [lx,ly,lz]。

    Returns:
        [(name, pose, dims), ...]
    """
    rx, ry, rz = center_arm
    lx, ly, lz = size_xyz
    hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0
    t = wall_thickness
    quat = [1.0, 0.0, 0.0, 0.0]
    out: List[Tuple[str, List[float], List[float]]] = []
    # 底面（箱底，z 向下半厚度）
    out.append((f"{name_prefix}_bottom", [rx, ry, rz - hz] + quat, [lx, ly, t]))
    # 四侧：+Y -Y +X -X（中心在面心）
    out.append((f"{name_prefix}_wall_py", [rx, ry + hy, rz] + quat, [lx, t, lz]))
    out.append((f"{name_prefix}_wall_ny", [rx, ry - hy, rz] + quat, [lx, t, lz]))
    out.append((f"{name_prefix}_wall_px", [rx + hx, ry, rz] + quat, [t, ly, lz]))
    out.append((f"{name_prefix}_wall_nx", [rx - hx, ry, rz] + quat, [t, ly, lz]))
    return out


def get_hollow_box_world_for_curobo() -> "WorldConfig":
    """
    方案 B：用 5 个薄 Cuboid（1 底 + 4 侧）表示每个箱子，中间为空，避免凸包导致目标在“实心内”而 IK_FAIL。

    与 get_mesh_world_for_curobo 使用相同的 WORLD_BOXES、臂基系转换；返回纯 cuboid 的 WorldConfig，
    供 CollisionCheckerType.PRIMITIVE 使用。
    """
    from curobo.geom.types import WorldConfig

    cuboid_dict: dict = {}
    for name, center_world in WORLD_BOXES:
        center_arm = world_to_robot_frame(center_world)
        for part_name, pose, dims in _hollow_box_cuboids(name, center_arm, BOX_SIZE):
            cuboid_dict[part_name] = {"pose": pose, "dims": dims}
    return WorldConfig.from_dict({"cuboid": cuboid_dict})


def get_mesh_world_for_curobo(asset_root: Optional[str] = None) -> "WorldConfig":
    """
    构建 Curobo WorldConfig，包含 s_box/t_box 的 mesh 障碍物。

    Args:
        asset_root: 资源根路径，用于 Box.usd/Box.obj。默认 ASSET_ROOT_PATH 或 Benchmark。

    Returns:
        WorldConfig 实例，mesh 属性为障碍物 mesh 列表。
    """
    from curobo.geom.types import WorldConfig, Mesh

    root = asset_root or os.environ.get("ASSET_ROOT_PATH") or ""
    if not root:
        pkg_dir = Path(__file__).resolve().parent.parent
        root = str(pkg_dir.parent / "Benchmark")
    box_obj_path = _ensure_box_obj(root)

    meshes: List[Mesh] = []
    for name, center_world in WORLD_BOXES:
        rx, ry, rz = world_to_robot_frame(center_world)
        pose = [rx, ry, rz, 1.0, 0.0, 0.0, 0.0]
        mesh_obstacle = Mesh(
            name=name,
            pose=pose,
            file_path=box_obj_path,
            scale=[1.0, 1.0, 1.0],
        )
        meshes.append(mesh_obstacle)

    return WorldConfig(mesh=meshes)
