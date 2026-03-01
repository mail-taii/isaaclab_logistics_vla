#!/usr/bin/env python3
"""
从 USD 中提取碰撞/可视化 mesh，导出为 OBJ 供 Curobo 使用。

支持 UsdGeom.Mesh 和 UsdGeom.Cube。若 pxr 不可用，则用 trimesh 创建长方体作为回退。

用法：
  python tools/extract_mesh_from_usd.py
  python tools/extract_mesh_from_usd.py --input /path/to/Box.usd --output /path/to/Box.obj
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
ASSET_ROOT = os.environ.get("ASSET_ROOT_PATH", str(ROOT.parent / "Benchmark"))
DEFAULT_USD = Path(ASSET_ROOT) / "env" / "Box.usd"
DEFAULT_OBJ = ROOT / "isaaclab_logistics_vla" / "assets" / "meshes" / "Box.obj"


def _extract_mesh_pxr(usd_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """使用 pxr.Usd 提取 mesh，返回 (vertices, faces) 或 None。"""
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return None

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return None

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            if pts is None:
                continue
            vertices = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
            if face_vertex_counts is None or face_vertex_indices is None:
                continue
            face_vertex_counts = list(face_vertex_counts)
            face_vertex_indices = list(face_vertex_indices)
            #  triangulate if needed
            faces_list: list[list[int]] = []
            idx = 0
            for count in face_vertex_counts:
                if count == 3:
                    faces_list.append(face_vertex_indices[idx : idx + 3])
                    idx += 3
                elif count == 4:
                    f = face_vertex_indices[idx : idx + 4]
                    faces_list.extend([[f[0], f[1], f[2]], [f[0], f[2], f[3]]])
                    idx += 4
                else:
                    for i in range(1, count - 1):
                        faces_list.append(
                            [
                                face_vertex_indices[idx],
                                face_vertex_indices[idx + i],
                                face_vertex_indices[idx + i + 1],
                            ]
                        )
                    idx += count
            faces = np.array(faces_list, dtype=np.int64)
            return vertices, faces
        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            size_attr = cube.GetSizeAttr()
            size = float(size_attr.Get()) if size_attr else 1.0
            m = trimesh.creation.box(extents=[size, size, size])
            return m.vertices, m.faces

    return None


def _extract_mesh_trimesh(usd_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """尝试用 trimesh 直接加载 USD（部分版本支持）。"""
    try:
        m = trimesh.load(str(usd_path), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        if m is not None and hasattr(m, "vertices") and hasattr(m, "faces"):
            return np.asarray(m.vertices), np.asarray(m.faces)
    except Exception:
        pass
    return None


def _fallback_box(dims: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """使用 WORK_BOX_PARAMS 尺寸创建长方体 mesh。"""
    m = trimesh.creation.box(extents=list(dims))
    return m.vertices, m.faces


def extract_mesh(
    usd_path: Path,
    fallback_dims: tuple[float, float, float] = (0.36, 0.56, 0.23),
) -> tuple[np.ndarray, np.ndarray]:
    """
    从 USD 提取 mesh，返回 (vertices, faces)。
    优先 pxr，其次 trimesh，最后用 fallback_dims 创建长方体。
    """
    if usd_path.exists():
        result = _extract_mesh_pxr(usd_path)
        if result is not None:
            return result
        result = _extract_mesh_trimesh(usd_path)
        if result is not None:
            return result
    return _fallback_box(fallback_dims)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 USD 提取 mesh 并导出 OBJ")
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_USD), help="输入 USD 路径")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OBJ), help="输出 OBJ 路径")
    parser.add_argument(
        "--dims",
        type=float,
        nargs=3,
        default=[0.36, 0.56, 0.23],
        help="pxr/trimesh 失败时的回退尺寸 (x y z)",
    )
    args = parser.parse_args()

    usd_path = Path(args.input)
    out_path = Path(args.output)
    fallback_dims = tuple(args.dims)

    vertices, faces = extract_mesh(usd_path, fallback_dims)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    print(f"已导出: {out_path} (vertices={len(vertices)}, faces={len(faces)})")


if __name__ == "__main__":
    main()
