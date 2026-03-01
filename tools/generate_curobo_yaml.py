#!/usr/bin/env python3
"""
根据环境碰撞箱定义自动生成 Curobo 可用的场景障碍物 YAML。
数据与 utils/env_collision_boxes.py 中 get_env_collision_boxes() 保持一致。

用法（在项目根目录）：
  python tools/generate_curobo_yaml.py
  # 默认输出到 isaaclab_logistics_vla/configs/worlds/scene_obstacles.yml

  python tools/generate_curobo_yaml.py --output /path/to/scene_obstacles.yml

坐标系：YAML 中的 pose 为「相对机器人臂基/肩膀」的 (x,y,z,qw,qx,qy,qz)。
Z 轴必须含平台高度：机器人本体 0.216m + 平台 0.8m = 1.016m，否则 Curobo 会误判障碍物在头顶（幽灵方块）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 与 env_collision_boxes.get_env_collision_boxes() 一致，不依赖 isaaclab
BOX_SIZE = (0.36, 0.56, 0.23)  # (X_LENGTH, Y_LENGTH, Z_LENGTH)
WORLD_BOXES = [
    ("s_box_1", (1.57989, 1.33474, 0.750)),
    ("s_box_2", (1.02500, 1.33614, 0.725)),
    ("s_box_3", (0.51025, 1.33614, 0.750)),
    ("t_box_1", (1.57989, 3.44290, 0.820)),
    ("t_box_2", (1.02500, 3.44290, 0.820)),
    ("t_box_3", (0.51000, 3.44290, 0.820)),
]

# 世界系下「机器人臂基/肩膀」参考点：本体 (0.96781, 2.28535, 0.216) + 平台高度 0.8
ROBOT_BASE_WORLD = (0.96781, 2.28535, 0.216 + 0.8)


def world_to_robot_frame(center: tuple[float, float, float]) -> tuple[float, float, float]:
    """世界系 (x,y,z) 转机器人基座系：X前 Y左 Z上。"""
    wx, wy, wz = center
    rbx, rby, rbz = ROBOT_BASE_WORLD
    rx = -(wy - rby)
    ry = wx - rbx
    rz = wz - rbz
    return (rx, ry, rz)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Curobo 场景障碍物 YAML")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(ROOT / "isaaclab_logistics_vla" / "configs" / "worlds" / "scene_obstacles.yml"),
        help="输出 YAML 路径",
    )
    args = parser.parse_args()

    boxes = [{"name": n, "center": c, "size": BOX_SIZE} for n, c in WORLD_BOXES]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Curobo 常用：pose = [x, y, z, qw, qx, qy, qz]，dims = [lx, ly, lz]（与当前 policy 中 dims 顺序一致）
    cuboids = []
    for b in boxes:
        rx, ry, rz = world_to_robot_frame(b["center"])
        pose = [rx, ry, rz, 1.0, 0.0, 0.0, 0.0]
        sz = b["size"]
        dims = [sz[1], sz[0], sz[2]]  # 与 policy 中 dims 顺序一致
        cuboids.append({"name": b["name"], "pose": pose, "dims": dims})

    import yaml
    data = {"cuboids": cuboids}

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Curobo 场景障碍物：pose [x,y,z,qw,qx,qy,qz] 为机器人基座系，dims [lx,ly,lz] 单位米\n")
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"已生成 {out_path}，共 {len(cuboids)} 个障碍物。")
    for c in cuboids:
        print(f"  - {c['name']}: pose={c['pose'][:3]}, dims={c['dims']}")


if __name__ == "__main__":
    main()
