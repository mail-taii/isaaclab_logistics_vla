"""
本团队固定机器人 + 固定评测场景时的**写死路径**。

- 使用者一般**不必**再传 ``cache_path`` / ``--asset_root_path`` / ``--task_scene_name``；
- 若目录布局不同，仍可显式覆盖各入口的对应参数。

首次部署示例（在仓库根执行）::

    python scripts/generate_curobo_robot_kinematics_yaml.py \\
        --urdf <Benchmark>/robot/realman/realman_franka_ee.urdf \\
        --output isaaclab_logistics_vla/assets/curobo/realman_kinematics.yaml
"""
from __future__ import annotations

import os

# ``.../isaaclab_logistics_vla/isaaclab_logistics_vla``（内含 utils、configs、assets）
_INNER_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 仓库根目录 ``.../isaaclab_logistics_vla``（与 scripts/ 同级）
_REPO_ROOT = os.path.dirname(_INNER_PKG)

# cuRobo Realman 双臂运动学 YAML（包内固定位置）
DEFAULT_CUROBO_KINEMATICS_YAML = os.path.join(
    _INNER_PKG,
    "assets",
    "curobo",
    "realman_kinematics.yaml",
)

# Isaac 评测资产根：默认与仓库根目录**同级**的 ``Benchmark``（本团队常用布局）
DEFAULT_VLA_ASSET_ROOT_PATH = os.path.join(os.path.dirname(_REPO_ROOT), "Benchmark")

# 固定任务/场景配置名（与 register 中环境名一致）
DEFAULT_VLA_TASK_SCENE_NAME = "Spawn_ms_st_dense_EnvCfg"
