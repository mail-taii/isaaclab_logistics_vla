#!/usr/bin/env python3
"""在任务环境中做一次 ``plan_dual_world_to_world_auto`` 冒烟测试。

**启动与导入顺序与 ``scripts/evaluate_vla.py`` 一致**（不修改 tasks / mdp / env_cfg）：
``AppLauncher`` → 资产路径检查 → ``isaaclab_tasks`` / ``isaaclab_logistics_vla`` →
``register.auto_scan`` → ``register.load_env_configs`` → ``VLAIsaacEnv``。

本脚本仅在 ``evaluate_vla`` 的 argparse 基础上增加 cuRobo 测试专用参数；其余与评测入口相同。

用法::

    cd <仓库根>/isaaclab_logistics_vla
    python scripts/test_curobo_world_auto_in_task_env.py --headless --device cuda:0 --num_envs 1
"""
import argparse
import os

from isaaclab.app import AppLauncher

# 注意：此处不能 import isaaclab_logistics_vla（会早于 AppLauncher 触发包初始化）。
# 与 ``configs/pinned_eval_curobo.py``、``evaluate_vla.py`` 保持同一套默认目录约定。
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
_DEFAULT_VLA_ASSET_ROOT_PATH = os.path.join(os.path.dirname(_REPO_ROOT), "Benchmark")
_DEFAULT_VLA_TASK_SCENE_NAME = "Spawn_ms_st_dense_EnvCfg"

# ---------- 与 evaluate_vla.py 相同的入口参数 ----------
parser = argparse.ArgumentParser(description="在任务环境中测试 plan_dual_world_to_world_auto（evaluate_vla 同源启动链）。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--policy", type=str, default="random", help="Name of the policy.")
parser.add_argument("--from_json", type=int, default=2, help="0: Record JSON, 1: Replay JSON, 2: Pure Random")
parser.add_argument(
    "--asset_root_path",
    type=str,
    default=_DEFAULT_VLA_ASSET_ROOT_PATH,
    help="默认：仓库同级 Benchmark 目录（与 configs/pinned_eval_curobo 一致）",
)
parser.add_argument(
    "--task_scene_name",
    type=str,
    default=_DEFAULT_VLA_TASK_SCENE_NAME,
    help="默认：团队固定场景配置名",
)

# ---------- 仅本脚本的 cuRobo 冒烟参数 ----------
parser.add_argument(
    "--goal_dx",
    type=float,
    default=0.005,
    help="World 系目标相对起点的平移 x（米）",
)
parser.add_argument("--goal_dy", type=float, default=0.0)
parser.add_argument("--goal_dz", type=float, default=0.0)
parser.add_argument(
    "--set_default_world",
    action="store_true",
    default=False,
    help="True：使用封装内默认桌子+箱子障碍（会变换到当前 base）",
)
parser.add_argument("--max_attempts", type=int, default=10)
parser.add_argument("--timeout", type=float, default=5.0)
parser.add_argument("--warmup_steps", type=int, default=0, help="reset 后先执行若干步零动作再规划")

# 与 Isaac Lab 一致：提供 --headless / --device / --enable_cameras 等
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # noqa: F841  须保持引用至进程结束

import gymnasium as gym  # noqa: F401  与 evaluate_vla.py 保持一致

import numpy as np
import torch

if not os.path.exists(args_cli.asset_root_path):
    print(f"资产路径{args_cli.asset_root_path}未配置！请检查")
    exit()
else:
    print(f"Asset Root Path: {args_cli.asset_root_path}")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

import isaaclab_tasks  # noqa: E402
import isaaclab_logistics_vla  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402, F401  与 evaluate_vla.py 保持一致

from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv  # noqa: E402
from isaaclab_logistics_vla.evaluation.models.policy import CuroboPlannerPolicy  # noqa: E402
from isaaclab_logistics_vla.utils.register import register  # noqa: E402

register.auto_scan("isaaclab_logistics_vla.tasks")


def main() -> int:
    # 与 evaluate_vla.main 相同的配置加载方式
    print(f"正在加载任务配置: {args_cli.task_scene_name}")
    env_cfg = register.load_env_configs(f"{args_cli.task_scene_name}")()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    if hasattr(env_cfg.commands, "object_commands"):
        env_cfg.commands.object_commands.from_json = args_cli.from_json

    print("[test] 正在创建 VLAIsaacEnv …", flush=True)
    env = VLAIsaacEnv(cfg=env_cfg)
    print("[test] env 已创建，执行 reset() …", flush=True)
    env.reset()
    print("[test] reset() 完成。", flush=True)

    goal_delta = np.array([args_cli.goal_dx, args_cli.goal_dy, args_cli.goal_dz], dtype=np.float64)
    print(
        "[test] 构造 CuroboPlannerPolicy（MotionGen.warmup 可能需数十秒）…",
        flush=True,
    )
    policy = CuroboPlannerPolicy(
        env,
        device=args_cli.device,
        goal_delta_w=goal_delta,
        debug_print=True,
        debug_print_every=1,
        debug_coordinates=True,
    )

    for _ in range(max(0, args_cli.warmup_steps)):
        z = torch.zeros((env.num_envs, 17), device=env.device, dtype=torch.float32)
        z[:, 16] = policy.platform_action
        env.step(z)

    env_id = 0
    start_left = policy._get_ee_pose_w(env_id, link_name="panda_left_hand")
    start_right = policy._get_ee_pose_w(env_id, link_name="panda_right_hand")
    goal_left = policy._make_goal_near_start(start_left)
    goal_right = policy._make_goal_near_start(start_right)

    print("[test] 调用 plan_dual_world_to_world_auto …", flush=True)
    out = policy.planner.plan_dual_world_to_world_auto(
        start_poses_world={"left": start_left, "right": start_right},
        goal_poses_world={"left": goal_left, "right": goal_right},
        set_default_world=args_cli.set_default_world,
        max_attempts=args_cli.max_attempts,
        timeout=args_cli.timeout,
        enable_graph=False,
        enable_opt=True,
        debug_coordinate_check=True,
    )

    ok = out.get("status") == "Success"
    print(f"[test] status={out.get('status')} detail={out.get('detail')!r}", flush=True)
    ik = out.get("ik") or {}
    print(f"[test] ik_success={ik.get('success')} ik_detail={ik.get('detail')!r}", flush=True)
    traj = out.get("position")
    if traj is not None:
        print(f"[test] traj shape={traj.shape} dof={policy.planner.dof}", flush=True)
    else:
        print("[test] traj=None", flush=True)
    if out.get("_debug"):
        print(f"[test] _debug keys={list(out['_debug'].keys())}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
