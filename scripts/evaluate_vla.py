import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA-benchmark for Isaac Lab environments.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--policy", type=str, default='random', help="Name of the policy.")
# 新增：FROM_JSON 参数，0: 生成JSON, 1: 消费JSON, 2: 独立随机(默认)
parser.add_argument("--from_json", type=int, default=2, help="0: Record JSON, 1: Replay JSON, 2: Pure Random")

parser.add_argument("--asset_root_path",type=str,default="/home/junzhe/Benchmark")
parser.add_argument("--task_scene_name",type=str,default="Spawn_ds_st_sparse_EnvCfg")
parser.add_argument(
    "--robot_id", type=str, default="realman_dual_left_arm",
    help="评估侧机器人 ID，对应 evaluation/robot_registry.py 中的注册键。新机器人需在 REGISTRY 中注册。",
)
parser.add_argument(
    "--sim_device", type=str, default=None,
    help="Isaac Lab 仿真/渲染使用的 GPU，如 cuda:6。未指定时使用 --device 或 cuda:0。",
)
parser.add_argument(
    "--curobo_device", type=str, default=None,
    help="Curobo 运动规划使用的 GPU，如 cuda:5。仅在使用 curobo_reach_box 策略时生效。",
)
parser.add_argument(
    "--use_mesh_obstacles", action="store_true",
    help="Curobo 使用 USD/OBJ 提取的 mesh 作为障碍物（与 Isaac 碰撞体一致）。需设置 ASSET_ROOT_PATH。",
)

args_cli, _ = parser.parse_known_args()

if args_cli.use_mesh_obstacles:
    os.environ["CUROBO_USE_MESH_OBSTACLES"] = "1"

# 在 AppLauncher 启动前，将 --sim_device 同步到 --device，否则 Isaac Sim 会用默认 cuda:0 初始化，
# 导致后续 env_cfg.sim.device=cuda:6 时出现 invalid device ordinal
if args_cli.sim_device is not None:
    args_cli.device = args_cli.sim_device
    # 强制 Vulkan 渲染器使用指定 GPU（否则可能仍用 GPU 0）
    gpu_id = int(args_cli.sim_device.split(":")[-1])
    renderer_arg = f"--/renderer/activeGpu={gpu_id}"
    args_cli.kit_args = (args_cli.kit_args or "").strip()
    args_cli.kit_args = f"{args_cli.kit_args} {renderer_arg}".strip() if args_cli.kit_args else renderer_arg

# VLA 评估依赖相机观测，必须启用相机渲染
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import sys

if not os.path.exists(args_cli.asset_root_path):
    print(f"资产路径{args_cli.asset_root_path}未配置！请检查")
    exit()
else:
    print(f"Asset Root Path: {args_cli.asset_root_path}")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

import isaaclab_tasks
import isaaclab_logistics_vla
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_logistics_vla.evaluation.evaluator.vla_evaluator import VLA_Evaluator

from isaaclab_logistics_vla.utils.register import register
register.auto_scan("isaaclab_logistics_vla.tasks")


def main():
    # --- 修改点 3: 使用 register 加载配置 ---
    # 动态从命令行参数加载，默认会加载 Spawn_ss_st_sparse_with_obstacles_EnvCfg
    print(f"正在加载任务配置: {args_cli.task_scene_name}")
    env_cfg = register.load_env_configs(f'{args_cli.task_scene_name}')()
    
    # 如果命令行指定了环境数量，则覆盖配置
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # 仿真/渲染设备：--sim_device > --device > cuda:0
    env_cfg.sim.device = (
        args_cli.sim_device or args_cli.device or "cuda:0"
    )

    # Curobo 计算设备（计算放 5，渲染放 6 时：--curobo_device cuda:5 --sim_device cuda:6）
    if args_cli.curobo_device:
        os.environ["CUROBO_DEVICE"] = args_cli.curobo_device

    # 初始化评估器并运行
    evaluator = VLA_Evaluator(
        env_cfg=env_cfg,
        policy=args_cli.policy,
        from_json=args_cli.from_json,
        robot_id=args_cli.robot_id,
    )
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()