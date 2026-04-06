import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA-benchmark for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--policy",
    type=str,
    default="random",
    help="策略名：random、curobo_plan（cuRobo 双臂规划演示）等。",
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--from_json", type=int, default=2, help="0: Record JSON, 1: Replay JSON, 2: Pure Random")

parser.add_argument("--asset_root_path", type=str, default="/home/wst/model_files/benchmark")
parser.add_argument("--task_scene_name", type=str, default="Spawn_ms_st_dense_EnvCfg")
parser.add_argument(
    "--robot_id",
    type=str,
    default="realman_dual_left_arm",
    help="评估 registry（evaluation/robot_registry.py）的 robot_id；curobo_plan 时传给 CuRoboPlanPolicy。",
)
parser.add_argument(
    "--use_mesh_obstacles",
    action="store_true",
    help="设置 CUROBO_USE_MESH_OBSTACLES=1。",
)

args_cli, _ = parser.parse_known_args()

if args_cli.use_mesh_obstacles:
    os.environ["CUROBO_USE_MESH_OBSTACLES"] = "1"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if not os.path.exists(args_cli.asset_root_path):
    print(f"资产路径{args_cli.asset_root_path}未配置！请检查")
    exit()
else:
    print(f"Asset Root Path: {args_cli.asset_root_path}")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

import isaaclab_tasks
import isaaclab_logistics_vla

from isaaclab_logistics_vla.evaluation.evaluator.vla_evaluator import VLA_Evaluator

from isaaclab_logistics_vla.utils.register import register

register.auto_scan("isaaclab_logistics_vla.tasks")


def main():
    print(f"正在加载任务配置: {args_cli.task_scene_name}")
    env_cfg = register.load_env_configs(f"{args_cli.task_scene_name}")()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    evaluator = VLA_Evaluator(
        env_cfg=env_cfg,
        policy=args_cli.policy,
        from_json=args_cli.from_json,
        device=env_cfg.sim.device,
        robot_id=args_cli.robot_id,
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
