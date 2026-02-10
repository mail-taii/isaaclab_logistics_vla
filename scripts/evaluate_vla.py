import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA-benchmark for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--policy", type=str, default='random', help="Name of the policy.")
parser.add_argument("--device", type=str, default='cuda:0')
# 新增：FROM_JSON 参数，0: 生成JSON, 1: 消费JSON, 2: 独立随机(默认)
parser.add_argument("--from_json", type=int, default=0, help="0: Record JSON, 1: Replay JSON, 2: Pure Random")

<<<<<<< HEAD
parser.add_argument("--asset_root_path",type=str,default="/home/junzhe/code/model/Benchmark/")
parser.add_argument("--task_scene_name",type=str,default="Spawn_ss_st_stack_EnvCfg")
=======
parser.add_argument("--asset_root_path",type=str,default="/home/daniel/fff/model_files/benchmark/")
parser.add_argument("--task_scene_name",type=str,default="Spawn_ss_st_dense_EnvCfg")
>>>>>>> 7bca851 (dense_scene_ss-st v0.1)

args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import os
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

<<<<<<< HEAD
#from isaaclab_logistics_vla.tasks.test_tasks.dual_arm_pick_and_place_series.env_cfg import DualArmPickAndPlaceEnvCfg 
#from isaaclab_logistics_vla.tasks.test_tasks.order_series.env_cfg import OrderEnvCfg # 我要把这里换成我的任务
from isaaclab_logistics_vla.tasks.ss_st_series.stack_scene.env_cfg import Spawn_ss_st_stack_EnvCfg
=======
<<<<<<< HEAD
# --- 修改点 2: 导入你的新任务配置 ---
# 确保此处的路径指向你新创建的带 obstacles 的文件夹
from isaaclab_logistics_vla.tasks.ss_st_series.sparse_scene_with_obstacles.env_cfg import Spawn_ss_st_sparse_with_obstacles_EnvCfg

=======
from isaaclab_logistics_vla.tasks.ss_st_series.sparse_scene.env_cfg import Spawn_ss_st_sparse_EnvCfg
from isaaclab_logistics_vla.tasks.ms_st_series.sparse_scene.env_cfg import Spawn_ms_st_sparse_EnvCfg
from isaaclab_logistics_vla.tasks.ss_st_series.dense_scene.env_cfg import Spawn_ss_st_dense_EnvCfg
>>>>>>> caaaf1d (dense_scene_ss-st v0.1)
>>>>>>> 7bca851 (dense_scene_ss-st v0.1)
from isaaclab_logistics_vla.evaluation.evaluator.vla_evaluator import VLA_Evaluator

from isaaclab_logistics_vla.utils.register import register
# 只需让注册器自行import所有类
register.auto_scan("isaaclab_logistics_vla.tasks")

def main():
    print(f"正在加载任务配置: {args_cli.task_scene_name}")
    env_cfg = register.load_env_configs(f'{args_cli.task_scene_name}')()
    
    # 如果命令行指定了环境数量，则覆盖配置
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # 初始化评估器并运行
    evaluator = VLA_Evaluator(
        env_cfg=env_cfg, 
        policy=args_cli.policy, 
        from_json=args_cli.from_json
    )
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()