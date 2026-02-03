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

parser.add_argument("--asset_root_path",type=str,default="/home/mail-robo/Benchmark")
parser.add_argument("--task_scene_name",type=str,default="Spawn_ss_st_sparse_EnvCfg")

args_cli,_ = parser.parse_known_args()

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
    print(args_cli.asset_root_path)
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

import isaaclab_tasks
import isaaclab_logistics_vla
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_logistics_vla.tasks.test_tasks.dual_arm_pick_and_place_series.env_cfg import DualArmPickAndPlaceEnvCfg 
from isaaclab_logistics_vla.tasks.test_tasks.order_series.env_cfg import OrderEnvCfg # 我要把这里换成我的任务
from isaaclab_logistics_vla.tasks.ss_st_series.sparse_scene.env_cfg import Spawn_ss_st_sparse_EnvCfg
from isaaclab_logistics_vla.evaluation.evaluator.vla_evaluator import VLA_Evaluator

from isaaclab_logistics_vla.utils.register import register

def main():
    #env_cfg = Spawn_ss_st_sparse_EnvCfg()
    env_cfg = register.load_env_configs(f'{args_cli.task_scene_name}')()
    #env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    evaluator = VLA_Evaluator(env_cfg=env_cfg, policy=args_cli.policy)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()