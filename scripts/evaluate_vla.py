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

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks
import isaaclab_logistics_vla
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_logistics_vla.tasks.realman_lift.realman_lift_env_cfg import LiftEnvCfg
from isaaclab_logistics_vla.evaluation.evaluator.vla_evaluator import VLA_Evaluator

def main():
    env_cfg = LiftEnvCfg()
    #env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    evaluator = VLA_Evaluator(env_cfg=env_cfg, policy=args_cli.policy)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()