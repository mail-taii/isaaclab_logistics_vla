import argparse
import sys

# 1. 启动 Isaac Sim 应用 (必须最先执行)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA-benchmark for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--policy", type=str, default='random', help="Name of the policy.")
parser.add_argument("--device", type=str, default='cuda:0')

# 解析参数
args_cli, _ = parser.parse_known_args()

# 启动 App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入仿真相关库 (必须在 App 启动后)
import gymnasium as gym
import torch

import isaaclab_tasks
import isaaclab_logistics_vla
from isaaclab_tasks.utils import parse_env_cfg

# 引入你的环境配置
# 注意：OrderEnvCfg 内部会引用 command_cfg.py -> NewOrderCommandCfg.py
from isaaclab_logistics_vla.tasks.test_tasks.order_series.env_cfg import OrderEnvCfg
from isaaclab_logistics_vla.evaluation.evaluator.vla_evaluator import VLA_Evaluator

def main():
    print(f"[INFO] Loading OrderEnvCfg...")
    env_cfg = OrderEnvCfg()

    # --- 修改点 1: 启用 num_envs 设置 ---
    # 之前这行被注释掉了，导致命令行参数无效。
    # 强制设为 args_cli.num_envs (默认为 1)，方便肉眼观察 9 个物品的分配
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # 设置计算设备
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # --- 修改点 2: 验证 9 物品配置是否生效 ---
    # 这一步是为了确认之前的 command_cfg.py 和 NewOrderCommandCfg.py 修改是否被正确加载
    try:
        # 尝试获取物品列表
        objects_list = env_cfg.commands.order_info.objects
        num_objects = len(objects_list)
        
        print("-" * 50)
        print(f"[INFO] Configuration Check:")
        print(f"       Target Objects Count: {num_objects}")
        print(f"       Object Names: {objects_list[:3]} ... {objects_list[-1]}")
        
        if num_objects == 9:
            print(f"[SUCCESS] ✅ Detected 9 objects configuration.")
            print(f"          The task will distribute 9 items into 3 boxes (max 4 per box).")
        else:
            print(f"[WARNING] ⚠️  Expected 9 objects but found {num_objects}!")
            print(f"          Please check if 'command_cfg.py' and 'scene_cfg.py' are updated correctly.")
        print("-" * 50)
        
    except AttributeError:
        print("[WARNING] Could not verify object list structure in config.")

    # --- 启动评测器 ---
    print(f"[INFO] Starting VLA Evaluator with policy: {args_cli.policy}")
    evaluator = VLA_Evaluator(env_cfg=env_cfg, policy=args_cli.policy)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()