import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test framework for Isaac Lab logistics tasks.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default='cuda:0', help="Device to run simulation on.")
parser.add_argument("--interval_steps", type=int, default=50, help="Steps between each teleport action.")
parser.add_argument(
    "--max_episodes",
    type=int,
    default=20,
    help="Number of episodes to run PER environment.",
)
parser.add_argument("--asset_root_path", type=str, default="/home/wst/model_files/benchmark")
parser.add_argument("--task_scene_name", type=str, default="Spawn_ss_st_sparse_with_obstacles_EnvCfg")
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

if not os.path.exists(args_cli.asset_root_path):
    print(f"资产路径{args_cli.asset_root_path}未配置！请检查")
    exit()
else:
    print(f"Asset Root Path: {args_cli.asset_root_path}")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

# ============ Isaac Lab imports (must be after AppLauncher) ============
import torch

import isaaclab_tasks
import isaaclab_logistics_vla

#from isaaclab_logistics_vla.tasks.ss_st_series.sparse_scene.env_cfg import Spawn_ss_st_sparse_EnvCfg
from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv
from isaaclab_logistics_vla.evaluation.tester import Tester, TEST_SUITE
from isaaclab_logistics_vla.utils.register import register
register.auto_scan("isaaclab_logistics_vla.tasks")

def main():
    """测试主函数"""
    print("\n" + "=" * 60)
    print("           物流任务测试框架")
    print("=" * 60)
    print(f"环境数量: {args_cli.num_envs}")
    print(f"瞬移间隔: {args_cli.interval_steps} steps")
    print(f"每个环境目标episode数: {args_cli.max_episodes}")
    print("=" * 60 + "\n")
    
    # ============ 1. 创建环境 ============
    env_cfg = register.load_env_configs(f'{args_cli.task_scene_name}')()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # 延长episode时间，便于测试
    env_cfg.episode_length_s = 30  # 30秒
    
    env = VLAIsaacEnv(cfg=env_cfg)
    print(f"[Main] 环境创建完成，共 {env.num_envs} 个并行环境")
    
    # ============ 2. 创建测试器 ============
    tester = Tester(
        env=env,
        command_term_name="order_info",
        interval_steps=args_cli.interval_steps,
        test_suite=TEST_SUITE
    )
    
    # ============ 3. 初始化 ============
    obs, info = env.reset()
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    tester.reset(all_env_ids)
    
    # ============ 4. 主循环 ============
    per_env_target_episodes = args_cli.max_episodes
    per_env_episode_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    step_count = 0
    
    # 生成空动作（测试时不需要实际控制机器人）
    zero_actions = torch.zeros((env.num_envs, 17), device=env.device)
    
    print("\n[Main] 开始测试循环...")
    
    try:
        # 只要还有环境未达到目标轮数，就继续运行
        while (per_env_episode_count < per_env_target_episodes).any():
            with torch.inference_mode():
                # ========== 1. 先执行瞬移（在物理step之前）==========
                tester.step()
                
                # ========== 2. 执行物理仿真（瞬移会在这里生效）==========
                obs, rew, terminated, truncated, info = env.step(zero_actions)
                step_count += 1
                
                # ========== 3. 检查 Tester 认为已完成测试的环境 ==========
                tester_completed_envs = tester.get_completed_envs()
                
                # ========== 4. 合并环境自动重置和测试完成的环境 ==========
                reset_mask = terminated | truncated
                env_reset_ids = torch.where(reset_mask)[0]
                
                # 合并两种重置触发（去重）
                all_reset_ids = torch.unique(torch.cat([env_reset_ids, tester_completed_envs]))
                
                # ========== 5. 过滤：只重置未达到目标轮数的环境 ==========
                valid_reset_mask = per_env_episode_count[all_reset_ids] < per_env_target_episodes
                valid_reset_ids = all_reset_ids[valid_reset_mask]
                
                if len(valid_reset_ids) > 0:
                    print(f"\n[Main] Step {step_count}: {len(valid_reset_ids)} 个环境即将重置")
                    
                    # 1. 检查指标（在环境重置前）
                    tester.check(valid_reset_ids)
                    
                    # 2. 记录统计数据（在环境重置前，此时metrics还有值）
                    tester.record_stats(valid_reset_ids)
                    
                    # 3. 对于测试完成但环境没有自动重置的环境，强制执行重置
                    force_reset_ids = []
                    for env_id in tester_completed_envs:
                        env_id_item = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
                        if env_id_item in valid_reset_ids.tolist() and env_id_item not in env_reset_ids.tolist():
                            force_reset_ids.append(env_id_item)
                    
                    if len(force_reset_ids) > 0:
                        force_reset_tensor = torch.tensor(force_reset_ids, dtype=torch.long, device=env.device)
                        env.unwrapped._reset_idx(force_reset_tensor)
                    
                    # 4. 重置测试器状态（在环境重置后）
                    tester.reset(valid_reset_ids)
                    
                    # 5. 更新每个环境的 episode 计数
                    per_env_episode_count[valid_reset_ids] += 1
                    
                    # 打印进度
                    total_completed = per_env_episode_count.sum().item()
                    print(f"[Main] 进度: {per_env_episode_count.tolist()} (总计 {total_completed}/{per_env_target_episodes * env.num_envs})\n")
                
                # 定期输出状态
                if step_count % 200 == 0:
                    total_completed = per_env_episode_count.sum().item()
                    print(f"[Main] Step {step_count} | 各环境episode数: {per_env_episode_count.tolist()} | 总计: {total_completed}")
                    
    except KeyboardInterrupt:
        print("\n[Main] 收到中断信号，停止测试...")
    
    # ============ 5. 输出总结 ============
    tester.print_summary()
    
    total_episodes = per_env_episode_count.sum().item()
    print(f"\n[Main] 测试完成！共运行 {step_count} 步")
    print(f"各环境episode数: {per_env_episode_count.tolist()} (目标: 每个环境 {per_env_target_episodes} 轮)")
    print(f"总计: {total_episodes} 个episodes")
    
    # 关闭环境
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
