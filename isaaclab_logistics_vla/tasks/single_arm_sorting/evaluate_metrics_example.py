"""示例脚本：如何评估 single_arm_sorting 任务的指标

这个脚本展示了如何运行环境并收集评估指标。
"""

import gymnasium as gym
import torch
import numpy as np
from collections import defaultdict


def evaluate_metrics(
    env_name: str = "Isaac-Logistics-SingleArmSorting-Franka-v0",
    num_episodes: int = 100,
    max_steps_per_episode: int = 500,
    policy=None,
):
    """评估环境并收集指标。
    
    Args:
        env_name: Gym 环境名称
        num_episodes: 评估的 episode 数量
        max_steps_per_episode: 每个 episode 的最大步数
        policy: 策略函数，输入 obs，输出 action。如果为 None，则使用随机策略
        
    Returns:
        包含所有指标统计信息的字典
    """
    # 创建环境
    env = gym.make(env_name)
    
    # 如果没有提供策略，使用随机策略
    if policy is None:
        def random_policy(obs):
            return env.action_space.sample()
        policy = random_policy
    
    # 收集所有指标
    all_metrics = {
        "grasping_success_rate": [],
        "intent_accuracy": [],
        "task_success_rate": [],
    }
    
    # 运行多个 episode
    episode_count = 0
    step_count = 0
    
    obs, info = env.reset()
    
    while episode_count < num_episodes:
        # 选择动作
        if isinstance(obs, torch.Tensor):
            action = policy(obs)
        else:
            action = policy(obs)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        # 收集指标（当 episode 结束时）
        if "log" in info:
            log = info["log"]
            
            # 收集完成的 episode 的指标
            if "grasping_success_rate" in log and log["grasping_success_rate"]:
                all_metrics["grasping_success_rate"].extend(log["grasping_success_rate"])
            
            if "intent_accuracy" in log and log["intent_accuracy"]:
                all_metrics["intent_accuracy"].extend(log["intent_accuracy"])
            
            if "task_success_rate" in log and log["task_success_rate"]:
                all_metrics["task_success_rate"].extend(log["task_success_rate"])
        
        # 检查是否需要重置
        if terminated.any() or truncated.any() or step_count >= max_steps_per_episode:
            episode_count += 1
            step_count = 0
            obs, info = env.reset()
            
            # 打印进度
            if episode_count % 10 == 0:
                print(f"Completed {episode_count}/{num_episodes} episodes...")
    
    env.close()
    
    # 计算统计信息
    stats = {}
    for metric_name, values in all_metrics.items():
        if values:
            stats[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }
        else:
            stats[metric_name] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
    
    return stats


def print_metrics(stats: dict):
    """打印指标统计信息。
    
    Args:
        stats: 从 evaluate_metrics() 返回的统计字典
    """
    print("\n" + "="*60)
    print("评估指标统计")
    print("="*60)
    
    metric_display_names = {
        "grasping_success_rate": "抓取成功率 (Grasping Success Rate)",
        "intent_accuracy": "意图正确率 (Intent Accuracy)",
        "task_success_rate": "任务成功率 (Task Success Rate)",
    }
    
    for metric_name, display_name in metric_display_names.items():
        if metric_name in stats:
            s = stats[metric_name]
            print(f"\n{display_name}:")
            print(f"  平均值 (Mean): {s['mean']:.2%}")
            print(f"  标准差 (Std):  {s['std']:.2%}")
            print(f"  最小值 (Min):  {s['min']:.2%}")
            print(f"  最大值 (Max):  {s['max']:.2%}")
            print(f"  样本数 (Count): {s['count']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # 示例：使用随机策略评估
    print("开始评估指标（使用随机策略）...")
    print("注意：随机策略的性能会很差，这只是演示如何收集指标。")
    
    stats = evaluate_metrics(
        env_name="Isaac-Logistics-SingleArmSorting-Franka-v0",
        num_episodes=50,  # 评估 50 个 episode
        max_steps_per_episode=500,
    )
    
    print_metrics(stats)
    
    print("\n提示：要使用训练好的策略，请将策略函数传递给 evaluate_metrics()。")

