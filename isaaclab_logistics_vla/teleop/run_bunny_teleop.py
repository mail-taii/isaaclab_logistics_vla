"""
Bunny 遥操作核心：将 Bunny 左右臂 qpos 映射为 realman 动作并驱动 env。
由 scripts/run_bunny_teleop.py 在 AppLauncher 启动后调用。
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from isaaclab_logistics_vla.teleop.bunny_client import BunnyQposListener
from isaaclab_logistics_vla.utils.register import register


def bunny_qpos_to_realman_action(
    left_arm_7: np.ndarray,
    right_arm_7: np.ndarray,
    left_gripper: float = 0.5,
    right_gripper: float = 0.5,
    platform: float = 0.0,
) -> np.ndarray:
    """
    Realman 动作顺序（与 RealmanFrankaEE_ActionsCfg 一致）：
    arm_joints 14 (l1..l7, r1..r7) + left_gripper 1 + right_gripper 1 + platform 1 = 17
    """
    action = np.zeros(17, dtype=np.float32)
    action[0:7] = left_arm_7
    action[7:14] = right_arm_7
    action[14] = left_gripper
    action[15] = right_gripper
    action[16] = platform
    return action


def run_bunny_teleop(
    task_scene_name: str,
    left_topic: str = "/bunny_teleop/left_qpos",
    right_topic: str = "/bunny_teleop/right_qpos",
    control_hz: float = 60.0,
    num_envs: Optional[int] = None,
    sim_device: Optional[str] = None,
) -> None:
    """
    在已启动的 Isaac App 中创建 env、订阅 Bunny qpos、运行遥操作循环。
    调用前需已设置 ASSET_ROOT_PATH 并完成 register.auto_scan。
    """
    from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv

    env_cfg = register.load_env_configs(task_scene_name)()
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = sim_device or "cuda:0"

    env = VLAIsaacEnv(cfg=env_cfg)
    action_dim = env.unwrapped.action_manager.total_action_dim
    device = env.device

    listener = BunnyQposListener(
        left_topic=left_topic,
        right_topic=right_topic,
        arm_dof=7,
    )
    listener.start()
    print(f"[Bunny Teleop] 已订阅 {left_topic}, {right_topic}，控制频率 {control_hz} Hz")

    env.reset()
    dt = 1.0 / control_hz
    last_left = np.zeros(7, dtype=np.float64)
    last_right = np.zeros(7, dtype=np.float64)
    has_received = False

    try:
        while True:
            t0 = time.perf_counter()
            left, right = listener.get_latest()
            if left is not None and right is not None:
                last_left = left
                last_right = right
                has_received = True
            if not has_received:
                left = last_left.copy()
                right = last_right.copy()
            else:
                left = last_left
                right = last_right

            action_np = bunny_qpos_to_realman_action(left, right)
            action = torch.from_numpy(action_np).unsqueeze(0).to(device=device, dtype=torch.float32)
            if action.shape[1] != action_dim:
                action = action[:, :action_dim]
            if action.shape[1] < action_dim:
                pad = torch.zeros(1, action_dim - action.shape[1], device=device, dtype=torch.float32)
                action = torch.cat([action, pad], dim=1)
            env.step(action)

            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[Bunny Teleop] 用户中断")
    finally:
        listener.stop()
        env.close()
