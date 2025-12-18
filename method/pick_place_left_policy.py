# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Left-hand heuristic pick-and-place policy for single_arm_sorting.

Goal:
- Use left arm to grasp the active target object.
- Move it to the target area and release.

Notes:
- Keeps right arm idle.
- Uses a simple waypoint FSM: approach -> descend -> grasp -> lift -> go target -> descend -> release -> lift.
- Assumes action layout (realman cfg): [left_arm(7), gripper(1), right_arm(7)] => total 15 dims.
"""

from __future__ import annotations

import os

# 先启动 Isaac Lab 应用，确保 carb/kit 可用
from isaaclab.app import AppLauncher


def _launch_app():
    # 允许通过环境变量控制：HEADLESS, ENABLE_CAMERAS
    headless = os.getenv("HEADLESS", "0") == "1"
    enable_cameras = os.getenv("ENABLE_CAMERAS", "0") == "1"

    # 直接使用 AppLauncher 默认参数（isaaclab.sh 会传入额外参数，AppLauncher 会处理）
    app_launcher = AppLauncher(
        headless=headless,
        enable_cameras=enable_cameras,
    )
    return app_launcher.app


# 启动 app
simulation_app = _launch_app()

import gymnasium as gym
import torch

from isaaclab_logistics_vla.tasks.single_arm_sorting.object_randomization import (
    get_active_object_pose_w,
)


class LeftPickPlacePolicy:
    """A lightweight heuristic controller using the left arm."""

    def __init__(
        self,
        approach_height: float = 0.10,
        place_height: float = 0.05,
        lift_height: float = 0.12,
        pos_gain: float = 0.5,
        max_step: float = 0.05,
        close_cmd: float = 0.0,
        open_cmd: float = 0.04,
        collision_reset_dist: float = 0.02,
    ):
        self.approach_height = approach_height
        self.place_height = place_height
        self.lift_height = lift_height
        self.pos_gain = pos_gain
        self.max_step = max_step
        self.close_cmd = close_cmd
        self.open_cmd = open_cmd
        self.collision_reset_dist = collision_reset_dist
        # FSM states
        self.state = "approach"

    def _clip_step(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.clamp(delta, -self.max_step, self.max_step)

    def __call__(self, env, obs):
        """Compute a single action."""
        device = env.unwrapped.device
        num_envs = env.unwrapped.scene.num_envs

        # get active object pose
        object_pos_w, object_quat_w = get_active_object_pose_w(env.unwrapped)
        # target area position
        target_pos_w = env.unwrapped.scene["target_area"].data.root_pos_w[:, :3]
        # current ee (left) pose
        ee_frame = env.unwrapped.scene["ee_frame"]
        ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # [N,3]

        # build action buffer: [left_arm(7), gripper(1), right_arm(7)] -> 15 dims
        action_dim = env.action_space.shape[1]
        action = torch.zeros((num_envs, action_dim), device=device)

        # simple finite-state machine
        # approach above object
        if self.state == "approach":
            goal = object_pos_w.clone()
            goal[:, 2] += self.approach_height
            delta = self.pos_gain * (goal - ee_pos_w)
            # 左臂位置增量（diff-IK 会将其转为关节角）+ 保持当前朝向
            action[:, :3] = self._clip_step(delta)
            action[:, 3:7] = 0.0  # 不显式控制姿态
            action[:, 7] = self.open_cmd  # gripper open
            if torch.all(torch.norm(goal - ee_pos_w, dim=1) < 0.02):
                self.state = "descend"

        elif self.state == "descend":
            goal = object_pos_w.clone()
            goal[:, 2] += self.place_height  # slight clearance
            delta = self.pos_gain * (goal - ee_pos_w)
            action[:, :3] = self._clip_step(delta)
            action[:, 3:7] = 0.0
            action[:, 7] = self.open_cmd
            if torch.all(torch.norm(goal - ee_pos_w, dim=1) < 0.01):
                self.state = "grasp"

        elif self.state == "grasp":
            action[:, 3:7] = 0.0
            action[:, 7] = self.close_cmd  # gripper close
            self.state = "lift"

        elif self.state == "lift":
            goal = ee_pos_w.clone()
            goal[:, 2] = object_pos_w[:, 2] + self.lift_height
            delta = self.pos_gain * (goal - ee_pos_w)
            action[:, :3] = self._clip_step(delta)
            action[:, 3:7] = 0.0
            action[:, 7] = self.close_cmd
            if torch.all(torch.norm(goal - ee_pos_w, dim=1) < 0.02):
                self.state = "go_target"

        elif self.state == "go_target":
            goal = target_pos_w.clone()
            goal[:, 2] += self.lift_height
            delta = self.pos_gain * (goal - ee_pos_w)
            action[:, :3] = self._clip_step(delta)
            action[:, 3:7] = 0.0
            action[:, 7] = self.close_cmd
            if torch.all(torch.norm(goal - ee_pos_w, dim=1) < 0.02):
                self.state = "place"

        elif self.state == "place":
            goal = target_pos_w.clone()
            goal[:, 2] += self.place_height
            delta = self.pos_gain * (goal - ee_pos_w)
            action[:, :3] = self._clip_step(delta)
            action[:, 3:7] = 0.0
            action[:, 7] = self.close_cmd
            if torch.all(torch.norm(goal - ee_pos_w, dim=1) < 0.01):
                self.state = "release"

        elif self.state == "release":
            action[:, 3:7] = 0.0
            action[:, 7] = self.open_cmd
            self.state = "done"

        else:  # done / idle
            action[:, :] = 0.0
            action[:, 7] = self.open_cmd

        # 明确将右臂动作置零，确保只用左手
        if action_dim > 8:
            action[:, 8:] = 0.0

        return action


def run_once(
    env_name: str = "Isaac-Logistics-SingleArmSorting-Realman-v0",
    max_steps: int = 500,
    num_envs: int = 1,
):
    # 建立环境配置
    try:
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(env_name, num_envs=num_envs)
    except ImportError:
        from isaaclab_logistics_vla.tasks.single_arm_sorting.config.realman import (
            RealmanSingleArmSortingEnvCfg,
        )

        env_cfg = RealmanSingleArmSortingEnvCfg()
        env_cfg.scene.num_envs = num_envs

    env = gym.make(env_name, cfg=env_cfg)
    policy = LeftPickPlacePolicy()

    obs, info = env.reset()
    step_count = 0
    while True:
        step_count += 1
        with torch.inference_mode():
            action = policy(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # 额外的“碰到目标物就重启”判定：
        # 当还处于 approach 阶段，如果末端执行器距离目标物过近，则认为碰撞，重置环境
        if policy.state == "approach":
            from isaaclab_logistics_vla.tasks.single_arm_sorting.object_randomization import (
                get_active_object_pose_w,
            )

            with torch.inference_mode():
                object_pos_w, _ = get_active_object_pose_w(env.unwrapped)
                ee_frame = env.unwrapped.scene["ee_frame"]
                ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
                dist = torch.norm(object_pos_w - ee_pos_w, dim=1)
                if torch.any(dist < policy.collision_reset_dist):
                    print("[INFO] Left arm touched object in approach phase, resetting episode.")
                    obs, info = env.reset()
                    policy.state = "approach"
                    step_count = 0
                    continue

        if terminated.any() or truncated.any():
            # 自动重置，持续运行
            obs, info = env.reset()
            step_count = 0
            continue

        if step_count >= max_steps:
            # 达到步数上限也重置，持续运行
            obs, info = env.reset()
            step_count = 0
            continue

    env.close()


if __name__ == "__main__":
    run_once()

