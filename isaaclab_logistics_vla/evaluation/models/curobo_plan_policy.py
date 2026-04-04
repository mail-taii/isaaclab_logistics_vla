# Copyright (c) 2025, Logistics VLA extension contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""
在真实 Isaac Lab 任务场景中使用 ``utils.curobo_planner.CuroboPlanner`` 的示例策略。

典型启动（仓库根目录，需已配置 ``ASSET_ROOT_PATH`` 等）::

    ./isaaclab.sh -p /path/to/isaaclab_logistics_vla/scripts/evaluate_vla.py \\
        --policy curobo_plan --task_scene_name Spawn_ms_st_dense_EnvCfg --num_envs 1

说明:
    - 默认仅对 **env 0** 做规划，并将同一关节目标广播到所有并行环境（便于多 env 可视化）。
    - 目标位姿在 **机器人根坐标系** 下取为「当前左右 TCP 位置 + 固定增量」；默认增量较小，便于在 dense 场景先跑通。
    - **关节动作顺序**：cuRobo 轨迹为左 7 + 右 7，与 Isaac ``arm_joints`` 的 **l1,r1,l2,r2,...** 交错顺序不同，本策略会自动转换。
    - 若规划失败，保持当前臂关节目标（从仿真读回），避免全零动作导致乱甩。
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
import torch

from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_logistics_vla import ISAACLAB_LOGISTICS_VLA_EXT_DIR

# 送入 cuRobo 的起始关节：左 7 + 右 7（与 verify / URDF 臂链顺序一致）
_Q_CUROBO_NAMES: List[str] = [f"l_joint{i}" for i in range(1, 8)] + [f"r_joint{i}" for i in range(1, 8)]


def _q_block_lr_to_action_interleaved(q_block: np.ndarray) -> np.ndarray:
    """(14,) 左7+右7 → Isaac 动作前 14 维：l1,r1,l2,r2,…,l7,r7。"""
    q = np.asarray(q_block, dtype=np.float32).reshape(14)
    out = np.empty(14, dtype=np.float32)
    left, right = q[:7], q[7:]
    for i in range(7):
        out[2 * i] = left[i]
        out[2 * i + 1] = right[i]
    return out


def _default_urdf_path() -> str:
    p = os.path.join(
        ISAACLAB_LOGISTICS_VLA_EXT_DIR,
        "isaaclab_logistics_vla",
        "assets",
        "robots",
        "realman",
        "realman_franka_ee.urdf",
    )
    if os.path.isfile(p):
        return p
    alt = os.environ.get("CUROBO_PLAN_URDF", "")
    if alt and os.path.isfile(alt):
        return alt
    raise FileNotFoundError(
        f"未找到 Realman URDF: {p} 。请设置环境变量 CUROBO_PLAN_URDF 指向 realman_franka_ee.urdf"
    )


class CuRoboPlanPolicy:
    """使用封装后的 CuroboPlanner，在仿真中执行一条双臂关节空间轨迹。"""

    def __init__(
        self,
        device: str = "cuda:0",
        urdf_path: Optional[str] = None,
        interpolation_dt: float = 0.05,
        apply_robot_to_curobo_frame_transform: bool = False,
        # 机器人根坐标系下相对当前 TCP 的平移增量 (x,y,z)，米；默认偏小以提高可达率（dense 场景）
        left_goal_delta_base: tuple[float, float, float] = (0.07, 0.04, 0.06),
        right_goal_delta_base: tuple[float, float, float] = (0.07, -0.04, 0.06),
        gripper_open: float = 0.04,
        warmup_env_steps: int = 5,
        max_plan_attempts: int = 24,
        plan_timeout: float = 12.0,
        enable_graph_plan: bool = False,
    ):
        self.device = device
        self.urdf_path = urdf_path or _default_urdf_path()
        self.interpolation_dt = interpolation_dt
        self.apply_frame_transform = apply_robot_to_curobo_frame_transform
        self.left_goal_delta_base = np.array(left_goal_delta_base, dtype=np.float64)
        self.right_goal_delta_base = np.array(right_goal_delta_base, dtype=np.float64)
        self.gripper_open = gripper_open
        self.warmup_env_steps = warmup_env_steps
        self.max_plan_attempts = max_plan_attempts
        self.plan_timeout = plan_timeout
        self.enable_graph_plan = enable_graph_plan

        self._planner: Optional[Any] = None
        self._planned = False
        self._traj: Optional[np.ndarray] = None
        self._traj_idx = 0
        self._env_step_counter = 0
        self._sim_step_s = 0.02  # 由 env 覆盖：decimation * sim.dt
        self._fail_printed = False

    def reset(self) -> None:
        self._planned = False
        self._traj = None
        self._traj_idx = 0
        self._env_step_counter = 0
        self._fail_printed = False
        if self._planner is not None:
            self._planner.reset(reset_seed=True)

    def _lazy_planner(self):
        if self._planner is None:
            from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner

            print(
                f"[CuRoboPlanPolicy] 正在初始化 CuroboPlanner（首次较慢）… URDF={self.urdf_path} device={self.device}"
            )
            self._planner = CuroboPlanner(
                urdf_path=self.urdf_path,
                device=self.device,
                interpolation_dt=self.interpolation_dt,
                apply_robot_to_curobo_frame_transform=self.apply_frame_transform,
                use_cuda_graph=False,
            )
        return self._planner

    def _gather_arm_q(self, robot, env_ids: int | torch.Tensor = 0) -> np.ndarray:
        jnames = list(robot.data.joint_names)
        q = robot.data.joint_pos
        if isinstance(env_ids, int):
            row = q[env_ids]
        else:
            row = q[env_ids[0]]
        out = np.zeros(14, dtype=np.float32)
        for i, name in enumerate(_Q_CUROBO_NAMES):
            out[i] = float(row[jnames.index(name)].item())
        return out

    def __call__(self, env: Any) -> torch.Tensor:
        isaac_env = env.unwrapped
        device = isaac_env.device
        num_envs = isaac_env.scene.num_envs
        action_dim = isaac_env.action_manager.total_action_dim
        actions = torch.zeros((num_envs, action_dim), device=device)

        robot = isaac_env.scene["robot"]
        ee_frame = isaac_env.scene["ee_frame"]

        # 仿真每步物理时间（与 ManagerBasedRLEnvCfg.decimation * sim.dt 一致）
        dec = getattr(isaac_env.cfg, "decimation", 1)
        sim_dt = float(isaac_env.cfg.sim.dt)
        self._sim_step_s = dec * sim_dt

        try:
            names = list(ee_frame.data.target_frame_names)
            i_left = names.index("left_ee_tcp")
            i_right = names.index("right_ee_tcp")
        except (AttributeError, ValueError) as e:
            raise RuntimeError(
                "CuRoboPlanPolicy 需要场景里存在 ee_frame，且包含 left_ee_tcp / right_ee_tcp。"
            ) from e

        # 夹爪张开、平台保持当前（最后一维 platform）
        actions[:, 14] = self.gripper_open
        actions[:, 15] = self.gripper_open
        pj = list(robot.data.joint_names).index("platform_joint")
        actions[:, 16] = robot.data.joint_pos[:, pj]

        e = 0
        q_current = self._gather_arm_q(robot, e)
        q_act = _q_block_lr_to_action_interleaved(q_current)
        q_t = torch.as_tensor(q_act, device=device, dtype=torch.float32)
        actions[:, :14] = q_t.unsqueeze(0).expand(num_envs, -1)

        self._env_step_counter += 1
        if self._env_step_counter <= self.warmup_env_steps:
            return actions

        if self._traj is not None:
            t = self._traj
            q_cmd = _q_block_lr_to_action_interleaved(t[self._traj_idx])
            actions[:, :14] = torch.as_tensor(q_cmd, device=device, dtype=torch.float32).unsqueeze(0).expand(
                num_envs, -1
            )
            if self._traj_idx < t.shape[0] - 1:
                step_adv = max(1, int(round(self._sim_step_s / self.interpolation_dt)))
                self._traj_idx = min(self._traj_idx + step_adv, t.shape[0] - 1)
            return actions

        if self._planned:
            return actions

        self._planned = True

        root_pos = robot.data.root_pos_w[e : e + 1, :3]
        root_quat = robot.data.root_quat_w[e : e + 1, :4]
        ee_pos_w = ee_frame.data.target_pos_w[e : e + 1, :, :3]
        ee_quat_w = ee_frame.data.target_quat_w[e : e + 1, :, :4]

        pos_l_w = ee_pos_w[:, i_left, :]
        pos_r_w = ee_pos_w[:, i_right, :]
        quat_l_w = ee_quat_w[:, i_left, :]
        quat_r_w = ee_quat_w[:, i_right, :]

        pos_l_b, quat_l_b = subtract_frame_transforms(root_pos, root_quat, pos_l_w, quat_l_w)
        pos_r_b, quat_r_b = subtract_frame_transforms(root_pos, root_quat, pos_r_w, quat_r_w)

        goal_l = pos_l_b[0].detach().cpu().numpy() + self.left_goal_delta_base
        goal_r = pos_r_b[0].detach().cpu().numpy() + self.right_goal_delta_base
        quat_l = quat_l_b[0].detach().cpu().numpy()
        quat_r = quat_r_b[0].detach().cpu().numpy()

        goal_poses = {
            "left": {"position": goal_l, "quaternion": quat_l},
            "right": {"position": goal_r, "quaternion": quat_r},
        }

        planner = self._lazy_planner()
        planner.clear_world()

        out = planner.plan_dual(
            q_current,
            goal_poses,
            max_attempts=self.max_plan_attempts,
            timeout=self.plan_timeout,
            enable_graph=self.enable_graph_plan,
            enable_opt=True,
        )

        if out["status"] == "Success" and out["position"] is not None:
            self._traj = np.asarray(out["position"], dtype=np.float32)
            self._traj_idx = 0
            print(
                f"[CuRoboPlanPolicy] 规划成功: {self._traj.shape[0]} 个插值点, "
                f"interpolation_dt≈{self.interpolation_dt}s"
            )
            q0 = _q_block_lr_to_action_interleaved(self._traj[0])
            actions[:, :14] = torch.as_tensor(q0, device=device, dtype=torch.float32).unsqueeze(0).expand(
                num_envs, -1
            )
        else:
            if not self._fail_printed:
                print(
                    f"[CuRoboPlanPolicy] 规划未成功 (status={out.get('status')!r}, detail={out.get('detail')!r})，"
                    "将保持当前臂关节。可减小 left/right_goal_delta_base、或改 enable_graph_plan=True 试 graph+opt。"
                )
                self._fail_printed = True

        return actions
