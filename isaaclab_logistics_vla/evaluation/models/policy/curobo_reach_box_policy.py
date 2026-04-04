from __future__ import annotations

import os
import time
from typing import Dict, Optional, Sequence, Union

import torch
import numpy as np

from isaaclab_logistics_vla.evaluation.models.policy.base import Policy
from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac
from isaaclab_logistics_vla.evaluation.robot_registry import RobotEvalConfig
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_logistics_vla.evaluation.curobo.planner import (
    WorldMode,
    CuroboPlanner,
)


class CuroboReachBoxPolicy(Policy):
    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        robot_eval_cfg: RobotEvalConfig,
        horizon: int = 200,
        curobo_device: Optional[Union[torch.device, str]] = None,
        use_mesh_obstacles: Optional[bool] = None,
        platform_joint_index: Optional[int] = None,
    ):
        self._device = device
        # Curobo 可单独指定 GPU（如 cuda:1），避免与 Isaac Lab 争抢显存
        _curobo_dev = curobo_device or os.environ.get("CUROBO_DEVICE")
        if isinstance(_curobo_dev, str):
            self._curobo_device = torch.device(_curobo_dev)
        elif isinstance(_curobo_dev, torch.device):
            self._curobo_device = _curobo_dev
        else:
            self._curobo_device = device
        self._action_dim = action_dim
        self._robot_eval_cfg = robot_eval_cfg
        self._horizon = horizon
        self._use_mesh_obstacles = (
            use_mesh_obstacles
            if use_mesh_obstacles is not None
            else (os.environ.get("CUROBO_USE_MESH_OBSTACLES", "").lower() in ("1", "true", "yes"))
        )

        self._curobo_planner: Optional[CuroboPlanner] = None
        self._trajectory: Optional[torch.Tensor] = None
        self._step_counter = 0
        self._platform_joint_index = platform_joint_index

        # ==========================================
        # 1. 坐标系同步：与仿真一致，按当前任务场景来。
        #    臂基 = root + R_root @ (arm_base_offset_in_root + [0,0,platform])，见 robot_registry。
        #    有 root_pos_w/root_quat_w 时用仿真当前状态计算臂基；无时用 fallback（与 sparse_scene 初始一致）。
        # ==========================================
        # 中间箱子 s_box_2 (1.025, 1.336, 0.725)，z 提到 1.0 悬空在箱子上方避免撞台面
        self.TARGET_BOX_WORLD_POS = np.array([1.025, 1.45, 0.9])
        # fallback 与 sparse_scene 一致：root=(0.96781,2.1,0.216) identity rot, platform=0.65 → 臂基
        # 0.96781, 2.1-0.11663, 0.216+0.271+0.65 = (0.96781, 1.98337, 1.137)
        self.ROBOT_WORLD_POS = np.array([0.96781, 1.88337, 1.137])
        # 8 个方向：夹爪从哪往哪抓（approach 接近方向）
        # r,p,y 欧拉角：pitch 控制上下，yaw 控制前后左右
        self.GRASP_POSE_CANDIDATES = [
            (0, 180, 0),    # 从上往下
            (0, 0, 0),      # 从下往上
            (0, 90, 0),     # 从前往后
            (0, 90, 180),   # 从后往前
            (0, 90, 90),    # 从左往右
            (0, 90, -90),   # 从右往左
        ]
        self._init_curobo_motion_gen()

    @property
    def name(self) -> str:
        return "curobo_reach_box_motiongen"

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        self._trajectory = None
        self._step_counter = 0

    def predict(self, obs: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        num_envs = 1
        qpos = None
        if obs and "robot_state" in obs:
            rs = obs["robot_state"]
            if isinstance(rs, dict) and "qpos" in rs and isinstance(rs["qpos"], torch.Tensor):
                qpos = rs["qpos"].to(self._device)
                num_envs = qpos.shape[0]

        actions = torch.zeros((num_envs, self._action_dim), device=self._device)

        if self._trajectory is None and qpos is not None and self._curobo_planner is not None:
            q_start = qpos[0].detach().clone()
            arm_base_pos = None
            arm_base_quat = None
            if isinstance(rs, dict):
                arm_offset = getattr(self._robot_eval_cfg, "arm_base_offset_in_root", None)
                if (
                    arm_offset is not None
                    and self._platform_joint_index is not None
                    and "root_pos_w" in rs
                    and "root_quat_w" in rs
                ):
                    root_pos = rs["root_pos_w"][0:1].to(self._device)
                    root_quat = rs["root_quat_w"][0:1].to(self._device)
                    platform_val = qpos[0, self._platform_joint_index].to(self._device)
                    offset = torch.tensor(
                        [arm_offset[0], arm_offset[1], arm_offset[2] + platform_val.item()],
                        device=self._device,
                        dtype=root_pos.dtype,
                    ).reshape(1, 3)
                    arm_base_pos_w, _ = combine_frame_transforms(root_pos, root_quat, offset)
                    arm_base_pos = arm_base_pos_w[0].detach().cpu().numpy()
                    arm_base_quat = root_quat[0].detach().cpu().numpy()
            with torch.enable_grad():
                self._trajectory = self._plan_motion(q_start, arm_base_pos=arm_base_pos, arm_base_quat=arm_base_quat)

        if self._trajectory is not None:
            idx = min(self._step_counter, self._trajectory.shape[0] - 1)
            q_target = self._trajectory[idx]
            dim = min(self._trajectory.shape[1], self._action_dim)
            actions[:, :dim] = q_target[:dim]

        self._step_counter += 1
        return actions

    def _init_curobo_motion_gen(self) -> None:
        if not all(
            [
                self._robot_eval_cfg.curobo_yml_name,
                self._robot_eval_cfg.curobo_asset_folder,
                self._robot_eval_cfg.curobo_urdf_name,
            ]
        ):
            return
        try:
            if self._curobo_device != self._device:
                print(f"[{self.name}] MotionGen 使用单独 GPU: {self._curobo_device}")

            use_hollow_box = os.environ.get("CUROBO_HOLLOW_BOX", "").lower() in ("1", "true", "yes")
            if self._use_mesh_obstacles and use_hollow_box:
                world_mode: WorldMode = "boxes_hollow"
            elif self._use_mesh_obstacles:
                world_mode = "boxes_mesh"
            else:
                world_mode = "boxes_cuboid"

            self._curobo_planner = CuroboPlanner(
                self._robot_eval_cfg,
                curobo_device=self._curobo_device,
                world_mode=world_mode,
                logger_name=self.name,
            )
        except Exception as e:
            print(f"[{self.name}] 初始化失败: {e}")
            import traceback

            traceback.print_exc()

    def _transform_world_to_robot_frame(
        self,
        target_world_pos: np.ndarray,
        arm_base_pos: Optional[np.ndarray] = None,
        arm_base_quat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """世界系 → 臂基系。Isaac 与 Realman 轴一致（+X 左、-Y 前），纯减法。"""
        base_pos = (
            np.asarray(arm_base_pos, dtype=np.float64).flatten()
            if arm_base_pos is not None
            else self.ROBOT_WORLD_POS
        )
        p = np.asarray(target_world_pos, dtype=np.float64).flatten()
        rx = p[0] - base_pos[0]
        ry = p[1] - base_pos[1]
        rz = p[2] - base_pos[2]
        return np.array([rx, ry, rz], dtype=np.float32)

    def _plan_motion(
        self,
        q_start: torch.Tensor,
        arm_base_pos: Optional[np.ndarray] = None,
        arm_base_quat: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        使用 MotionGen 生成避障轨迹
        """
        # 1. 关节索引 (左臂: 偶数位)
        left_arm_indices = [0, 2, 4, 6, 8, 10, 12] 
        dof = len(left_arm_indices)
        
        # 提取起始关节（移至 Curobo 设备）
        q_start_left = q_start[left_arm_indices].detach().clone().to(
            dtype=torch.float32, device=self._curobo_device
        )

        # 2. 目标设定 (世界 -> 臂基系，纯减法)
        target_world = self.TARGET_BOX_WORLD_POS.copy()
        target_local_np = self._transform_world_to_robot_frame(
            target_world, arm_base_pos=arm_base_pos, arm_base_quat=arm_base_quat
        )

        print(f"[{self.name}] 🎯 MotionGen 目标: {target_local_np}")

        # 3. 构建 MotionGen 请求，依次尝试多种抓取姿态直至成功
        target_pos_vec = torch.tensor(
            target_local_np, device=self._curobo_device, dtype=torch.float32
        )

        plan_result: Optional[dict] = None
        t_plan_start = time.perf_counter()
        for r, p, y in self.GRASP_POSE_CANDIDATES:
            _qw, _qx, _qy, _qz = euler_to_quat_isaac(r=r, p=p, y=y, return_tensor=False)
            target_quat_vec = torch.tensor(
                [_qw, _qx, _qy, _qz], device=self._curobo_device, dtype=torch.float32
            )
            # 诊断：传给规划器的最终数值（仅第一次姿态打印）
            if (r, p, y) == self.GRASP_POSE_CANDIDATES[0]:
                _pos = target_pos_vec.detach().cpu().tolist()
                _quat = target_quat_vec.detach().cpu().tolist()
                print(f"[{self.name}] 传给 CuroboPlanner 的 goal_pose: pos={_pos}, quat(wxyz)={_quat}")
            t_single = time.perf_counter()
            assert self._curobo_planner is not None
            plan_result = self._curobo_planner.plan_ee(
                q_start=q_start_left,
                target_pos_b=target_pos_vec,
                target_quat_b=target_quat_vec,
                max_attempts=10,
                timeout=2.0,
                enable_graph=True,
                enable_opt=False,
            )
            elapsed = (time.perf_counter() - t_single) * 1000
            if plan_result["status"] == "Success":
                total_ms = (time.perf_counter() - t_plan_start) * 1000
                print(f"[{self.name}] 使用姿态 (r={r}, p={p}, y={y})° 规划成功 | 本次: {elapsed:.0f}ms | 累计: {total_ms:.0f}ms")
                break
            else:
                detail = plan_result.get("detail", "?")
                print(f"[{self.name}] 姿态 (r={r}, p={p}, y={y})° 失败 {detail} | {elapsed:.0f}ms")

        # 5. 处理结果
        traj = torch.zeros((self._horizon, self._action_dim), device=self._device)
        base_q = q_start.detach().clone()

        if plan_result is not None and plan_result["status"] == "Success" and plan_result["position"] is not None:
            pos_np = plan_result["position"]
            print(f"[{self.name}] ✅ CuroboPlanner 规划成功! 路径点数: {pos_np.shape[0]}")

            raw_path = torch.from_numpy(pos_np).to(self._device)
            raw_steps = raw_path.shape[0]
            
            # 重采样
            indices = torch.linspace(0, raw_steps - 1, self._horizon, device=self._device)
            
            for t in range(self._horizon):
                idx_float = indices[t]
                idx_floor = int(idx_float)
                idx_ceil = min(idx_floor + 1, raw_steps - 1)
                alpha = idx_float - idx_floor
                
                q_t_left = (1 - alpha) * raw_path[idx_floor] + alpha * raw_path[idx_ceil]
                
                action_t = base_q.clone()
                for i, joint_idx in enumerate(left_arm_indices):
                    if joint_idx < self._action_dim:
                        action_t[joint_idx] = q_t_left[i]
                
                traj[t, : min(self._action_dim, action_t.numel())] = action_t[: self._action_dim]
                
        else:
            total_ms = (time.perf_counter() - t_plan_start) * 1000
            detail = plan_result.get("detail", "?") if plan_result else "no_result"
            print(f"[{self.name}] ❌ CuroboPlanner 规划失败: {detail} | 总耗时: {total_ms:.0f}ms")
            # 失败保持不动
            for t in range(self._horizon):
                traj[t] = base_q[:self._action_dim]

        return traj