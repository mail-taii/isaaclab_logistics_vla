# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single arm sorting environment with metrics tracking.

This environment extends ManagerBasedRLEnv to track evaluation metrics:
- Grasping Success Rate
- Intent Accuracy  
- Task Success Rate
"""

from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg

from . import mdp


class SingleArmSortingEnv(ManagerBasedRLEnv):
    """Single arm sorting environment with metrics tracking.
    
    This environment extends ManagerBasedRLEnv to automatically track and log
    benchmark evaluation metrics in the extras dictionary.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, **kwargs):
        """Initialize the environment with metrics tracking.
        
        Args:
            cfg: The environment configuration.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(cfg, **kwargs)
        # Track当前每个环境的“焦点”物体索引（支持集合形式的物体随机化）
        self._active_object_indices = torch.zeros(self.scene.num_envs, dtype=torch.long, device=self.device)
        
        # 确保USD场景中的所有物体都有碰撞检测
        # 修复物体穿透USD场景的问题
        # 关键：机器人有碰撞（因为是Articulation），地面有碰撞（因为是GroundPlane）
        # 但USD文件中的静态物体可能没有RigidBodyAPI，需要手动添加
        try:
            import isaaclab.sim.schemas as schemas
            import isaaclab.sim.schemas.schemas_cfg as schemas_cfg
            from pxr import UsdPhysics
            
            # 为USD场景中的所有物体定义碰撞属性
            # 这会递归应用到所有子prim
            for env_idx in range(self.scene.num_envs):
                env_ns = f"/World/envs/env_{env_idx}"
                base_scene_path = f"{env_ns}/BaseScene"
                
                # 检查prim是否存在
                prim = self.sim.stage.GetPrimAtPath(base_scene_path)
                if prim and prim.IsValid():
                    # 递归为所有子prim定义碰撞属性
                    def ensure_collision_for_prim(prim_path: str):
                        """递归为prim及其所有子prim定义碰撞属性"""
                        try:
                            prim = self.sim.stage.GetPrimAtPath(prim_path)
                            if not prim or not prim.IsValid():
                                return
                            
                            # 跳过已经有关节的prim（如机器人）
                            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                                return
                            
                            # 如果有mesh或geometry，确保有碰撞
                            has_geometry = False
                            for child in prim.GetChildren():
                                if child.GetTypeName() in ["Mesh", "Cylinder", "Sphere", "Cube", "Capsule"]:
                                    has_geometry = True
                                    break
                            
                            # 如果prim有几何体，确保有刚体和碰撞属性
                            if has_geometry or prim.GetTypeName() in ["Mesh", "Cylinder", "Sphere", "Cube", "Capsule"]:
                                # 定义刚体属性（如果还没有）
                                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                                    schemas.define_rigid_body_properties(
                                        prim_path,
                                        schemas_cfg.RigidBodyPropertiesCfg(
                                            kinematic_enabled=False,
                                            rigid_body_enabled=True,
                                            solver_position_iteration_count=32,  # 提高碰撞精度
                                            solver_velocity_iteration_count=1,
                                        ),
                                        stage=self.sim.stage,
                                    )
                                
                                # 定义碰撞属性（如果还没有）
                                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                                    schemas.define_collision_properties(
                                        prim_path,
                                        schemas_cfg.CollisionPropertiesCfg(
                                            collision_enabled=True,
                                            contact_offset=0.005,  # 接触偏移量，提高碰撞检测敏感度
                                            rest_offset=0.0,
                                        ),
                                        stage=self.sim.stage,
                                    )
                            
                            # 递归处理子prim
                            for child in prim.GetChildren():
                                ensure_collision_for_prim(str(child.GetPath()))
                        except Exception as e:
                            # 忽略单个prim的错误，继续处理其他prim
                            pass
                    
                    ensure_collision_for_prim(base_scene_path)
        except Exception as e:
            # 如果修改失败，记录警告但不中断初始化
            print(f"[WARNING] Failed to ensure collision for USD scene: {e}")
            import traceback
            traceback.print_exc()
        
        # 强制设置 platform_joint 的位置，确保身体部分升高
        # 这可以确保即使初始状态设置没有生效，也能在环境初始化后强制设置
        # 注意：目标位置应该与初始配置中的位置一致，避免机器人移动
        try:
            robot = self.scene["robot"]
            if "platform_joint" in robot.joint_names:
                platform_joint_idx = robot.joint_names.index("platform_joint")
                # 使用与初始配置一致的位置 (0.9m from config)
                # 这样机器人就不会在初始化时移动
                target_platform_pos = torch.full((self.scene.num_envs, 1), 0.9, device=self.device)
                joint_ids = torch.tensor([platform_joint_idx], device=self.device)
                # 设置目标位置和当前位置，确保一致
                if hasattr(robot, "set_joint_position_target"):
                    robot.set_joint_position_target(target_platform_pos, joint_ids=joint_ids)
                if hasattr(robot, "set_joint_position"):
                    robot.set_joint_position(target_platform_pos, joint_ids=joint_ids)
                print(f"[INFO] 设置 platform_joint 目标位置为 0.9m (索引: {platform_joint_idx})，与初始配置一致")
            else:
                print(f"[WARNING] platform_joint 不在关节列表中，可用关节: {robot.joint_names[:10]}...")
        except Exception as e:
            print(f"[WARNING] 无法设置 platform_joint 位置: {e}")
            import traceback
            traceback.print_exc()
        
        # Initialize metric tracking buffers
        # These track the best (or final) state achieved during each episode
        num_envs = self.scene.num_envs
        self._was_grasped_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._intent_correct_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._task_completed_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        
        # 动作噪声参数（用于末端目标的轻微扰动）
        self.action_noise_std = 0.01  # 末端目标轻微随机性，保持轨迹自然
        

    def _update_metrics(self):
        """Update metric tracking buffers based on current state."""
        # Update grasping success: track if object was ever grasped during episode
        is_grasped = mdp.is_object_grasped(
            self,
            threshold=0.02,
            object_cfg=SceneEntityCfg("object"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
        )
        self._was_grasped_buf = self._was_grasped_buf | is_grasped
        
        # Update intent correctness: track if robot followed correct intent
        intent_correct = mdp.is_intent_correct(
            self,
            object_cfg=SceneEntityCfg("object"),
            source_cfg=SceneEntityCfg("source_area"),
            target_cfg=SceneEntityCfg("target_area"),
        )
        self._intent_correct_buf = self._intent_correct_buf | intent_correct
        
        # Update task completion: track if task was ever completed during episode
        task_completed = mdp.is_task_completed(
            self,
            threshold=0.05,
            min_height=0.05,
            object_cfg=SceneEntityCfg("object"),
            target_cfg=SceneEntityCfg("target_area"),
        )
        self._task_completed_buf = self._task_completed_buf | task_completed

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment and update metrics.
        
        Args:
            action: Actions to apply. Should be in the format expected by the action manager.
                   For Differential IK actions, this should be [x, y, z, qw, qx, qy, qz, ...] for each arm.
            
        Returns:
            A tuple containing observations, rewards, terminated, truncated, and extras.
            The extras dictionary includes the metrics.
        """
        # 直接使用传入的action，不再硬编码到source_area
        # 这样外部策略（如测试脚本或训练的策略）可以完全控制机器人
        # 如果action维度不匹配，进行适当的填充或截断
        num_envs = action.shape[0]
        total_dim = self.action_manager.total_action_dim
        
        if action.shape[1] < total_dim:
            # 如果action维度不足，用0填充
            full_action = torch.zeros((num_envs, total_dim), device=self.device, dtype=action.dtype)
            full_action[:, :action.shape[1]] = action
            action = full_action
        elif action.shape[1] > total_dim:
            # 如果action维度过多，截断
            action = action[:, :total_dim]
        
        # Step the environment
        obs, reward, terminated, truncated, extras = super().step(action)
        
        # Update metrics tracking
        self._update_metrics()
        
        # Record metrics in extras
        if "log" not in extras:
            extras["log"] = {}
        
        # Compute metrics for completed episodes
        # Only log metrics when an episode terminates
        reset_mask = terminated | truncated
        
        if reset_mask.any():
            # Compute metrics for reset environments
            grasping_success = self._was_grasped_buf[reset_mask].float()
            intent_accuracy = self._intent_correct_buf[reset_mask].float()
            task_success = self._task_completed_buf[reset_mask].float()
            
            # Store in extras (accumulated over all resets in this step)
            if "grasping_success_rate" not in extras["log"]:
                extras["log"]["grasping_success_rate"] = []
            if "intent_accuracy" not in extras["log"]:
                extras["log"]["intent_accuracy"] = []
            if "task_success_rate" not in extras["log"]:
                extras["log"]["task_success_rate"] = []
            
            extras["log"]["grasping_success_rate"].extend(grasping_success.cpu().tolist())
            extras["log"]["intent_accuracy"].extend(intent_accuracy.cpu().tolist())
            extras["log"]["task_success_rate"].extend(task_success.cpu().tolist())
            
            # Reset metric buffers for reset environments
            self._was_grasped_buf[reset_mask] = False
            self._intent_correct_buf[reset_mask] = False
            self._task_completed_buf[reset_mask] = False
        
        # Also log current metric values for all environments (for monitoring)
        extras["log"]["metrics"] = {
            "grasping_success": self._was_grasped_buf.float().mean().item(),
            "intent_correct": self._intent_correct_buf.float().mean().item(),
            "task_completed": self._task_completed_buf.float().mean().item(),
        }
        
        return obs, reward, terminated, truncated, extras

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and metrics.
        
        Args:
            seed: Random seed for reset.
            options: Additional options for reset.
            
        Returns:
            A tuple containing initial observations and extras.
        """
        obs, extras = super().reset(seed=seed, options=options)
        
        # Reset metric buffers
        self._was_grasped_buf.fill_(False)
        self._intent_correct_buf.fill_(False)
        self._task_completed_buf.fill_(False)
        if hasattr(self, "_active_object_indices"):
            self._active_object_indices.zero_()
        
        return obs, extras

