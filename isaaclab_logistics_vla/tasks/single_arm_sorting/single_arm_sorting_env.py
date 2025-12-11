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
            action: Actions to apply.
            
        Returns:
            A tuple containing observations, rewards, terminated, truncated, and extras.
            The extras dictionary includes the metrics.
        """
        # 使用差分 IK：将末端执行器目标设为 source 区域位置（绝对位姿），并做小步长插值与限幅
        source_area = self.scene["source_area"]
        target_pos = source_area.data.root_pos_w
        target_rot = source_area.data.root_quat_w

        # 当前末端位姿（用于插值）
        try:
            ee_frame = self.scene["ee_frame"]
            current_pos = ee_frame.data.root_pos_w
            current_rot = ee_frame.data.root_quat_w
        except (KeyError, AttributeError):
            # 兜底：若未找到 ee_frame，则使用 target 作为当前值
            current_pos = target_pos.clone()
            current_rot = target_rot.clone()

        # 目标姿态添加轻微扰动，让轨迹更自然
        pos_noise = torch.randn_like(target_pos) * self.action_noise_std
        rot_noise = torch.randn_like(target_rot) * 0.0  # 暂不扰动旋转，保持对准

        # 位置限幅：以 source 为中心，xy 限制 ±0.25m，z 限制在 [0.2, 1.2]
        bbox_min = target_pos + torch.tensor([-0.25, -0.25, -0.2], device=self.device)
        bbox_max = target_pos + torch.tensor([0.25, 0.25, 0.3], device=self.device)
        clipped_pos = torch.clamp(target_pos + pos_noise, min=bbox_min, max=bbox_max)

        # 小步长插值，避免一次跃迁过大
        alpha = 0.05
        interp_pos = current_pos + alpha * (clipped_pos - current_pos)
        interp_rot = current_rot + alpha * (target_rot + rot_noise - current_rot)
        interp_rot = interp_rot / torch.linalg.norm(interp_rot, dim=-1, keepdim=True).clamp(min=1e-6)

        target_pose = torch.cat([interp_pos, interp_rot], dim=-1)

        # 将目标位姿填入完整动作向量（匹配 action_manager.total_action_dim）
        num_envs = target_pose.shape[0]
        total_dim = self.action_manager.total_action_dim
        full_action = torch.zeros((num_envs, total_dim), device=self.device, dtype=target_pose.dtype)
        # 假设前 7 维是末端位姿命令，其余（如夹爪等）保持为 0
        full_action[:, : target_pose.shape[1]] = target_pose
        action = full_action
        
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
        
        return obs, extras

