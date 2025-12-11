# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Your task name environment with metrics tracking.

This environment extends ManagerBasedRLEnv to track evaluation metrics:
- Grasping Success Rate
- Intent Accuracy  
- Task Success Rate

TODO: 根据你的任务调整指标名称和计算方式
"""

from __future__ import annotations

import torch
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg

from . import mdp


class YourTaskNameEnv(ManagerBasedRLEnv):
    """Your task name environment with metrics tracking.
    
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
        

    def _update_metrics(self):
        """Update metric tracking buffers based on current state.
        
        TODO: 根据你的任务实现指标更新逻辑
        """
        # Update grasping success: track if object was ever grasped during episode
        # TODO: 实现抓取检测
        # is_grasped = mdp.is_object_grasped(
        #     self,
        #     threshold=0.02,
        #     object_cfg=SceneEntityCfg("object"),
        #     ee_frame_cfg=SceneEntityCfg("ee_frame"),
        # )
        # self._was_grasped_buf = self._was_grasped_buf | is_grasped
        
        # Update intent correctness: track if robot followed correct intent
        # TODO: 实现意图正确性检测
        # intent_correct = mdp.is_intent_correct(
        #     self,
        #     object_cfg=SceneEntityCfg("object"),
        #     source_cfg=SceneEntityCfg("source_area"),
        #     target_cfg=SceneEntityCfg("target_area"),
        # )
        # self._intent_correct_buf = self._intent_correct_buf | intent_correct
        
        # Update task completion: track if task was ever completed during episode
        # TODO: 实现任务完成检测
        # task_completed = mdp.is_task_completed(
        #     self,
        #     threshold=0.05,
        #     min_height=0.05,
        #     object_cfg=SceneEntityCfg("object"),
        #     target_cfg=SceneEntityCfg("target_area"),
        # )
        # self._task_completed_buf = self._task_completed_buf | task_completed

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment and update metrics.
        
        Args:
            action: Actions to apply.
            
        Returns:
            A tuple containing observations, rewards, terminated, truncated, and extras.
            The extras dictionary includes the metrics.
        """
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
