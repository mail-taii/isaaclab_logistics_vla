from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg

from isaaclab_logistics_vla.tasks.OrderCommandTerm import OrderCommandTerm

class AssignSSMTCommandTerm(OrderCommandTerm):
    """
    单原料箱->多目标箱的分配管理项，支持目标物前往特定箱子，或作为干扰物(target_id==-1)留在原位
    """
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _update_assign_metrics(self):
        current_states = self.object_states # (N, O)

        # 1. 基础掩码与统计
        # 真正的目标物：活跃且分配了目标箱(target_id >= 0)
        actual_target_mask = self.is_target_mask & (self.obj_to_target_id >= 0)
        # 干扰物/障碍物：活跃但未分配目标箱(target_id == -1)
        distractor_mask = self.is_active_mask & (self.obj_to_target_id == -1)

        num_success = (current_states == 3).sum(dim=1).float()
        num_targets = actual_target_mask.sum(dim=1).float()
        num_distractors = distractor_mask.sum(dim=1).float()
        
        valid_envs = (num_targets > 0)

        # --- Metric 1: 订单完成率 (仅针对真正的目标物品) ---
        completion_rate = torch.zeros_like(num_success)
        completion_rate[valid_envs] = num_success[valid_envs] / num_targets[valid_envs]
        self.metrics["order_completion_rate"] = completion_rate

        # --- Metric 2: 全订单成功率 ---
        all_done = (num_success == num_targets) & valid_envs
        self.metrics["whole_order_success"] = all_done.float()

        # --- Metric 3: 错抓率 (Wrong Pick Rate) ---
        # 判定：干扰物离开了原料箱 (状态 > 1)
        # 即使它没进目标箱，只要被移动了就算错抓
        distractor_moved = (distractor_mask & (current_states > 1)).sum(dim=1).float()
        wrong_pick_rate = torch.zeros_like(distractor_moved)
        valid_distractors = (num_distractors > 0)
        wrong_pick_rate[valid_distractors] = distractor_moved[valid_distractors] / num_distractors[valid_distractors]
        self.metrics["wrong_pick_rate"] = wrong_pick_rate

        # --- Metric 4: 错放率 (Wrong Place Rate) ---
        # 逻辑：目标物进了错误的箱子，或者干扰物进了任何一个目标箱
        wrong_place_count = self._count_wrong_placements(actual_target_mask, distractor_mask)
        # 已处理物品总数 (离开了原料箱的活跃物品)
        processed_active = (self.is_active_mask & (current_states > 1)).sum(dim=1).float()
        
        wrong_place_rate = torch.zeros_like(wrong_place_count)
        valid_processed = (processed_active > 0)
        wrong_place_rate[valid_processed] = wrong_place_count[valid_processed] / processed_active[valid_processed]
        self.metrics["wrong_place_rate"] = wrong_place_rate

    def _count_wrong_placements(self, actual_target_mask, distractor_mask) -> torch.Tensor:
        """
        统计每个环境中，目标物被放到错误目标箱的数量
        返回: (N,) 每个环境的错放数量
        """
        from isaaclab_logistics_vla.utils.object_position import check_object_in_box
        
        env_ids = torch.arange(self.num_envs, device=self.device)
        wrong_place_count = torch.zeros(self.num_envs, device=self.device)
        
        for obj_idx, obj_asset in enumerate(self.object_assets):
            is_target_env = actual_target_mask[:, obj_idx]
            is_distractor_env = distractor_mask[:, obj_idx]
            
            if not (is_target_env | is_distractor_env).any():
                continue
                
            correct_target_ids = self.obj_to_target_id[:, obj_idx] # (N,)
            
            for k in range(self.num_targets):
                in_box = check_object_in_box(env_ids, obj_asset, self.target_box_assets[k], self.box_size_tensor)
                
                # 情况 A: 目标物进了箱子 k，但 k 不是它的正确目的地
                wrong_target = is_target_env & in_box & (correct_target_ids != k)
                # 情况 B: 干扰物进了任何一个目标箱 k
                wrong_distractor = is_distractor_env & in_box
                
                wrong_place_count += (wrong_target | wrong_distractor).float()
        
        return wrong_place_count

    def _update_spawn_metrics(self): pass
    def _update_command(self): pass
    def command(self): pass