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


class AssignSSSTCommandTerm(OrderCommandTerm):
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)


    def _update_assign_metrics(self):
        current_states = self.object_states

        # 1. 统计成功数量 (N,)
        num_success = (current_states == 3).sum(dim=1).float()

        # 2. 统计【本局】有效目标总数 (N,) is_target_mask 的和
        num_targets = self.is_target_mask.sum(dim=1).float()
        
        # 3. 统计活跃物品总数 (N,)
        num_active = self.is_active_mask.sum(dim=1).float()
        
        # 防止除零 (如果有环境完全没有目标)
        valid_envs = (num_targets > 0)
        valid_active = (num_active > 0)

        # --- Metric 1: 订单完成率 ---
        completion_rate = torch.zeros_like(num_success)
        completion_rate[valid_envs] = num_success[valid_envs] / num_targets[valid_envs]
        self.metrics["order_completion_rate"] = completion_rate

        # --- Metric 2: 平均完成时间 ---
        # 逻辑：当前耗时 / 已完成数量
        current_time_s = self.env.episode_length_buf.float() * self.env.step_dt
        
        mean_time = torch.zeros_like(num_success)
        has_success = (num_success > 0)
        mean_time[has_success] = current_time_s[has_success] / num_success[has_success]
        self.metrics["mean_action_time"] = mean_time

        # --- Metric 3: 失败率 (failure_rate) ---
        # 所有活跃物品中，状态为4（失败）的比例
        num_failed = ((current_states == 4) & self.is_active_mask).sum(dim=1).float()
        failure_rate = torch.zeros_like(num_failed)
        failure_rate[valid_active] = num_failed[valid_active] / num_active[valid_active]
        self.metrics["failure_rate"] = failure_rate

        # --- Metric 4: 错抓率 (wrong_pick_rate) ---
        # 干扰物 = 活跃但非目标物 (is_active_mask & ~is_target_mask)
        # 错抓 = 干扰物被移动了（状态不是1-待处理）
        distractor_mask = self.is_active_mask & (~self.is_target_mask)
        num_distractors = distractor_mask.sum(dim=1).float()
        
        # 干扰物被移动 = 干扰物 且 (状态不是1，说明离开了原料箱)
        distractor_moved = (distractor_mask & (current_states != 1)).sum(dim=1).float()
        
        wrong_pick_rate = torch.zeros_like(distractor_moved)
        valid_distractors = (num_distractors > 0)
        wrong_pick_rate[valid_distractors] = distractor_moved[valid_distractors] / num_distractors[valid_distractors]
        self.metrics["wrong_pick_rate"] = wrong_pick_rate

        # --- Metric 5: 错放率 (wrong_place_rate) ---
        # 目标物放到了错误的目标箱
        # 需要检测：目标物 且 在某个目标箱中 但不是正确的目标箱
        wrong_place_count = self._count_wrong_placements()
        
        # 已处理的目标物 = 目标物 且 状态不是1（已离开原料箱）
        processed_targets = (self.is_target_mask & (current_states != 1)).sum(dim=1).float()
        
        wrong_place_rate = torch.zeros_like(wrong_place_count)
        valid_processed = (processed_targets > 0)
        wrong_place_rate[valid_processed] = wrong_place_count[valid_processed] / processed_targets[valid_processed]
        self.metrics["wrong_place_rate"] = wrong_place_rate

    def _count_wrong_placements(self) -> torch.Tensor:
        """
        统计每个环境中，目标物被放到错误目标箱的数量
        返回: (N,) 每个环境的错放数量
        """
        from isaaclab_logistics_vla.utils.object_position import check_object_in_box
        
        env_ids = torch.arange(self.num_envs, device=self.device)
        wrong_place_count = torch.zeros(self.num_envs, device=self.device)
        
        for obj_idx, obj_asset in enumerate(self.object_assets):
            # 只检查目标物
            is_target = self.is_target_mask[:, obj_idx]
            if not is_target.any():
                continue
                
            # 该物品应该去的目标箱ID
            correct_target_id = self.obj_to_target_id[:, obj_idx]
            
            # 检查是否在某个错误的目标箱中
            for k in range(self.num_targets):
                in_box = check_object_in_box(
                    env_ids, obj_asset, self.target_box_assets[k], self.box_size_tensor
                )
                # 错放条件：是目标物 且 在箱子k中 且 k不是正确的目标箱
                wrong_in_box = is_target & in_box & (correct_target_id != k)
                wrong_place_count += wrong_in_box.float()
        
        return wrong_place_count