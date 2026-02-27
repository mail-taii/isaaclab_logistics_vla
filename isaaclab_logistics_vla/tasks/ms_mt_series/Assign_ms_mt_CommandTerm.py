from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg

from isaaclab_logistics_vla.tasks.OrderCommandTerm import OrderCommandTerm
from isaaclab_logistics_vla.utils.object_position import check_object_in_box

class AssignMSMTCommandTerm(OrderCommandTerm):
    """
    多原料箱-多目标箱(MS-MT)任务分配管理项，负责计算订单完成率、错放率等核心MDP指标
    """
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _update_assign_metrics(self):

        # 获取当前所有物体的状态(num_envs, num_objects)
        # 状态定义: 1: 在原料箱, 2: 移动中, 3: 在目标箱(成功), 4: 失败(掉落等)
        current_states = self.object_states

        # --- 基础计数 ---
        # 1. 成功归位的目标物数量(num_envs,)
        num_success = (self.is_target_mask & (current_states == 3)).sum(dim=1).float()

        # 2. 本局有效目标总数(num_envs,)
        num_targets = self.is_target_mask.sum(dim=1).float()
        
        # 3. 活跃物品总数 (num_envs,) - 包含干扰物
        num_active = self.is_active_mask.sum(dim=1).float()
        
        # 有效性掩码，防止除以零
        valid_envs = (num_targets > 0)
        valid_active = (num_active > 0)

        #---指标1: 订单完成率(Completion Rate)---
        completion_rate = torch.zeros(self.num_envs, device=self.device)
        completion_rate[valid_envs] = num_success[valid_envs] / num_targets[valid_envs]
        self.metrics["order_completion_rate"] = completion_rate

        #---指标2: 全订单成功率(Whole Order Success)---
        # 只有当成功数量等于目标总数时，该环境的任务才算彻底完成
        whole_order_success = torch.zeros(self.num_envs, device=self.device)
        whole_order_success[valid_envs] = (num_success[valid_envs] == num_targets[valid_envs]).float()
        self.metrics["whole_order_success"] = whole_order_success

        #---指标3: 平均操作时间(Mean Action Time)---
        current_time_s = self.env.episode_length_buf.float() * self.env.step_dt
        mean_time = torch.zeros(self.num_envs, device=self.device)
        has_success = (num_success > 0)
        mean_time[has_success] = current_time_s[has_success] / num_success[has_success]
        self.metrics["mean_action_time"] = mean_time

        #---指标4: 失败率(Failure Rate)---
        # 活跃物品中状态为4的比例
        num_failed = (self.is_active_mask & (current_states == 4)).sum(dim=1).float()
        failure_rate = torch.zeros(self.num_envs, device=self.device)
        failure_rate[valid_active] = num_failed[valid_active] / num_active[valid_active]
        self.metrics["failure_rate"] = failure_rate

        #---指标5: 错抓率(Wrong Pick Rate)---
        #干扰物定义: 活跃但非目标物(is_active_mask & ~is_target_mask)
        distractor_mask = self.is_active_mask & (~self.is_target_mask)
        num_distractors = distractor_mask.sum(dim=1).float()
        #错抓判定: 干扰物离开了初始位置（状态不再是1）
        distractor_moved = (distractor_mask & (current_states != 1)).sum(dim=1).float()
        
        wrong_pick_rate = torch.zeros(self.num_envs, device=self.device)
        valid_distractors = (num_distractors > 0)
        wrong_pick_rate[valid_distractors] = distractor_moved[valid_distractors] / num_distractors[valid_distractors]
        self.metrics["wrong_pick_rate"] = wrong_pick_rate

        #---指标 6: 错放率(Wrong Place Rate)---
        #在MS-MT中，目标物进入错误的Target Box，或干扰物进入任何Target Box
        wrong_place_count = self._count_wrong_placements()
        #已离开原料箱的物品总数作为分母
        processed_items = (self.is_active_mask & (current_states != 1)).sum(dim=1).float()
        
        wrong_place_rate = torch.zeros(self.num_envs, device=self.device)
        valid_processed = (processed_items > 0)
        wrong_place_rate[valid_processed] = wrong_place_count[valid_processed] / processed_items[valid_processed]
        self.metrics["wrong_place_rate"] = wrong_place_rate

    def _count_wrong_placements(self) -> torch.Tensor:
        """
        统计MS-MT场景下的错放数量：
            1. 目标物出现在与其obj_to_target_id不符的目标箱中
            2. 干扰物（TargetID = -1）出现在任何一个目标箱中
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        wrong_place_count = torch.zeros(self.num_envs, device=self.device)
        
        for obj_idx, obj_asset in enumerate(self.object_assets):
            is_active = self.is_active_mask[:, obj_idx]
            if not is_active.any():
                continue
                
            #获取该物品正确的目标箱索引(-1表示它是干扰物)
            correct_target_id = self.obj_to_target_id[:, obj_idx]
            
            for k in range(self.num_targets):
                # 检查该物品是否在第k个目标箱内
                in_box_k = check_object_in_box(
                    env_ids, obj_asset, self.target_box_assets[k], self.box_size_tensor
                )
                
                # 错放判定：
                # 物品在箱子k中，且k不是它该去的那个箱子
                # (这也涵盖了干扰物，因为干扰物的correct_target_id 是-1，永远不会等于k)
                is_wrong_in_k = is_active & in_box_k & (correct_target_id != k)
                wrong_place_count += is_wrong_in_k.float()
        
        return wrong_place_count
