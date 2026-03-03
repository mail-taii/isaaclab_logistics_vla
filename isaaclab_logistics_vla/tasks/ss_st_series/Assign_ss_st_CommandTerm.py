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
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg

from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import BaseOrderCommandTerm


class AssignSSSTCommandTerm(BaseOrderCommandTerm):
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env,False)

    def _update_assign_metrics(self):
        """
            基于SKU需求量(target_need_sku_num)和现有量(target_contain_sku_num) 
            更新MDP的核心度量指标
        """
        #---1. 核心数量统计(Shape: [num_envs])---
        #实际有效数量：即使放多了，也只计算到需求量上限
        effective_contain = torch.clamp(self.target_contain_sku_num, max=self.target_need_sku_num)
        
        #汇总统计：每个环境的总需求量、总有效获得量、总活跃物体数
        total_need = self.target_need_sku_num.sum(dim=(1, 2)).float()
        total_effective_have = effective_contain.sum(dim=(1, 2)).float()
        total_active = self.is_active_mask.sum(dim=1).float()
        
        #预防除零错误的安全掩码
        valid_envs = (total_need > 0)
        valid_active = (total_active > 0)

        #---Metric 1: 订单完成率 (Order Completion Rate) ---
        completion_rate = torch.zeros_like(total_need)
        completion_rate[valid_envs] = total_effective_have[valid_envs] / total_need[valid_envs]
        self.metrics["order_completion_rate"] = completion_rate

        # --- Metric 2: 全订单成功率 (Whole Order Success) ---
        # 当实际有效拥有的数量 == 订单需要的总数量时，视为全订单成功
        whole_order_success = torch.zeros_like(total_need)
        whole_order_success[valid_envs] = (total_effective_have[valid_envs] == total_need[valid_envs]).float()
        self.metrics["whole_order_success"] = whole_order_success

        # --- Metric 3: 平均完成时间 (Mean Action Time) ---
        current_time_s = self.env.episode_length_buf.float() * self.env.step_dt
        mean_time = torch.zeros_like(total_need)
        has_success = (total_effective_have > 0)
        mean_time[has_success] = current_time_s[has_success] / total_effective_have[has_success]
        self.metrics["mean_action_time"] = mean_time

        # --- Metric 4: 失败率 (Failure Rate) ---
        # 依据 BaseOrderCommandTerm，状态 10 代表物理上失败（例如掉出场外）
        num_failed = (self.is_active_mask & (self.object_states == 10)).sum(dim=1).float()
        failure_rate = torch.zeros_like(total_need)
        failure_rate[valid_active] = num_failed[valid_active] / total_active[valid_active]
        self.metrics["failure_rate"] = failure_rate

        # --- Metric 5: 错抓率 (Wrong Pick Rate) ---
        # 依然使用分配时留下的标记 (-1) 来识别绝对的干扰物
        distractor_mask = self.is_active_mask & (self.obj_to_target_id == -1)
        num_distractors = distractor_mask.sum(dim=1).float()
        
        # 错抓判定：干扰物离开了原料箱。
        # 依据 BaseOrderCommandTerm: 1 代表原料箱1。状态不是 1 且激活，说明被移动了。
        distractor_moved = (distractor_mask & (self.object_states != 1) & (self.object_states != -1)).sum(dim=1).float()
        
        wrong_pick_rate = torch.zeros_like(total_need)
        valid_distractors = (num_distractors > 0)
        wrong_pick_rate[valid_distractors] = distractor_moved[valid_distractors] / num_distractors[valid_distractors]
        self.metrics["wrong_pick_rate"] = wrong_pick_rate

        # --- Metric 6: 错放率 (Wrong Place Rate) ---
        # 【全新算法】: 基于数量的优雅判定
        # 错放数量 = 箱子内某 SKU 实际数量 - 需要的数量 (且最小为 0)
        # 无论是错放了不需要的干扰物，还是放入了过多(冗余)的目标 SKU，都会被精准捕获。
        wrong_place_count = torch.clamp(self.target_contain_sku_num - self.target_need_sku_num, min=0).sum(dim=(1, 2)).float()
        
        # 已处理物体数 = 离开原料箱的激活物体数
        processed_items = (self.is_active_mask & (self.object_states != 1) & (self.object_states != -1)).sum(dim=1).float()
        
        wrong_place_rate = torch.zeros_like(wrong_place_count)
        valid_processed = (processed_items > 0)
        wrong_place_rate[valid_processed] = wrong_place_count[valid_processed] / processed_items[valid_processed]
        self.metrics["wrong_place_rate"] = wrong_place_rate