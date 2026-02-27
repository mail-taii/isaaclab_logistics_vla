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


class AssignDSSTCommandTerm(OrderCommandTerm):
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)


    def _update_assign_metrics(self):
        current_states = self.object_states

        # 1. 统计成功数量 (N,)
        num_success = (current_states == 3).sum(dim=1).float()

        # 2. 统计【本局】有效目标总数 (N,)s_target_mask 的和
        num_targets = self.is_target_mask.sum(dim=1).float()
        
        # 防止除零 (如果有环境完全没有目标)
        valid_envs = (num_targets > 0)

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

        # 源箱维度分析 (Source Box Analytics) 
        num_source_boxes = len(self.source_box_assets)
        box_clearance_list = []

        for s_idx in range(num_source_boxes):
            # 找到在该环境下，属于这个特定原料箱的目标物体掩码
            source_mask = (self.obj_to_source_id == s_idx) & self.is_target_mask
            
            # 如果该环境在这个箱子里没放东西，跳过
            if not source_mask.any():
                continue
            
            # 计算该箱子的清理率
            cleared = ((current_states == 3) & source_mask).sum(dim=1).float()
            total = source_mask.sum(dim=1).float()
            
            clearance = cleared / (total + 1e-5)
            self.metrics[f"source_box_{s_idx}_clearance"] = clearance
            box_clearance_list.append(clearance)
        # --- Metric 3. [新增] 源箱抓取均衡度 (Balance Score) ---
        # 如果有多个源箱，计算它们清理率的标准差（越小代表越均衡）
        if len(box_clearance_list) > 1:
            all_clearance = torch.stack(box_clearance_list, dim=1)
            # 标准差越小，说明机器人对各个箱子的关注度越均匀
            self.metrics["source_box_std"] = torch.std(all_clearance, dim=1)

        self.metrics["mean_action_time"] = mean_time