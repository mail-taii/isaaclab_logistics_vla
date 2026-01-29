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

        self.metrics["mean_action_time"] = mean_time