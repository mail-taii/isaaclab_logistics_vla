from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
from isaaclab_logistics_vla.tasks.ss_mt_series.Assign_ss_mt_CommandTerm import AssignSSMTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg

class Spawn_ss_mt_sparse_CommandTerm(AssignSSMTCommandTerm):
    """
    负责将物品分配到不同的订单箱，并在唯一的原料箱内进行随机布局。
    """
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        """分配物品逻辑：支持多订单目标分配及干扰物设置。"""
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)             # 本局选几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)       # 每种选几个

        for env_id in env_ids:
            #---1. 采样 SKU 种类---
            num_to_sample = min(n_active_skus, self.num_skus)
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            #---2. 确定哪些 SKU 是目标，哪些是干扰物---
            # 随机决定目标 SKU 的数量 (至少 1 个)
            num_target_skus = torch.randint(1, len(selected_sku_indices) + 1, (1,)).item()
            target_sku_indices = selected_sku_indices[:num_target_skus]
            distractor_sku_indices = selected_sku_indices[num_target_skus:]

            #---A. 处理目标物(Target Items)---
            for sku_idx in target_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]
                
                # 为每个实例随机分配一个目标箱(0 ~ num_targets-1)
                target_box_ids = torch.randint(0, self.num_targets, (k,), device=self.device)
                
                self.obj_to_source_id[env_id, selected_objs] = 0
                self.obj_to_target_id[env_id, selected_objs] = target_box_ids

            #---B. 处理干扰物(Distractor Items)---
            for sku_idx in distractor_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]

                # Source=0, Target=-1 代表它只出现在原料箱，不属于任何订单
                self.obj_to_source_id[env_id, selected_objs] = 0
                self.obj_to_target_id[env_id, selected_objs] = -1

        # 更新掩码
        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        # 只要 active 且 target_id != -1，就是本局需要抓取的目标
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        num_envs = len(env_ids)
        zero_vel = torch.zeros((num_envs, 6), device=self.device)

        #---1. 预处理：将所有物品先行瞬移至远处并重置物理状态---
        for obj_asset in self.object_assets:
            far_pos = torch.tensor([[100.0, 100.0, -50.0]], device=self.device).repeat(num_envs, 1)
            set_asset_position(self.env, env_ids, obj_asset, far_pos)
            if hasattr(obj_asset, "reset"):
                obj_asset.reset(env_ids=env_ids)
            obj_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        # 辅助函数：获取物品参数
        def get_params_and_dims(obj_name):
            if "cracker" in obj_name: p = CRACKER_BOX_PARAMS
            elif "sugar" in obj_name: p = SUGER_BOX_PARAMS
            elif "soup" in obj_name:  p = TOMATO_SOUP_CAN_PARAMS
            else: p = CRACKER_BOX_PARAMS 
            raw_x, raw_y, raw_z = p['X_LENGTH'], p['Y_LENGTH'], p['Z_LENGTH']
            ori_deg = p.get('SPARSE_ORIENT', (0, 0, 0))
            real_x, real_y, real_z = get_rotated_aabb_size(raw_x, raw_y, raw_z, ori_deg, device=self.device)
            return real_x, real_y, real_z, ori_deg
            
        box_x, box_y = WORK_BOX_PARAMS['X_LENGTH'], WORK_BOX_PARAMS['Y_LENGTH']
        cell_x, cell_y = box_x / 3.0, box_y / 2.0

        #---2. 定义 6 个槽位的锚点---
        anchors = torch.tensor([
            [-box_x/3, -box_y/4], [0, -box_y/4], [box_x/3, -box_y/4],
            [-box_x/3,  box_y/4], [0,  box_y/4], [box_x/3,  box_y/4]
        ], device=self.device)

        #---3. 为每个环境生成随机的槽位排列---
        slot_perms = torch.stack([torch.randperm(6, device=self.device) for _ in range(num_envs)])
        #---4. 计算每个活跃物品在当前环境中的顺序编号---
        active_ranks = (self.obj_to_source_id[env_ids] != -1).long().cumsum(dim=1) - 1

        for obj_idx, obj_asset in enumerate(self.object_assets):
            #---5. 获取该物品在哪些环境中是活跃的---
            assigned_mask = (self.obj_to_source_id[env_ids, obj_idx] == 0)
            if not assigned_mask.any():
                continue

            item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
            active_env_ids = env_ids[assigned_mask]
            num_active = len(active_env_ids)

            #---6. 分配槽位---
            current_ranks = active_ranks[assigned_mask, obj_idx]
            current_slots = slot_perms[assigned_mask].gather(1, current_ranks.unsqueeze(1)).squeeze(1)
            batch_anchors = anchors[current_slots]

            #---7. 随机偏移---
            margin_x = max(0, (cell_x - item_x) / 2.0 - 0.01)
            margin_y = max(0, (cell_y - item_y) / 2.0 - 0.01)
            rand_x = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_x
            rand_y = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_y
            
            z_pos = (item_z / 2.0) + 0.015 + 0.01

            rel_pos = torch.stack([
                batch_anchors[:, 0] + rand_x,
                batch_anchors[:, 1] + rand_y,
                torch.full((num_active,), z_pos, device=self.device)
            ], dim=-1)
            rel_quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2]).repeat(num_active, 1)

            # 设置位置并重置物理
            set_asset_relative_position(
                self.env, active_env_ids, obj_asset, self.source_box_assets[0], 
                rel_pos, rel_quat
            )
            if hasattr(obj_asset, "reset"):
                obj_asset.reset(env_ids=active_env_ids)
            obj_asset.write_root_velocity_to_sim(torch.zeros((num_active, 6), device=self.device), env_ids=active_env_ids)

    def _update_spawn_metrics(self): pass
    def _update_command(self): pass
    def command(self): pass