from __future__ import annotations

import torch
import json
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab_logistics_vla.tasks.ss_st_series.Assign_ss_st_CommandTerm import AssignSSSTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg


class Spawn_ss_st_sparse_with_obstacles_CommandTerm(AssignSSSTCommandTerm):
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.obstacle_names = getattr(cfg, "obstacles", ["large_obstacle"])
        self.obstacle_assets = [env.scene[name] for name in self.obstacle_names]
        
        # 索引定义：
        # 槽位[0, 1, 2] -> 现在给 3 个物品
        # 槽位[3, 4, 5] -> 现在给障碍物
        self.ITEM_COL_INDICES = [0, 1, 2]
        self.OBSTACLE_COL_INDICES = [3, 4, 5]
        self.OBSTACLE_CENTER_INDEX = 4

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)         # 本局选几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)   # 每种选几个

        for env_id in env_ids:
            num_to_sample = min(n_active_skus, self.num_skus)    #从所有 SKU 中随机选 n 种类
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            # 约定：第一个选中的是干扰物，剩下的是目标
            distractor_sku_idx = selected_sku_indices[0]
            target_sku_indices = selected_sku_indices[1:]

            # --- B. Instance 层级采样 (处理干扰物) ---
            sku_name = self.sku_names[distractor_sku_idx]
            global_indices = self.sku_to_indices[sku_name] # 拿到该 SKU 下所有实例 ID
            
            # 随机选 1 ~ m 个
            k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
            # 从 global_indices 中随机选 k 个
            selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]
            
            # 写入：干扰物 (Source=0, Target=-1)
            self.obj_to_source_id[env_id, selected_objs] = 0
            self.obj_to_target_id[env_id, selected_objs] = -1

            # --- C. Instance 层级采样 (处理目标物) ---
            for sku_idx in target_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                # 随机选 1 ~ m 个
                k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]

                # 写入：目标物 (Source=0, Target=0) (SSST模式)
                self.obj_to_source_id[env_id, selected_objs] = 0
                self.obj_to_target_id[env_id, selected_objs] = 0

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
            """
            核心放置函数：包含障碍物 Scale 随机化及自适应位置计算
            """
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device)
            
            num_envs = len(env_ids)
            box_x = WORK_BOX_PARAMS['X_LENGTH']
            box_y = WORK_BOX_PARAMS['Y_LENGTH']

            anchors = torch.tensor([
                [-box_x/3, -box_y/4], [0, -box_y/4], [box_x/3, -box_y/4], # 左列
                [-box_x/3,  box_y/4], [0,  box_y/4], [box_x/3,  box_y/4]  # 右列
            ], device=self.device)

            zero_vel = torch.zeros((len(env_ids), 6), device=self.device)
            far_far_away = torch.tensor([100.0, 100.0, -50.0], device=self.device).repeat(len(env_ids), 1)
            
            for asset in self.object_assets:
                # 瞬移到地下，防止在视野内乱飞
                set_asset_position(self.env, env_ids, asset, far_far_away)
                # 抹除所有残留速度
                asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

            # 辅助函数：获取旋转后的物理尺寸
            def get_params_and_dims(obj_name):
                if "cracker" in obj_name: p = CRACKER_BOX_PARAMS
                elif "sugar" in obj_name: p = SUGER_BOX_PARAMS
                elif "soup" in obj_name:  p = TOMATO_SOUP_CAN_PARAMS
                else: p = CRACKER_BOX_PARAMS 

                raw_x, raw_y, raw_z = p['X_LENGTH'], p['Y_LENGTH'], p['Z_LENGTH']

                ori_deg = p.get('SPARSE_ORIENT', (0, 0, 0))
                real_x, real_y, real_z = get_rotated_aabb_size(
                    raw_x, raw_y, raw_z, 
                    ori_deg, 
                    device=self.device
                )
                print(obj_name,real_x, real_y, real_z, ori_deg)
                return real_x, real_y, real_z, ori_deg

            # --- 1. 放置大障碍物 (尺寸随机化) ---
            scale_range = (0.4, 1)

            for obs_asset in self.obstacle_assets:
                # A. 随机化Scale并写入仿真
                rand_scales = torch.rand((num_envs, 1), device=self.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
                rand_scales = rand_scales.repeat(1, 3)  # 扩展为 (num_envs, 3)
                
                if hasattr(obs_asset, "write_root_scale_to_sim"):
                    obs_asset.write_root_scale_to_sim(rand_scales, env_ids=env_ids)
                
                # B. 获取缩放后的实际物理尺寸
                obs_cfg = obs_asset.cfg.spawn
                raw_size_z = obs_cfg.size[2]
                
                # 使用 asset.data 中的实时 scale 计算高度 (确保 write 后数据同步)
                current_scale_z = rand_scales[:, 2] 
                obs_z = (raw_size_z * current_scale_z / 2.0) + 0.015 + 0.019
                
                # C. 设置位置
                obs_center_anchor = anchors[self.OBSTACLE_CENTER_INDEX]
                obs_rel_pos = torch.zeros((num_envs, 3), device=self.device)
                obs_rel_pos[:, 0] = obs_center_anchor[0]
                obs_rel_pos[:, 1] = obs_center_anchor[1]
                obs_rel_pos[:, 2] = obs_z
                obs_rel_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)

                set_asset_relative_position(
                    self.env, env_ids, obs_asset, self.source_box_assets[0], 
                    obs_rel_pos, obs_rel_quat
                )

                obs_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

            # --- 2. 放置3个物品 ---
            item_slots_base = torch.tensor(self.ITEM_COL_INDICES, device=self.device)
            env_item_perms = torch.stack([item_slots_base[torch.randperm(3)] for _ in range(num_envs)])
            active_ranks = (self.obj_to_source_id[env_ids] != -1).long().cumsum(dim=1) - 1

            for obj_idx, obj_asset in enumerate(self.object_assets):
                assigned_mask = (self.obj_to_source_id[env_ids, obj_idx] != -1)
                
                if not assigned_mask.any():
                    far_pos = torch.zeros((num_envs, 3), device=self.device)
                    far_pos[:, 0] = 100.0
                    far_pos[:, 1] = 100.0
                    set_asset_position(self.env, env_ids, obj_asset, far_pos)
                    continue

                item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
                active_env_ids = env_ids[assigned_mask]
                current_slots = env_item_perms[assigned_mask, active_ranks[assigned_mask, obj_idx]]
                batch_anchors = anchors[current_slots]

                margin_x = max(0, (box_x/3.0 - item_x) / 2.0 - 0.01)
                margin_y = max(0, (box_y/2.0 - item_y) / 2.0 - 0.01)
                
                num_active = len(active_env_ids)
                rand_x = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_x
                rand_y = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_y
                z_pos = (item_z / 2.0) + 0.015 + 0.01

                rel_pos = torch.stack([
                    batch_anchors[:, 0] + rand_x,
                    batch_anchors[:, 1] + rand_y,
                    torch.full((num_active,), z_pos, device=self.device)
                ], dim=-1)

                rel_quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2]).repeat(num_active, 1)

                set_asset_relative_position(
                    self.env, active_env_ids, obj_asset, self.source_box_assets[0],
                    rel_pos, rel_quat
                )
    
    def _update_spawn_metrics(self): 
        pass
    def _update_command(self): 
        pass
    def command(self): 
        pass