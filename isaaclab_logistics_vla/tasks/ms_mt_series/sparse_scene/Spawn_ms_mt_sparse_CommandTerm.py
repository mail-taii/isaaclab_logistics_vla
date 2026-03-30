from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg

from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import BaseOrderCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *


class Spawn_ms_mt_sparse_CommandTerm(BaseOrderCommandTerm):
    
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __str__(self) -> str:
        return "ms_st_sparse"

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        #---初始化：-1表示不激活/未分配原料箱---
        self.obj_to_source_id[env_ids] = -1
        self.target_need_sku_num[env_ids] = 0

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)         # 本局选几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)   # 每种选几个
        num_sources = len(self.source_box_assets)
        num_targets = len(self.target_box_assets)

        for env_id in env_ids:
            #---1. 随机选出本局会出现的SKU种类池---
            num_to_sample = min(n_active_skus, self.num_skus)
            all_selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()
            
            # 为了让3个源箱都有货，建立一个轮询池
            source_pool = torch.randperm(num_sources, device=self.device).tolist()
            source_ptr = 0

            # 记录每个箱子已被分配了多少物体，防止超过槽位限制(6个)
            box_fill_counts = torch.zeros(num_sources, device=self.device, dtype=torch.long)

            #---2. 遍历每个目标箱(生成订单需求)---
            for target_idx in range(num_targets):
                num_skus_in_order = torch.randint(1, 4, (1,)).item()
                order_skus = torch.tensor(all_selected_sku_indices)[torch.randperm(len(all_selected_sku_indices))[:num_skus_in_order]]
                
                for sku_idx in order_skus.tolist():
                    sku_name = self.sku_names[sku_idx]
                    global_indices = self.sku_to_indices[sku_name]
                    
                    # 通过obj_to_source_id判断该实例是否还“空闲”
                    available_indices = [idx for idx in global_indices if self.obj_to_source_id[env_id, idx] == -1]
                    if not available_indices:
                        continue
                    
                    k = torch.randint(1, min(len(available_indices), m_max_per_sku) + 1, (1,)).item()
                    
                    #指定一个唯一的原料箱ID
                    assigned_source_id = source_pool[source_ptr % num_sources]
                    if box_fill_counts[assigned_source_id] + k <= 6:
                        selected_objs = torch.tensor(available_indices, device=self.device)[torch.randperm(len(available_indices))[:k]]
                        
                        #仅记录原料箱来源，不再记录target_id
                        self.obj_to_source_id[env_id, selected_objs] = assigned_source_id
                        
                        #记录订单需求量
                        self.target_need_sku_num[env_id, target_idx, sku_idx] += k
                        box_fill_counts[assigned_source_id] += k
                    
                    source_ptr += 1

            #---3. 处理干扰物(放入源箱但没有任何订单需要的物品)---
            for sku_idx in all_selected_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                #找出剩余的空闲实例
                available_indices = [idx for idx in global_indices if self.obj_to_source_id[env_id, idx] == -1]
                
                if available_indices:
                    rand_source = torch.randint(0, num_sources, (1,), device=self.device).item()
                    remaining_space = 6 - box_fill_counts[rand_source].item()
                    
                    if remaining_space > 0:
                        k = torch.randint(0, min(len(available_indices), remaining_space) + 1, (1,)).item()
                        if k > 0:
                            distractor_objs = torch.tensor(available_indices)[torch.randperm(len(available_indices))[:k]]
                            self.obj_to_source_id[env_id, distractor_objs] = rand_source
                            box_fill_counts[rand_source] += k

        #只要被放进原料箱了，就是激活状态
        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        num_envs = len(env_ids)
        zero_vel = torch.zeros((num_envs, 6), device=self.device)

        #---1. 预设所有物品到地下并零速化---
        for obj_asset in self.object_assets:
            far_pos = torch.tensor([[0.0, 0.0, -10.0]], device=self.device).repeat(num_envs, 1)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
            set_asset_position(self.env, env_ids, obj_asset, far_pos, quat)
            if hasattr(obj_asset, "reset"):
                obj_asset.reset(env_ids=env_ids)
            if hasattr(obj_asset, "write_root_velocity_to_sim"):
                obj_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

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
        anchors = torch.tensor([
            [-box_x/3, -box_y/4], [0, -box_y/4], [box_x/3, -box_y/4],
            [-box_x/3,  box_y/4], [0,  box_y/4], [box_x/3,  box_y/4]
        ], device=self.device)

        #---2. 为每个原料箱独立计算槽位占用---
        for box_idx, box_asset in enumerate(self.source_box_assets):
            slots_used = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            slot_perms = torch.stack([torch.randperm(6, device=self.device) for _ in range(num_envs)])

            for obj_idx, obj_asset in enumerate(self.object_assets):
                mask = (self.obj_to_source_id[env_ids, obj_idx] == box_idx)
                if not mask.any(): 
                    continue

                active_env_ids = env_ids[mask]
                num_active = len(active_env_ids)
                item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
                
                # 分配槽位
                current_slots = slot_perms[mask].gather(1, slots_used[mask].unsqueeze(1)).squeeze(1)
                batch_anchors = anchors[current_slots]
                slots_used[mask] += 1 

                # 随机偏移逻辑
                margin_x = max(0, (cell_x - item_x) / 2.0 - 0.01)
                margin_y = max(0, (cell_y - item_y) / 2.0 - 0.01)
                rand_x = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_x
                rand_y = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_y
                
                z_pos = (item_z / 2.0) + 0.015 + 0.02
                rel_pos = torch.stack([
                    batch_anchors[:, 0] + rand_x,
                    batch_anchors[:, 1] + rand_y,
                    torch.full((num_active,), z_pos, device=self.device)
                ], dim=-1)
                rel_quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2]).repeat(num_active, 1)

                set_asset_relative_position(self.env, active_env_ids, obj_asset, box_asset, rel_pos, rel_quat)
                
                # 重置物理状态，防止在空中乱飘
                if hasattr(obj_asset, "reset"):
                    obj_asset.reset(env_ids=active_env_ids)
                if hasattr(obj_asset, "write_root_velocity_to_sim"):
                    obj_asset.write_root_velocity_to_sim(torch.zeros((num_active, 6), device=self.device), env_ids=active_env_ids)

    def _update_spawn_metrics(self): 
        target_needs = self.target_need_sku_num 
        actual_in_target = self.target_contain_sku_num

        correct_picks = torch.minimum(actual_in_target, target_needs).sum(dim=(1, 2))
        wrong_picks = torch.clamp(actual_in_target - target_needs, min=0).sum(dim=(1, 2))
        dropped_count = (self.object_states == 10).sum(dim=1)
        total_needed = target_needs.sum(dim=(1, 2))

        completion_rate = torch.where(
            total_needed > 0,
            correct_picks.float() / total_needed.float(),
            torch.tensor(1.0, device=self.device) 
        )
        
        is_success = (correct_picks == total_needed) & (wrong_picks == 0)

        self.metrics = {
            "completion_rate": completion_rate,
            "wrong_pick_count": wrong_picks,
            "dropped_count": dropped_count,
            "is_success": is_success.float(),
            "correct_picks": correct_picks,
            "total_needed": total_needed
        }
        
        return self.metrics
    
    def _update_command(self): 
        pass

    def command(self): 
        pass