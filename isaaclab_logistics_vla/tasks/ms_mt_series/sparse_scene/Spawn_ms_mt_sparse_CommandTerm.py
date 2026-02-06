from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab_logistics_vla.tasks.ms_mt_series.Assign_ms_mt_CommandTerm import AssignMSMTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg

class Spawn_ms_mt_sparse_CommandTerm(AssignMSMTCommandTerm):
    
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        #---初始化：-1表示不激活/不属于任何箱子---
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)         # 本局选几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)   # 每种选几个
        num_sources = len(self.source_box_assets)
        num_targets = len(self.target_box_assets)

        for env_id in env_ids:
            #---1. 随机选出本局会出现的SKU种类池---
            num_to_sample = min(n_active_skus, self.num_skus)

            #随机打乱索引并截取，得到本局活跃的SKU索引列表
            all_selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()
            
            #为了让3个源箱都有货，建立一个轮询池
            #随机打乱源箱ID，例如得到[2, 0, 1]，确保前三次分配各占一个箱子
            source_pool = torch.randperm(num_sources, device=self.device).tolist()
            source_ptr = 0      #源箱轮询指针

            #记录每个箱子已被分配了多少物体，防止超过槽位限制(6个)
            box_fill_counts = torch.zeros(num_sources, device=self.device, dtype=torch.long)

            #---2. 遍历每个目标箱(每个订单)---
            for target_idx in range(num_targets):
                #每个订单随机选1~3种SKU(从池子中选)
                num_skus_in_order = torch.randint(1, 4, (1,)).item()
                order_skus = torch.tensor(all_selected_sku_indices)[torch.randperm(len(all_selected_sku_indices))[:num_skus_in_order]]
                
                #遍历该订单选中的每一种SKU
                for sku_idx in order_skus.tolist():
                    sku_name = self.sku_names[sku_idx]
                    global_indices = self.sku_to_indices[sku_name]
                    
                    #过滤掉在本环境下已经被分配给其他订单的该类SKU实例
                    available_indices = [idx for idx in global_indices if self.obj_to_target_id[env_id, idx] == -1]
                    #如果该SKU已经被分光了，跳过
                    if not available_indices:
                        continue
                    
                    #决定该订单对该SKU的具体需求数量k(1到上限m_max_per_sku)
                    k = torch.randint(1, min(len(available_indices), m_max_per_sku) + 1, (1,)).item()
                    
                    #约束1. 同一订单所需要的同种sku必须来自同一个原料箱
                    #约束2. 不同订单所需要的同种sku可以来自同一个原料箱子，也可以来自不同的原料箱     
                    #为当前订单的当前SKU随机指定一个唯一的原料箱ID(实现同订单同SKU同源)
                    assigned_source_id = source_pool[source_ptr % num_sources]
                    if box_fill_counts[assigned_source_id] + k <= 6:
                        selected_objs = torch.tensor(available_indices)[torch.randperm(len(available_indices))[:k]]
                        self.obj_to_target_id[env_id, selected_objs] = target_idx
                        self.obj_to_source_id[env_id, selected_objs] = assigned_source_id
                        box_fill_counts[assigned_source_id] += k
                    
                    source_ptr += 1

            #---3. 处理干扰物 (那些还没被分配到订单的active_skus实例)---
            for sku_idx in all_selected_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                #找出那些在前面的订单分配阶段没被选中的“剩余”实例
                available_indices = [idx for idx in global_indices if self.obj_to_target_id[env_id, idx] == -1]
                
                if available_indices:
                    #随机选源箱
                    rand_source = torch.randint(0, num_sources, (1,), device=self.device).item()
                    #检查该箱子剩余空间
                    remaining_space = 6 - box_fill_counts[rand_source].item()
                    
                    if remaining_space > 0:
                        #在剩余空间内随机决定干扰物数量
                        k = torch.randint(0, min(len(available_indices), remaining_space) + 1, (1,)).item()
                        if k > 0:
                            distractor_objs = torch.tensor(available_indices)[torch.randperm(len(available_indices))[:k]]
                            self.obj_to_target_id[env_id, distractor_objs] = -1
                            #确保source_id也是Tensor并对齐设备
                            self.obj_to_source_id[env_id, distractor_objs] = torch.full((k,), rand_source, device=self.device, dtype=torch.long)
                            box_fill_counts[rand_source] += k

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        num_envs = len(env_ids)

        # 预设所有物品到远处
        for obj_asset in self.object_assets:
            far_pos = torch.full((num_envs, 3), 100.0, device=self.device)
            set_asset_position(self.env, env_ids, obj_asset, far_pos)

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

        # 为每个原料箱独立计算槽位占用
        for box_idx, box_asset in enumerate(self.source_box_assets):
            # 记录该箱子里每个环境已经用了多少个槽位
            slots_used = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            # 每个环境对该箱子生成随机槽位序列
            slot_perms = torch.stack([torch.randperm(6, device=self.device) for _ in range(num_envs)])

            for obj_idx, obj_asset in enumerate(self.object_assets):
                # 检查哪些环境的该物品被分配到了当前原料箱
                mask = (self.obj_to_source_id[env_ids, obj_idx] == box_idx)
                if not mask.any(): continue

                active_env_ids = env_ids[mask]
                num_active = len(active_env_ids)
                item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
                
                # 分配槽位
                current_slots = slot_perms[mask].gather(1, slots_used[mask].unsqueeze(1)).squeeze(1)
                batch_anchors = anchors[current_slots]
                slots_used[mask] += 1 # 槽位指针增加

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

    def _update_spawn_metrics(self): pass
    def _update_command(self): pass
    def command(self): pass