from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import BaseOrderCommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *


class Spawn_ss_mt_sparse_CommandTerm(BaseOrderCommandTerm):
    """
        采用最新的target_need_sku_num数量级跟踪逻辑。
        随机将SKU需求量打散分配给多个订单箱。
    """
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # --- 障碍物兼容初始化 ---
        self.obstacle_names = getattr(cfg, "obstacles", []) or []
        if self.obstacle_names:
            self.obstacle_assets = [env.scene[name] for name in self.obstacle_names if name in env.scene.keys()]
            self.has_obstacles = True
            # 有障碍物时：左边列 [0,1,2] 给物品，障碍物放在右边列中心 [4]
            self.ITEM_COL_INDICES = [0, 1, 2]
            self.OBSTACLE_CENTER_INDEX = 4
        else:
            self.obstacle_assets = []
            self.has_obstacles = False
            # 无障碍物时：全箱 6 个槽位都给物品
            self.ITEM_COL_INDICES = [0, 1, 2, 3, 4, 5]

    def __str__(self) -> str:
        has_obs = getattr(self, "has_obstacles", False)
        return "ss_mt_sparse_with_obstacles" if has_obs else "ss_mt_sparse"

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        """分配物品逻辑：基于数量的多订单目标分配及槽位保护。"""
        #移除obj_to_target_id依赖
        self.obj_to_source_id[env_ids] = -1
        self.target_need_sku_num[env_ids] = 0

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)
        
        max_slots = len(self.ITEM_COL_INDICES)

        for env_id in env_ids:
            #---1. 采样本局活跃的SKU种类---
            num_to_sample = min(n_active_skus, self.num_skus)
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            num_distractor_skus = 1 if num_to_sample > 1 else 0
            distractor_only_sku_indices = selected_sku_indices[:num_distractor_skus]
            target_sku_indices = selected_sku_indices[num_distractor_skus:]

            slots_used = 0

            #---2. 处理目标SKU(多订单随机分配逻辑)---
            for sku_idx in target_sku_indices:
                if slots_used >= max_slots: 
                    break

                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                max_avail = min(len(global_indices), m_max_per_sku)
                max_can_spawn = min(max_avail, max_slots - slots_used)
                if max_can_spawn < 1: 
                    continue
                
                #A: 确定本局总共需要几个该SKU
                n_need = torch.randint(1, max_can_spawn + 1, (1,)).item()
                
                #B: 【核心SS-MT逻辑】将这些需求量随机打散，分配到不同的订单箱中
                #随机生成n_need个目标箱ID
                target_box_ids = torch.randint(0, self.num_targets, (n_need,), device=self.device)
                for tb_id in target_box_ids:
                    self.target_need_sku_num[env_id, tb_id.item(), sku_idx] += 1
                
                #C: 确定实际生成量(包含盈余干扰)
                n_spawn = torch.randint(n_need, max_can_spawn + 1, (1,)).item()
                slots_used += n_spawn
                
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:n_spawn]]
                self.obj_to_source_id[env_id, selected_objs] = 0

            #---3. 处理纯干扰物SKU---
            for sku_idx in distractor_only_sku_indices:
                if slots_used >= max_slots:
                    break
                
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                max_avail = min(len(global_indices), m_max_per_sku)
                max_can_spawn = min(max_avail, max_slots - slots_used)
                if max_can_spawn < 1:
                    continue
                
                n_spawn = torch.randint(1, max_can_spawn + 1, (1,)).item()
                slots_used += n_spawn
                
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:n_spawn]]
                self.obj_to_source_id[env_id, selected_objs] = 0

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """物理放置逻辑(完全兼容SS-ST的优良特性)"""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        num_envs = len(env_ids)

        #---1. 清场与零速化---
        zero_vel = torch.zeros((num_envs, 6), device=self.device)
        for obj_asset in self.object_assets:
            far_position = torch.tensor([[100.0, 100.0, -50.0]], device=self.device).repeat(num_envs, 1)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
            set_asset_position(self.env, env_ids, obj_asset, far_position, quat)
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

        #---2. 处理大障碍物(如果存在)---
        if self.has_obstacles:
            scale_range = (0.4, 1.0)
            for obs_asset in self.obstacle_assets:
                rand_scales = torch.rand((num_envs, 1), device=self.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
                rand_scales = rand_scales.repeat(1, 3) 
                if hasattr(obs_asset, "write_root_scale_to_sim"):
                    obs_asset.write_root_scale_to_sim(rand_scales, env_ids=env_ids)
                
                obs_cfg = obs_asset.cfg.spawn
                raw_size_z = obs_cfg.size[2]
                current_scale_z = rand_scales[:, 2] 
                obs_z = (raw_size_z * current_scale_z / 2.0) + 0.015 + 0.019
                
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
                if hasattr(obs_asset, "write_root_velocity_to_sim"):
                    obs_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        #---3. 处理普通物品---
        item_slots_base = torch.tensor(self.ITEM_COL_INDICES, device=self.device)
        env_item_perms = torch.stack([item_slots_base[torch.randperm(len(self.ITEM_COL_INDICES))] for _ in range(num_envs)])
        
        active_ranks = (self.obj_to_source_id[env_ids] != -1).long().cumsum(dim=1) - 1

        for obj_idx, obj_asset in enumerate(self.object_assets):
            assigned_mask = (self.obj_to_source_id[env_ids, obj_idx] != -1)
            if not assigned_mask.any():
                continue     

            item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
            active_env_ids = env_ids[assigned_mask]
            num_active = len(active_env_ids)
            
            relative_quat = euler_to_quat_isaac(item_ori[0],item_ori[1],item_ori[2]).repeat(num_active, 1)
            
            current_ranks = active_ranks[assigned_mask, obj_idx]
            current_slots = env_item_perms[assigned_mask, current_ranks]
            batch_anchors = anchors[current_slots]

            margin_x = max(0, (cell_x - item_x) / 2.0 - 0.01)
            margin_y = max(0, (cell_y - item_y) / 2.0 - 0.01)
            rand_x = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_x
            rand_y = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_y
            
            z_pos = (item_z / 2.0) + 0.015 + 0.02 

            relative_pos = torch.stack([
                batch_anchors[:, 0] + rand_x,
                batch_anchors[:, 1] + rand_y,
                torch.full((num_active,), z_pos, device=self.device)
            ], dim=-1)

            set_asset_relative_position(
                env=self.env, env_ids=active_env_ids,
                target_asset=obj_asset, reference_asset=self.source_box_assets[0],
                relative_pos=relative_pos, relative_quat=relative_quat
            )
            
            if hasattr(obj_asset, "reset"):
                obj_asset.reset(env_ids=active_env_ids)
            if hasattr(obj_asset, "write_root_velocity_to_sim"):
                obj_asset.write_root_velocity_to_sim(torch.zeros((num_active, 6), device=self.device), env_ids=active_env_ids)

    def _update_spawn_metrics(self):
        # =========================================================
        # 1. 读取完整的需求与实际矩阵(包含所有的目标箱和所有的SKU)
        # Shape: (num_envs, num_targets, num_skus)
        # =========================================================
        target_needs = self.target_need_sku_num 
        actual_in_target = self.target_contain_sku_num

        # =========================================================
        # 2. 核心得分项计算(SS-MT特有：沿dim=1(箱子)和dim=2(SKU)同时求和)
        # =========================================================
        # A. 正确抓取量：在每个订单箱内，每种SKU的min(需要量, 实际量)的总和
        correct_picks = torch.minimum(actual_in_target, target_needs).sum(dim=(1, 2))
        
        # B. 错误/多余抓取量：放错了箱子，或者放对了箱子但超过了该箱子该SKU的需求量
        wrong_picks = torch.clamp(actual_in_target - target_needs, min=0).sum(dim=(1, 2))
        
        # C. 物理掉落计数：状态被标记为10的物品总数(仅需沿物品维度求和)
        # Shape: object_states是(num_envs, num_objects)
        dropped_count = (self.object_states == 10).sum(dim=1)
        
        # D. 订单总需求量：所有目标箱、所有SKU的需求总和
        total_needed = target_needs.sum(dim=(1, 2))

        # =========================================================
        # 3. 最终复合指标导出
        # =========================================================
        # 订单完成度: 0.0~1.0(防止total_needed为0导致除以0报错)
        completion_rate = torch.where(
            total_needed > 0,
            correct_picks.float() / total_needed.float(),
            torch.tensor(1.0, device=self.device) 
        )
        
        # 完美成功标志: 需求全满，且没有任何错拿/多拿/乱扔
        is_success = (correct_picks == total_needed) & (wrong_picks == 0)

        # 组装Metrics字典
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