from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import BaseOrderCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg

class Spawn_ss_st_sparse_CommandTerm(BaseOrderCommandTerm):

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        #---障碍物兼容初始化---
        self.obstacle_names = getattr(cfg, "obstacles", []) or []
        if self.obstacle_names:
            self.obstacle_assets = [env.scene[name] for name in self.obstacle_names if name in env.scene.keys()]
            self.has_obstacles = True
            #有障碍物时：左边列[0,1,2]给物品，障碍物放在右边列中心[4]
            self.ITEM_COL_INDICES = [0, 1, 2]
            self.OBSTACLE_CENTER_INDEX = 4
        else:
            self.obstacle_assets = []
            self.has_obstacles = False
            #无障碍物时：全箱6个槽位都给物品
            self.ITEM_COL_INDICES = [0, 1, 2, 3, 4, 5]

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        #---初始化清理：抹除上一局的记忆---
        #-1代表该物品不属于任何订单(Target)或不出现在任何原料箱(Source)
        self.obj_to_source_id[env_ids] = -1

        #清空当前批次环境的订单需求记录，防止上一局的残余数据干扰本局计算
        self.target_need_sku_num[env_ids] = 0

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)         # 规定本局总共出现几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)   # 规定每种 SKU 最多生成几个实例

        #获取当前模式下最大可用的物品槽位数(3或6)
        max_slots = len(self.ITEM_COL_INDICES)

        for env_id in env_ids:
            #---第1步：采样本局活跃的SKU种类池---
            num_to_sample = min(n_active_skus, self.num_skus)
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            #角色划分：如果抽出的SKU种类超过1种，我们把第1种作为“纯干扰物”(订单绝对不需要它)
            #剩下的种类作为“目标物”(订单需要它，但也可能有多余的同类干扰物)
            num_distractor_skus = 1 if num_to_sample > 1 else 0
            distractor_only_sku_indices = selected_sku_indices[:num_distractor_skus]
            target_sku_indices = selected_sku_indices[num_distractor_skus:]

            # 容量保护指针：记录当前环境已经分配了几个物品
            slots_used = 0

            #---第2步：处理目标SKU(核心需求与盈余干扰逻辑)---
            for sku_idx in target_sku_indices:
                if slots_used >= max_slots: 
                    break       #槽位已满，停止分配剩余SKU

                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]  #获取该类SKU在大数组中的所有实例索引
                
                #确定可用实例上限(不能超过配置的max_instances_per_sku，也不能超过模型库里实际拥有的数量)
                max_avail = min(len(global_indices), m_max_per_sku)
                #核心限制：最大生成量不能超过剩余的槽位数
                max_can_spawn = min(max_avail, max_slots - slots_used)
                if max_can_spawn < 1: 
                    continue
                
                #【逻辑 A：确定需求量】
                #目标箱真正需要几个？(随机1到max_avail个)
                n_need = torch.randint(1, max_can_spawn + 1, (1,)).item()
                #将需求量写入核心张量(SS-ST模式下，订单箱索引固定为0，即第0个目标箱)
                self.target_need_sku_num[env_id, 0, sku_idx] = n_need
                
                #【逻辑 B：确定实际生成量】
                #实际在箱子里生成几个？(必须>=n_need，多出来的就是同种类的干扰物)
                n_spawn = torch.randint(n_need, max_can_spawn + 1, (1,)).item()
                slots_used += n_spawn
                
                #从该SKU的所有实例库中，随机抽出n_spawn个具体物体
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:n_spawn]]
                
                #由于这是SS-ST(单原料箱)，所以选中的物体统统放在第0号原料箱里
                self.obj_to_source_id[env_id, selected_objs] = 0

            #---第3步：处理纯干扰物SKU(这类物品订单完全不需要)---
            for sku_idx in distractor_only_sku_indices:
                if slots_used >= max_slots:
                    break       #槽位满了就停止塞入纯干扰物

                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                max_avail = min(len(global_indices), m_max_per_sku)
                max_can_spawn = min(max_avail, max_slots - slots_used)
                if max_can_spawn < 1:
                    continue
                
                #这里不需要改target_need_sku_num，因为它初始化就是0，代表不需要
                #随机生成1到Max个纯干扰物
                n_spawn = torch.randint(1, max_can_spawn + 1, (1,)).item()
                slots_used += n_spawn
                
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:n_spawn]]
                self.obj_to_source_id[env_id, selected_objs] = 0    #同样放在0号原料箱

        #---第4步：更新全局活跃物体掩码---
        #只要SourceID不是-1，就说明这个物体在本局游戏中出场了(无论是目标还是干扰)
        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """
            根据_assign_objects_boxes决定好的SourceID，将物体物理放置到对应的箱子槽位中。
            兼容障碍物：根据是否有障碍物动态选择可用槽位。
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        num_envs = len(env_ids)

        #---第1步：清场与零速化---
        zero_vel = torch.zeros((num_envs, 6), device=self.device)
        for obj_asset in self.object_assets:
            far_position = torch.tensor([[100.0, 100.0, -50.0]], device=self.device).repeat(num_envs, 1)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
            set_asset_position(self.env, env_ids, obj_asset, far_position, quat)
            if hasattr(obj_asset, "write_root_velocity_to_sim"):
                obj_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

        def get_params_and_dims(obj_name):
            if "cracker" in obj_name: 
                p = CRACKER_BOX_PARAMS
            elif "sugar" in obj_name: 
                p = SUGER_BOX_PARAMS
            elif "soup" in obj_name: 
                p = TOMATO_SOUP_CAN_PARAMS
            else:
                p = CRACKER_BOX_PARAMS 
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

        #---第2步：处理大障碍物(如果存在)---
        if self.has_obstacles:
            scale_range = (0.4, 1.0)
            for obs_asset in self.obstacle_assets:
                #A. 随机化Scale并写入仿真
                rand_scales = torch.rand((num_envs, 1), device=self.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
                rand_scales = rand_scales.repeat(1, 3) 
                if hasattr(obs_asset, "write_root_scale_to_sim"):
                    obs_asset.write_root_scale_to_sim(rand_scales, env_ids=env_ids)
                
                #B. 计算缩放后的实际物理高度
                obs_cfg = obs_asset.cfg.spawn
                raw_size_z = obs_cfg.size[2]
                current_scale_z = rand_scales[:, 2] 
                obs_z = (raw_size_z * current_scale_z / 2.0) + 0.015 + 0.019
                
                #C. 设置固定位置
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

        #---第3步：处理普通物品---
        #生成槽位序列池。根据是否有障碍物，长度可能是3(左列)或6(全部)
        item_slots_base = torch.tensor(self.ITEM_COL_INDICES, device=self.device)
        #为每个环境生成独享的乱序可用槽位
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
            
            #查表：提取分配的槽位锚点
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
                env=self.env, env_ids=env_ids,
                target_asset=obj_asset, reference_asset=self.source_box_assets[0],
                relative_pos=relative_pos, relative_quat=relative_quat
            )
            
            if hasattr(obj_asset, "reset"):
                obj_asset.reset(env_ids=active_env_ids)
            if hasattr(obj_asset, "write_root_velocity_to_sim"):
                obj_asset.write_root_velocity_to_sim(torch.zeros((num_active, 6), device=self.device), env_ids=active_env_ids)

    def _update_spawn_metrics(self): 
        pass

    def _update_command(self): 
        pass

    def command(self):
        pass

    def __str__(self) -> str:
        has_obs = getattr(self, "has_obstacles", False)
        return "ss_st_sparse_with_obstacles" if has_obs else "ss_st_sparse"