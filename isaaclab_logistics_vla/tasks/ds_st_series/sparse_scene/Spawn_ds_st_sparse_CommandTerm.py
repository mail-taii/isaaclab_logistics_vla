from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab_logistics_vla.tasks.ds_st_series.Assign_ds_st_CommandTerm import AssignDSSTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg

class Spawn_ds_st_sparse_CommandTerm(AssignDSSTCommandTerm):
    
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)      # 本局选几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2) # 每种选几个
        num_source_box = getattr(self.cfg, "num_source_box", 2)         # 每局生成几个原料箱

        num_envs = len(env_ids)
        # 那cfg.num_source_box 就没用了，直接随机生成 2~3 个箱子??
        #self.num_source_box = torch.randint(low=2, high=4, size=(num_envs,), device=self.device)

        for env_id in env_ids:
            num_to_sample = min(n_active_skus, self.num_skus)    #从所有 SKU 中随机选 n 种类
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            # 约定：第一个选中的是干扰物，剩下的是目标
            distractor_sku_idx = selected_sku_indices[0]
            target_sku_indices = selected_sku_indices[1:]

            # --- B. Instance 层级采样 (处理干扰物) ---
            sku_name = self.sku_names[distractor_sku_idx]
            global_indices = self.sku_to_indices[sku_name] # 拿到该 SKU 下所有实例 ID

            for box_id in range (num_source_box): 
            
                # 随机选 1 ~ m 个
                k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
                # 从 global_indices 中随机选 k 个
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]

                # 写入：干扰物 (Source=0, Target=-1)
                self.obj_to_source_id[env_id, selected_objs] = box_id
                self.obj_to_target_id[env_id, selected_objs] = -1

                # --- C. Instance 层级采样 (处理目标物) ---
                for sku_idx in target_sku_indices:
                    sku_name = self.sku_names[sku_idx]
                    global_indices = self.sku_to_indices[sku_name]
                
                    # 随机选 1 ~ m 个
                    k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
                    selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]

                    # 写入：目标物 (Source=0, Target=0) (SSST模式)
                    self.obj_to_source_id[env_id, selected_objs] = box_id
                    self.obj_to_target_id[env_id, selected_objs] = 0

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        num_envs = len(env_ids)
        num_source_box = getattr(self.cfg, "num_source_box", 2)         # 每局生成几个原料箱

        for obj_idx, obj_asset in enumerate(self.object_assets):
            far_position = torch.zeros((num_envs, 3), device=self.device)
            far_position[:,0] = 100
            far_position[:,1] = 100

            quat = torch.zeros((num_envs, 4), device=self.device)
            quat[:,0] = 1

            set_asset_position(self.env,env_ids,obj_asset,far_position,quat)

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
            
        box_x = WORK_BOX_PARAMS['X_LENGTH']
        box_y = WORK_BOX_PARAMS['Y_LENGTH']

        cell_x = box_x / 3.0
        cell_y = box_y / 2.0

        anchors = torch.tensor([
            [-box_x/3, -box_y/4], [0,-box_y/4 ], [box_x/3,  -box_y/4],
            [ -box_x/3, box_y/4], [ 0, box_y/4], [ box_x/3, box_y/4]
        ], device=self.device)

        # 预先生成随机槽位排列，为了让每个箱子的摆放都不一样，增加一个维度 (num_boxes)
        # shape: (num_envs, num_boxes, 6)
        num_boxes = num_source_box
        slot_perms = torch.stack([
            torch.rand(num_envs, 6, device=self.device).argsort(dim=1) 
            for _ in range(num_boxes)
        ], dim=1)

        # self.obj_to_source_id[env_ids] != -1 shape(num_envs,num_objects),bool,为0的是不生成SKU
        # cumsum(dim=1)  所有生成的SKU，给一个编号，从0递增 e.g. [[0,-,-,1,2,-,-]]
        active_ranks = (self.obj_to_source_id[env_ids] != -1).long().cumsum(dim=1) - 1

        for obj_idx, obj_asset in enumerate(self.object_assets):
            item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
            # assigned_box_indices:shape(num_envs,)  e.g. [-1,0,1,-1,0] 每个SKU 在每个环境中的原料箱
            assigned_box_indices = self.obj_to_source_id[env_ids, obj_idx]

            for box_idx, box_asset in enumerate(self.source_box_assets):
                #shape(num_envs,) bool 自然排除了不生成物体 
                mask = (assigned_box_indices == box_idx) 
                if not mask.any(): continue

                active_env_ids = env_ids[mask]    #某SKU应在当前箱子生成的环境列表
                num_active = len(active_env_ids)

                # --- 核心修改点 1: 独立计算当前箱子内的排名 ---
                # 我们需要知道在当前箱子里，这个 obj_idx 是第几个进去的
                # 只针对属于当前 box_idx 的物体进行 cumsum
                items_in_this_box_mask = (self.obj_to_source_id[env_ids] == box_idx)
                # shape: (num_envs, num_objects) -> 取出当前物体的排名
                # 第二步：转成数字并累加（就是报数计数）
                # .long() 后变成 [1, 0, 1, 1]
                # .cumsum() 累加后变成 [1, 1, 2, 3]  <-- 重点：它只在属于 0 号箱的位置增加
                # - 1 后变成 [0, 0, 1, 2]
                current_ranks = (items_in_this_box_mask.long().cumsum(dim=1) - 1)[mask, obj_idx]

                # --- 核心修改点 2: 从 slot_perms 中精准提取 ---
                # slot_perms[mask] 拿到了活跃环境的排列，shape: (num_active, num_boxes, 6)
                # 我们需要先选定当前的 box_idx，再根据 rank 拿 slot
                current_slots = slot_perms[mask, box_idx].gather(1, current_ranks.unsqueeze(1)).squeeze(1)

                
                #遗留问题：constant中记录的是SKU的绝对旋转量，不是相对箱子的旋转。现在假定箱子无旋转
                relative_quat = euler_to_quat_isaac(item_ori[0],item_ori[1],item_ori[2]).repeat(num_active, 1)
                # shape(num_active,) 应当在当前箱子生成此SKU的环境中，此SKU排第几申领槽位
                #current_ranks = active_ranks[mask, obj_idx]
                # slot_perms[mask]：shape(num_active,6) current_ranks.unsqueeze(1)：shape(num_active,1)
                # gather：按照current_ranks，在num_active个环境中，为SKU申请槽位
                # current_slots: shape(num_active,) 当前SKU在环境中的具体槽位编号
                #current_slots = slot_perms[mask].gather(1, current_ranks.unsqueeze(1)).squeeze(1)  
                  
                # batch_anchors: shape(num_active, 2) 转槽位为相对位置
                batch_anchors = anchors[current_slots]

                margin_x = max(0, (cell_x - item_x) / 2.0 - 0.01) # 留1cm缝隙
                margin_y = max(0, (cell_y - item_y) / 2.0 - 0.01)

                rand_x = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_x
                rand_y = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_y
                
                z_pos = (item_z / 2.0) + 0.015+0.01    #0.015是箱子底厚度

                relative_pos = torch.stack([
                    batch_anchors[:, 0] + rand_x,
                    batch_anchors[:, 1] + rand_y,
                    torch.full((num_active,), z_pos, device=self.device)
                ], dim=-1)

                set_asset_relative_position(
                    env=self.env,
                    env_ids=active_env_ids,
                    target_asset=obj_asset,
                    reference_asset=box_asset,
                    relative_pos=relative_pos,
                    relative_quat=relative_quat
                )

    def _update_spawn_metrics(self):
        pass

    def _update_command(self):
        pass

    def command(self):
        pass