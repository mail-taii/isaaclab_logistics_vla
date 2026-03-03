from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab_logistics_vla.tasks.ss_st_series.Assign_ss_st_CommandTerm import AssignSSSTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg

class Spawn_ss_st_sparse_CommandTerm(AssignSSSTCommandTerm):
    """
        1. 逻辑分配：决定每个环境的订单需求量(target_need_sku_num)和实际生成量。
        2. 物理生成：计算物体的位姿并将其放置到场景中。
    """

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        #---初始化清理：抹除上一局的记忆---
        #-1代表该物品不属于任何订单(Target)或不出现在任何原料箱(Source)
        self.obj_to_source_id[env_ids] = -1

        #清空当前批次环境的订单需求记录，防止上一局的残余数据干扰本局计算
        self.target_need_sku_num[env_ids] = 0

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)         # 规定本局总共出现几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)   # 规定每种 SKU 最多生成几个实例

        for env_id in env_ids:
            #---第1步：采样本局活跃的SKU种类池---
            num_to_sample = min(n_active_skus, self.num_skus)
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            #角色划分：如果抽出的SKU种类超过1种，我们把第1种作为“纯干扰物”(订单绝对不需要它)
            #剩下的种类作为“目标物”(订单需要它，但也可能有多余的同类干扰物)
            num_distractor_skus = 1 if num_to_sample > 1 else 0
            distractor_only_sku_indices = selected_sku_indices[:num_distractor_skus]
            target_sku_indices = selected_sku_indices[num_distractor_skus:]

            #---第2步：处理目标SKU(核心需求与盈余干扰逻辑)---
            for sku_idx in target_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]  #获取该类SKU在大数组中的所有实例索引
                
                #确定可用实例上限(不能超过配置的max_instances_per_sku，也不能超过模型库里实际拥有的数量)
                max_avail = min(len(global_indices), m_max_per_sku)
                if max_avail < 1:
                    continue    # 如果没货了，直接跳过
                
                #【逻辑 A：确定需求量】
                #目标箱真正需要几个？(随机1到max_avail个)
                n_need = torch.randint(1, max_avail + 1, (1,)).item()
                #将需求量写入核心张量(SS-ST模式下，订单箱索引固定为0，即第0个目标箱)
                self.target_need_sku_num[env_id, 0, sku_idx] = n_need
                
                #【逻辑 B：确定实际生成量】
                #实际在箱子里生成几个？(必须>=n_need，多出来的就是同种类的干扰物)
                n_spawn = torch.randint(n_need, max_avail + 1, (1,)).item()
                
                #从该SKU的所有实例库中，随机抽出n_spawn个具体物体
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:n_spawn]]

                #由于这是SS-ST(单原料箱)，所以选中的物体统统放在第0号原料箱里
                self.obj_to_source_id[env_id, selected_objs] = 0

            #---第3步：处理纯干扰物SKU(这类物品订单完全不需要)---
            for sku_idx in distractor_only_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                
                max_avail = min(len(global_indices), m_max_per_sku)
                if max_avail < 1:
                    continue
                
                #这里不需要改target_need_sku_num，因为它初始化就是0，代表不需要
                #随机生成1到Max个纯干扰物
                n_spawn = torch.randint(1, max_avail + 1, (1,)).item()
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:n_spawn]]
                
                self.obj_to_source_id[env_id, selected_objs] = 0    #同样放在0号原料箱

        #---第4步：更新全局活跃物体掩码---
        #只要SourceID不是-1，就说明这个物体在本局游戏中出场了(无论是目标还是干扰)
        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)

        #此处彻底废弃了self.is_target_mask，因为现在的指标全靠target_need_sku_num来算了

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """
            根据_assign_objects_boxes决定好的SourceID，将物体物理放置到对应的箱子槽位中。
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        num_envs = len(env_ids)

        #---第1步：清场---
        #把所有物体强制瞬移到天上/地下很远的地方，防止旧物体的碰撞箱影响新物体的生成
        for obj_asset in self.object_assets:
            far_position = torch.tensor([[100.0, 100.0, -50.0]], device=self.device).repeat(num_envs, 1)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)
            set_asset_position(self.env, env_ids, obj_asset, far_position, quat)

        #辅助函数：根据物品名称获取预设的物理尺寸和旋转角度
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
            
        #箱子的长宽参数
        box_x, box_y = WORK_BOX_PARAMS['X_LENGTH'], WORK_BOX_PARAMS['Y_LENGTH']
        #把箱子划分为3行x2列的6个格子，计算每个格子的尺寸
        cell_x, cell_y = box_x / 3.0, box_y / 2.0
        
        #预先计算好6个格子的中心点坐标(基于箱子中心的相对坐标)
        anchors = torch.tensor([
            [-box_x/3, -box_y/4], [0, -box_y/4], [box_x/3, -box_y/4],
            [-box_x/3,  box_y/4], [0,  box_y/4], [box_x/3,  box_y/4]
        ], device=self.device)

        #---第2步：分配槽位---
        #为每个环境生成一个0~5的随机排列序列 (例如[3, 0, 5, 1, 2, 4])，用于决定物品占据哪个格子
        #argsort的作用是将随机数排序并返回索引，从而巧妙地获得不重复的乱序列表
        slot_perms = torch.rand(num_envs, 6, device=self.device).argsort(dim=1)
        
        #累计求和(cumsum)：为每个环境里即将生成的物品打上序号(0, 1, 2...)
        #这个序号将作为提取slot_perms的索引，保证同一个环境里的物品不会拿到相同的格子
        active_ranks = (self.obj_to_source_id[env_ids] != -1).long().cumsum(dim=1) - 1

        #---第3步：逐个物体进行放置---
        for obj_idx, obj_asset in enumerate(self.object_assets):
            item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
            assigned_box_indices = self.obj_to_source_id[env_ids, obj_idx]

            for box_idx, box_asset in enumerate(self.source_box_assets):
                #筛出本局中确实需要放在当前box_idx里的物品
                mask = (assigned_box_indices == box_idx) 
                if not mask.any(): continue     #没有的话就看下一个箱子

                active_env_ids = env_ids[mask]
                num_active = len(active_env_ids)
                
                #转换欧拉角为四元数
                relative_quat = euler_to_quat_isaac(item_ori[0],item_ori[1],item_ori[2]).repeat(num_active, 1)
                
                #查表：当前物品是该环境里的第几个生成的？根据这个序号去槽位数组里拿具体的格子ID
                current_ranks = active_ranks[mask, obj_idx]
                current_slots = slot_perms[mask].gather(1, current_ranks.unsqueeze(1)).squeeze(1)    
                # 拿到格子的中心点相对坐标
                batch_anchors = anchors[current_slots]

                #---添加随机扰动(Jitter)---
                #为了防止物品永远呆在格子正中央导致过拟合，我们加入随机偏移
                #容差margin确保即使偏移，物品边缘也不会超出当前格子边界(减去0.01是留出安全缝隙)
                margin_x = max(0, (cell_x - item_x) / 2.0 - 0.01)
                margin_y = max(0, (cell_y - item_y) / 2.0 - 0.01)

                #生成-1到1之间的随机数并乘以容差
                rand_x = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_x
                rand_y = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_y
                
                #Z轴高度计算：物品高度的一半+箱子底厚度(0.015)+空气垫高度(0.02)
                #增加0.02的空气垫是为了防止刚体穿模瞬间产生巨大的排斥力把物品弹飞
                z_pos = (item_z / 2.0) + 0.015 + 0.02 

                #拼接最终的三维相对坐标
                relative_pos = torch.stack([
                    batch_anchors[:, 0] + rand_x,
                    batch_anchors[:, 1] + rand_y,
                    torch.full((num_active,), z_pos, device=self.device)
                ], dim=-1)

                #执行底层放置命令
                set_asset_relative_position(
                    env=self.env, active_env_ids=active_env_ids,
                    target_asset=obj_asset, reference_asset=box_asset,
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