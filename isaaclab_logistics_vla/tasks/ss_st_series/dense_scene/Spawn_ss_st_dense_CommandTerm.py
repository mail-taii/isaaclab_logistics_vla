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

class Spawn_ss_st_dense_CommandTerm(AssignSSSTCommandTerm):

    
    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    
    def _precompute_sku_data(self):
        """
        这个函数只在 __init__ 时运行一次。
        它负责计算体积，并将 SKU 索引分类存入 shape_groups。
        """
        # 1. 定义存储结构
        self.sku_dims = {}
        self.sku_volumes = {}
        
        # === 这里就是 shape_groups 的来源 ===
        self.shape_groups = {
            'cuboid': [],    # 存放所有长方体物品的 index
            'irregular': []  # 存放所有异形物品的 index
        }

        # 定义名字到参数的映射
        sku_config_map = {
            "003_cracker_box": CRACKER_BOX_PARAMS,
            "004_sugar_box":   SUGER_BOX_PARAMS,
            "005_tomato_soup_can": TOMATO_SOUP_CAN_PARAMS
        }

        # 遍历你所有的 SKU
        for idx, sku_name in enumerate(self.sku_names):
            # A. 找到对应的参数
            cfg = None
            for key, val in sku_config_map.items():
                if key in sku_name:
                    cfg = val
                    break
            
            # B. 判断形状并分类
            if cfg:
                dims = torch.tensor([cfg['X_LENGTH'], cfg['Y_LENGTH'], cfg['Z_LENGTH']])
                vol = torch.prod(dims).item()
                
                # 核心分类逻辑：有 RADIUS 字段的算异形，否则算长方体
                if 'RADIUS' in cfg:
                    shape_type = 'irregular'
                else:
                    shape_type = 'cuboid'
            else:
                # 找不到参数的默认处理
                dims = torch.tensor([0.1, 0.1, 0.1])
                vol = 0.001
                shape_type = 'cuboid'
                print(f"[Warning] Using default params for {sku_name}")

            # C. 存数据
            self.sku_dims[sku_name] = dims
            self.sku_volumes[sku_name] = vol
            
            # === 这里将 ID 加入对应的组 ===
            self.shape_groups[shape_type].append(idx)

        print(f"[Info] Shape Groups Initialization: {self.shape_groups}")

    '''
    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)      # 本局选几种 SKU
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2) # 每种选几个

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
    '''

    def _assign_objects_boxes_dense(self, env_ids: Sequence[int]):
        # 初始化
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        self.obj_to_sub_container_id[env_ids] = -1

        sub_box_dims = self._get_sub_box_dims()
        # 遍历 self.shape_groups 字典，提取出包含有效物品的形态名称列表（如 ['cuboid', 'irregular']）
        available_shapes = [k for k, v in self.shape_groups.items() if len(v) > 0]

        for env_id in env_ids:
            
            # =========================================================
            # Phase 1: 全局决策 (Decide Strategy)
            # =========================================================
            
            # 1. 随机决定两个箱子的形态
            shape_0 = available_shapes[torch.randint(0, len(available_shapes), (1,)).item()]
            shape_1 = available_shapes[torch.randint(0, len(available_shapes), (1,)).item()]
            
            # 2. 决定干扰物 (Distractor Logic)
            if shape_0 == shape_1:
                # 【关键逻辑】形态撞车 -> 强制使用同一种干扰物
                candidates = self.shape_groups[shape_0]
                # 随机选出一个 SKU 的索引，作为“公用干扰物”
                common_dist_idx = candidates[torch.randint(0, len(candidates), (1,)).item()]
                
                dist_idx_0 = common_dist_idx
                dist_idx_1 = common_dist_idx
            else:
                # 形态不同 -> 各选各的
                cand_0 = self.shape_groups[shape_0]
                dist_idx_0 = cand_0[torch.randint(0, len(cand_0), (1,)).item()]
                
                cand_1 = self.shape_groups[shape_1]
                dist_idx_1 = cand_1[torch.randint(0, len(cand_1), (1,)).item()]

            # 3. 决定目标物 (Target Logic) - 始终独立选择
            cand_0 = self.shape_groups[shape_0]
            target_idx_0 = cand_0[torch.randint(0, len(cand_0), (1,)).item()]
            
            cand_1 = self.shape_groups[shape_1]
            target_idx_1 = cand_1[torch.randint(0, len(cand_1), (1,)).item()]

            # 将决策打包，方便后面循环处理
            # 列表索引 0 对应左框，1 对应右框
            box_configs = [
                {'shape': shape_0, 'target_idx': target_idx_0, 'dist_idx': dist_idx_0},
                {'shape': shape_1, 'target_idx': target_idx_1, 'dist_idx': dist_idx_1}
            ]

            # =========================================================
            # Phase 2: 执行填充 (Execute Filling)
            # =========================================================
            
            for sub_box_idx, config in enumerate(box_configs):
                shape_type = config['shape']
                target_sku_idx = config['target_idx']
                dist_sku_idx = config['dist_idx']
                
                target_sku_name = self.sku_names[target_sku_idx]
                dist_sku_name = self.sku_names[dist_sku_idx]

                # --- 1. 计算容量 (基于目标物面积) ---
                efficiency = 0.6 if shape_type == 'irregular' else 0.75
                k_limit = self._calculate_fill_count(target_sku_name, sub_box_dims, packing_efficiency=efficiency)
                
                # --- 2. 随机总数 ---
                if k_limit > 1:
                    total_count = torch.randint(1, k_limit + 1, (1,)).item()
                else:
                    total_count = 1
                
                # --- 3. 切分目标与干扰比例 ---
                if total_count > 1:
                    # 至少 1 个目标
                    n_targets = torch.randint(1, total_count + 1, (1,)).item()
                    n_distractors = total_count - n_targets
                else:
                    n_targets = 1
                    n_distractors = 0
                
                # --- 4. 写入 ID (Target) ---
                global_indices_t = self.sku_to_indices[target_sku_name]
                n_t_final = min(n_targets, len(global_indices_t))
                
                # 记录已经被选为目标的 ID，防止同种物品冲突
                used_indices = []
                
                if n_t_final > 0:
                    selected_targets = torch.tensor(global_indices_t)[torch.randperm(len(global_indices_t))[:n_t_final]]
                    
                    self.obj_to_source_id[env_id, selected_targets] = 0
                    self.obj_to_target_id[env_id, selected_targets] = 0
                    self.obj_to_sub_container_id[env_id, selected_targets] = sub_box_idx
                    
                    used_indices = selected_targets.tolist()

                # --- 5. 写入 ID (Distractor) ---
                if n_distractors > 0:
                    global_indices_d = self.sku_to_indices[dist_sku_name]
                    
                    # 冲突检测：如果干扰物和目标物是同一种 SKU
                    if target_sku_idx == dist_sku_idx:
                        # 排除掉刚才被选走的目标
                        remaining = [i for i in global_indices_d if i not in used_indices]
                        n_d_final = min(n_distractors, len(remaining))
                        if n_d_final > 0:
                            selected_distractors = torch.tensor(remaining)[torch.randperm(len(remaining))[:n_d_final]]
                    else:
                        # 不同 SKU，直接选
                        n_d_final = min(n_distractors, len(global_indices_d))
                        selected_distractors = torch.tensor(global_indices_d)[torch.randperm(len(global_indices_d))[:n_d_final]]
                    
                    # 赋值
                    if 'selected_distractors' in locals() and len(selected_distractors) > 0:
                        self.obj_to_source_id[env_id, selected_distractors] = 0
                        self.obj_to_target_id[env_id, selected_distractors] = -1
                        self.obj_to_sub_container_id[env_id, selected_distractors] = sub_box_idx

        # 更新 mask
        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)


    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        num_envs = len(env_ids)

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

        # slot_perms:shape(num_envs, 6), e.g. [[3,0,5,1,2,4]]
        slot_perms = torch.rand(num_envs, 6, device=self.device).argsort(dim=1)

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
                #遗留问题：constant中记录的是SKU的绝对旋转量，不是相对箱子的旋转。现在假定箱子无旋转
                relative_quat = euler_to_quat_isaac(item_ori[0],item_ori[1],item_ori[2]).repeat(num_active, 1)
                # shape(num_active,) 应当在当前箱子生成此SKU的环境中，此SKU排第几申领槽位
                current_ranks = active_ranks[mask, obj_idx]
                # slot_perms[mask]：shape(num_active,6) current_ranks.unsqueeze(1)：shape(num_active,1)
                # gather：按照current_ranks，在num_active个环境中，为SKU申请槽位
                # current_slots: shape(num_active,) 当前SKU在环境中的具体槽位编号
                current_slots = slot_perms[mask].gather(1, current_ranks.unsqueeze(1)).squeeze(1)    
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