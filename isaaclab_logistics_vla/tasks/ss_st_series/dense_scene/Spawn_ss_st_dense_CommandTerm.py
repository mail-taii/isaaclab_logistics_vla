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
        self.cfg.obj_to_tray_id = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
        )

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        # ==========================================
        # 0. 初始化状态 & 存储容器
        # ==========================================
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # 重置 ID 映射
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        
        # 初始化预计算容器
        if not hasattr(self, "saved_relative_pos"):
            self.saved_relative_pos = torch.zeros((self.num_envs, self.num_objects, 3), device=self.device)
            self.saved_relative_quat = torch.zeros((self.num_envs, self.num_objects, 4), device=self.device)
            self.saved_relative_quat[..., 0] = 1.0 
        
        self.saved_relative_pos[env_ids] = 0
        self.saved_relative_quat[env_ids] = 0
        self.saved_relative_quat[env_ids, :, 0] = 1.0

        # 获取配置
        n_active_skus = getattr(self.cfg, "num_active_skus", 3)
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 6)
        tray_config = getattr(self.cfg, "tray_or_not", [0, 0, 0])
        is_tray_scene = tray_config[0]

        BOX_X_LEN = 0.36
        BOX_Y_LEN = 0.56
        TRAY_X_LEN = 0.30
        TRAY_Y_LEN = 0.23
        resolution = 0.005
        
        SCALED_OBJECTS1= ["CN_big", "SF_small", "empty_plastic_package", "SF_big"]
        SCALE_FACTOR1= torch.tensor([0.3, 0.3, 0.3], device=self.device)
        SCALED_OBJECTS2= ["cracker_box","sugar_box","tomato_soup_can"]
        SCALE_FACTOR2= torch.tensor([0.8, 0.8, 0.8], device=self.device)

        # --- 内部辅助函数: 支持传入 force_ori ---
        def get_real_dims_local(obj_name, force_ori=None):
            if "cracker" in obj_name: p = CRACKER_BOX_PARAMS
            elif "sugar" in obj_name: p = SUGER_BOX_PARAMS
            elif "soup" in obj_name:  p = TOMATO_SOUP_CAN_PARAMS
            elif "CN_big" in obj_name: p = CN_BIG_PARAMS
            elif "SF_small" in obj_name: p = SF_SMALL_PARAMS
            elif "empty_plastic_package" in obj_name: p = EMPTY_PLASTIC_PACKAGE_PARAMS
            elif "SF_big" in obj_name: p = SF_BIG_PARAMS
            else: p = CRACKER_BOX_PARAMS 

            raw_dims = torch.tensor([p['X_LENGTH'], p['Y_LENGTH'], p['Z_LENGTH']], device=self.device)
            if any(s in obj_name for s in SCALED_OBJECTS1): raw_dims *= SCALE_FACTOR1
            elif any(s in obj_name for s in SCALED_OBJECTS2): raw_dims *= SCALE_FACTOR2
            
            if force_ori is not None:
                # 情况 A: 强制使用传入的朝向 (保持一致性)
                ori_deg = force_ori
                print(f"Object {obj_name} is forced to use orientation {ori_deg}")
            else:
                # 情况 B: 没有强制，则随机选一个新的 (通常用于初始化)
                ori_deg_options = p.get('DENSE_ORIENT', (0, 0, 0)) # 这里根据你需要用 DENSE 或 SPARSE
                
                if isinstance(ori_deg_options, list):
                    # 使用 torch 生成随机索引
                    idx = torch.randint(0, len(ori_deg_options), (1,)).item()
                    ori_deg = ori_deg_options[idx]
                else:
                    ori_deg = ori_deg_options

            real_x, real_y, real_z = get_rotated_aabb_size(
                raw_dims[0], raw_dims[1], raw_dims[2], ori_deg, device=self.device
            )
            return real_x, real_y, real_z, ori_deg

        # ==========================================
        # 1. 遍历环境进行分配和计算
        # ==========================================
        for env_id in env_ids:
            env_idx_int = int(env_id.item())

            # ---------------------------------------------------
            # 分支 A: 托盘模式
            # ---------------------------------------------------
            if is_tray_scene:
                num_trays = 2
                num_to_sample = min(num_trays, self.num_skus)
                selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

                for tray_idx, sku_idx in enumerate(selected_sku_indices):
                    sku_name = self.sku_names[sku_idx]
                    global_indices = self.sku_to_indices[sku_name]
                    
                    # 托盘模式相对简单，我们在计算容量前先确定好朝向
                    # 调用一次 helper，不传参，让它随机选一个，然后存下来
                    _, _, _, fixed_ori = get_real_dims_local(sku_name, force_ori=None)

                    # 再次调用 helper，这次强制使用刚才选好的 fixed_ori
                    item_x, item_y, item_z, item_ori = get_real_dims_local(sku_name, force_ori=fixed_ori)
                    
                    # ... (后续容量计算逻辑保持不变) ...
                    margin = 0.01
                    inner_x = TRAY_X_LEN - 0.01
                    inner_y = TRAY_Y_LEN - 0.01
                    eff_item_x = item_x + margin
                    eff_item_y = item_y + margin
                    
                    cols = int(inner_x / eff_item_x)
                    rows = int(inner_y / eff_item_y)
                    if cols < 1: cols = 1
                    if rows < 1: rows = 1
                    max_capacity = cols * rows
                    
                    actual_max = min(max_capacity, len(global_indices))
                    if actual_max < 1: k = 1 
                    else: k = torch.randint(1, actual_max + 1, (1,)).item()
                    selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]
                    
                    for rank, obj_idx in enumerate(selected_objs):
                        grid_x = rank % cols
                        grid_y = rank // cols
                        start_x_local = -inner_x / 2.0 + eff_item_x / 2.0
                        start_y_local = -inner_y / 2.0 + eff_item_y / 2.0
                        pos_x = start_x_local + grid_x * eff_item_x
                        pos_y = start_y_local + grid_y * eff_item_y
                        
                        TRAY_THICK = 0.01
                        pos_z = TRAY_THICK + (item_z / 2.0) + 0.005
                        
                        self.saved_relative_pos[env_idx_int, obj_idx] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                        
                        # [注意] 这里使用的是 item_ori (即 fixed_ori)
                        quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                        self.saved_relative_quat[env_idx_int, obj_idx] = quat

                        self.obj_to_source_id[env_idx_int, obj_idx] = 0 
                        self.obj_to_target_id[env_idx_int, obj_idx] = 0 
                        if hasattr(self.cfg, "obj_to_tray_id"):
                            self.cfg.obj_to_tray_id[env_idx_int, obj_idx] = tray_idx

            # ---------------------------------------------------
            # 分支 B: 无托盘模式 (No Tray Mode)
            # ---------------------------------------------------
            else:
                num_to_sample = min(n_active_skus, self.num_skus)
                selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()
                
                distractor_sku_idx = selected_sku_indices[0]
                target_sku_indices = selected_sku_indices[1:]
                
                # --- 预先为本环境的每种 SKU 决定一个朝向 ---
                sku_orientation_map = {} # 字典：sku_name -> (r, p, y)

                # 处理干扰物 SKU
                d_name = self.sku_names[distractor_sku_idx]
                # 随机选一次，存入字典
                _, _, _, d_ori = get_real_dims_local(d_name, force_ori=None) 
                sku_orientation_map[d_name] = d_ori 

                # 处理所有目标物 SKU
                for sku_idx in target_sku_indices:
                    t_name = self.sku_names[sku_idx]
                    # 随机选一次，存入字典
                    _, _, _, t_ori = get_real_dims_local(t_name, force_ori=None) 
                    sku_orientation_map[t_name] = t_ori 

                # --- 收集候选物体 ---
                candidates = []
                
                # A. 干扰物候选
                d_indices = self.sku_to_indices[d_name]
                k_d = torch.randint(1, min(len(d_indices), m_max_per_sku) + 1, (1,)).item()
                sel_d = torch.tensor(d_indices)[torch.randperm(len(d_indices))[:k_d]]
                for idx in sel_d: candidates.append((idx.item(), False, d_name))
                
                # B. 目标物候选
                for sku_idx in target_sku_indices:
                    t_name = self.sku_names[sku_idx]
                    t_indices = self.sku_to_indices[t_name]
                    k_t = torch.randint(1, min(len(t_indices), m_max_per_sku) + 1, (1,)).item()
                    sel_t = torch.tensor(t_indices)[torch.randperm(len(t_indices))[:k_t]]
                    for idx in sel_t: candidates.append((idx.item(), True, t_name))
                
                # 打乱放置顺序（可选，保持随机性）
                #random.shuffle(candidates)
                
                # --- 初始化 Occupancy Map ---
                grid_H = int(BOX_X_LEN / resolution)
                grid_W = int(BOX_Y_LEN / resolution)
                env_map = np.zeros((grid_H, grid_W), dtype=bool)
                border = 1
                env_map[:border, :] = 1; env_map[-border:, :] = 1
                env_map[:, :border] = 1; env_map[:, -border:] = 1
                GAP = 0.005
                
                # --- 尝试逐个放置 ---
                for obj_idx, is_target, obj_name in candidates:
                    # 从字典里取出该 SKU 预定的朝向
                    fixed_ori = sku_orientation_map[obj_name]

                    # 传入 force_ori，强制使用预定朝向
                    item_x, item_y, item_z, item_ori = get_real_dims_local(obj_name, force_ori=fixed_ori)
                    
                    # ... (后续放置逻辑保持不变) ...
                    total_w = item_x + GAP
                    total_h = item_y + GAP
                    obj_w_grid = int(math.ceil(total_w / resolution))
                    obj_h_grid = int(math.ceil(total_h / resolution))
                    obj_w_grid = min(obj_w_grid, grid_H)
                    obj_h_grid = min(obj_h_grid, grid_W)
                    
                    max_x = grid_H - obj_w_grid
                    max_y = grid_W - obj_h_grid
                    
                    placed = False
                    for rx in range(0, max_x + 1, 2): 
                        for ry in range(0, max_y + 1, 2):
                            if not np.any(env_map[rx : rx + obj_w_grid, ry : ry + obj_h_grid]):
                                env_map[rx : rx + obj_w_grid, ry : ry + obj_h_grid] = 1
                                placed = True
                                
                                center_offset_x = (obj_w_grid * resolution) / 2.0
                                center_offset_y = (obj_h_grid * resolution) / 2.0
                                pos_x = (rx * resolution) - (BOX_X_LEN / 2.0) + center_offset_x
                                pos_y = (ry * resolution) - (BOX_Y_LEN / 2.0) + center_offset_y
                                z_pos = (item_z / 2.0) + 0.015 + 0.002
                                
                                self.saved_relative_pos[env_idx_int, obj_idx] = torch.tensor([pos_x, pos_y, z_pos], device=self.device)
                                
                                # 使用 fixed_ori (item_ori)
                                quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                                self.saved_relative_quat[env_idx_int, obj_idx] = quat
                                
                                self.obj_to_source_id[env_idx_int, obj_idx] = 0
                                self.obj_to_target_id[env_idx_int, obj_idx] = 0 if is_target else -1
                                break
                        if placed: break
                    
                    if not placed:
                        self.obj_to_source_id[env_idx_int, obj_idx] = -1
                        self.obj_to_target_id[env_idx_int, obj_idx] = -1

        # 更新 Masks
        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)


    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
            
        num_envs = len(env_ids)
        
        # 1. 预处理：重置所有物体到远处 (保持不变)
        for obj_idx, obj_asset in enumerate(self.object_assets):
            far_position = torch.zeros((num_envs, 3), device=self.device)
            far_position[:, 0] = 100.0 + obj_idx * 10.0
            far_position[:, 1] = 100.0
            quat = torch.zeros((num_envs, 4), device=self.device)
            quat[:, 0] = 1.0 
            set_asset_position(self.env, env_ids, obj_asset, far_position, quat)

        tray_config = getattr(self.cfg, "tray_or_not", [0, 0, 0])
        is_tray_mode = tray_config[0] == 1

        # 2. 执行生成
        # 遍历每一个物体
        for obj_idx, obj_asset in enumerate(self.object_assets):
            
            # 找到哪些环境需要生成这个物体 (source_id != -1)
            # obj_to_source_id 在 assign 阶段已经设置好了 (-1 的就是被截断的)
            assigned_box_indices = self.obj_to_source_id[env_ids, obj_idx]
            
            # 我们只需要处理 source_id = 0 的情况 (假设目前只有 Box 0)
            # 实际上，只要不等于 -1 就说明需要生成
            mask = (assigned_box_indices != -1)
            if not mask.any(): continue
            
            active_env_ids = env_ids[mask]
            
            # 从预存的 Tensor 中直接读取 位置 和 旋转
            # self.saved_relative_pos shape: (num_envs, num_objects, 3)
            # 我们需要取出 [active_env_ids, obj_idx] 对应的数据
            target_pos = self.saved_relative_pos[active_env_ids, obj_idx]
            target_quat = self.saved_relative_quat[active_env_ids, obj_idx]
            
            # 3. 决定 Reference Asset (参考系)
            # 这点很关键！Assign 阶段计算的坐标是相对于谁的？
            
            if is_tray_mode:
                # 在 Assign 阶段，托盘模式计算的是相对于 "Tray" 的坐标
                # 问题：不同的环境，Tray 的名字不一样 (tray_0 vs tray_1)
                # 且同一个物体在不同环境可能分到不同 Tray
                # 所以我们不能批量调用 set_asset_relative_position (因为它只能接受一个 reference)
                
                # === 解决方案 ===
                # 我们需要按 "Tray ID" 分组处理
                current_tray_ids = self.cfg.obj_to_tray_id[active_env_ids, obj_idx]
                
                for tid in [0, 1]: # Tray 0 和 Tray 1
                    sub_mask = (current_tray_ids == tid)
                    if not sub_mask.any(): continue
                    
                    sub_env_ids = active_env_ids[sub_mask]
                    sub_pos = target_pos[sub_mask]
                    sub_quat = target_quat[sub_mask]
                    
                    # 获取参考系对象
                    # box_idx 默认为 0，所以是 tray_0 或 tray_1
                    tray_asset_name = f"tray_{tid}" 
                    target_tray_asset = self.env.scene[tray_asset_name]
                    
                    set_asset_relative_position(
                        env=self.env,
                        env_ids=sub_env_ids,
                        target_asset=obj_asset,
                        reference_asset=target_tray_asset, # 参考系：Tray
                        relative_pos=sub_pos,
                        relative_quat=sub_quat
                    )
            
            else:
                # 无托盘模式，Assign 阶段计算的是相对于 "Box" 的坐标
                # 假设都在 Box 0
                box_asset = self.source_box_assets[0] 
                
                set_asset_relative_position(
                    env=self.env,
                    env_ids=active_env_ids,
                    target_asset=obj_asset,
                    reference_asset=box_asset, # 参考系：Box
                    relative_pos=target_pos,
                    relative_quat=target_quat
                )

    def _update_spawn_metrics(self):
        pass

    def _update_command(self):
        pass

    def command(self):
        pass