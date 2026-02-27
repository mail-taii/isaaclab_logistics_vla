from __future__ import annotations

import torch
import math
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab_logistics_vla.tasks.ms_st_series.Assign_ms_st_CommandTerm import AssignMSSTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg

class Spawn_ms_st_dense_CommandTerm(AssignMSSTCommandTerm):

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

        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        
        if not hasattr(self, "saved_relative_pos"):
            self.saved_relative_pos = torch.zeros((self.num_envs, self.num_objects, 3), device=self.device)
            self.saved_relative_quat = torch.zeros((self.num_envs, self.num_objects, 4), device=self.device)
            self.saved_relative_quat[..., 0] = 1.0 
        
        self.saved_relative_pos[env_ids] = 0
        self.saved_relative_quat[env_ids] = 0
        self.saved_relative_quat[env_ids, :, 0] = 1.0

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 6)
        tray_config = getattr(self.cfg, "tray_or_not", [0, 0, 0])
        is_tray_scene = tray_config
        
        num_source_box = getattr(self.cfg, "num_source_box", 2) 

        BOX_X_LEN = WORK_BOX_PARAMS['X_LENGTH']
        BOX_Y_LEN = WORK_BOX_PARAMS['Y_LENGTH']
        TRAY_X_LEN = 0.30
        TRAY_Y_LEN = 0.23
        resolution = 0.005
        
        SCALED_OBJECTS1= ["CN_big", "SF_small", "empty_plastic_package", "SF_big"]
        SCALE_FACTOR1= torch.tensor([0.3, 0.3, 0.3], device=self.device)
        SCALED_OBJECTS2= ["cracker_box","sugar_box","tomato_soup_can"]
        SCALE_FACTOR2= torch.tensor([0.8, 0.8, 0.8], device=self.device)

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
            if any(s in obj_name for s in SCALED_OBJECTS1): 
                raw_dims *= SCALE_FACTOR1
            elif any(s in obj_name for s in SCALED_OBJECTS2): 
                raw_dims *= SCALE_FACTOR2
            
            if force_ori is not None:
                ori_deg = force_ori
            else:
                ori_deg_options = p.get('DENSE_ORIENT', (0, 0, 0)) 
                if isinstance(ori_deg_options, list):
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

            # 物理实例池：保证同一个SKU可以分到不同的箱子，但不复用同一个物理刚体ID
            available_sku_indices = {
                sku: list(self.sku_to_indices[sku]) for sku in self.sku_names
            }

            # 遍历每个原料箱，每个原料箱独立计算
            for source_box_idx in range(num_source_box):

                # ---------------------------------------------------
                # 分支 A: 托盘模式
                # ---------------------------------------------------
                if is_tray_scene[source_box_idx]:
                    num_trays = 2
                    num_to_sample = min(num_trays, self.num_skus)
                    # 【核心需求体现】：每个原料箱独立随机抽取自己想要的 SKU
                    selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

                    for tray_idx, sku_idx in enumerate(selected_sku_indices):
                        sku_name = self.sku_names[sku_idx]
                        
                        _, _, _, fixed_ori = get_real_dims_local(sku_name, force_ori=None)
                        item_x, item_y, item_z, item_ori = get_real_dims_local(sku_name, force_ori=fixed_ori)
                        
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
                        
                        # 获取当前SKU在全局池子中还剩几个物理实例
                        curr_indices = available_sku_indices[sku_name]
                        actual_max = min(max_capacity, len(curr_indices))
                        
                        if actual_max < 1: 
                            continue  # 如果前面箱子把该SKU耗尽了，当前箱子跳过
                        else: 
                            k = torch.randint(1, actual_max + 1, (1,)).item()
                            
                        selected_objs = torch.tensor(curr_indices)[torch.randperm(len(curr_indices))[:k]].tolist()
                        
                        for rank, obj_idx in enumerate(selected_objs):
                            available_sku_indices[sku_name].remove(obj_idx)

                            grid_x = rank % cols
                            grid_y = rank // cols
                            start_x_local = -inner_x / 2.0 + eff_item_x / 2.0
                            start_y_local = -inner_y / 2.0 + eff_item_y / 2.0
                            pos_x = start_x_local + grid_x * eff_item_x
                            pos_y = start_y_local + grid_y * eff_item_y
                            
                            TRAY_THICK = 0.01
                            pos_z = TRAY_THICK + (item_z / 2.0) + 0.005
                            
                            self.saved_relative_pos[env_idx_int, obj_idx] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                            quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                            self.saved_relative_quat[env_idx_int, obj_idx] = quat

                            self.obj_to_source_id[env_idx_int, obj_idx] = source_box_idx 
                            self.obj_to_target_id[env_idx_int, obj_idx] = 0 
                            
                            if hasattr(self.cfg, "obj_to_tray_id"):
                                global_tray_idx = source_box_idx * num_trays + tray_idx
                                self.cfg.obj_to_tray_id[env_idx_int, obj_idx] = global_tray_idx

                # ---------------------------------------------------
                # 分支 B: 无托盘模式 (No Tray Mode)
                # ---------------------------------------------------
                else:
                    num_to_sample = min(n_active_skus, self.num_skus)
                    # 【核心需求体现】：每个原料箱独立随机抽取自己想要的 SKU
                    selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()
                    
                    distractor_sku_idx = selected_sku_indices[0]
                    target_sku_indices = selected_sku_indices[1:]
                    
                    sku_orientation_map = {}

                    d_name = self.sku_names[distractor_sku_idx]
                    _, _, _, d_ori = get_real_dims_local(d_name, force_ori=None) 
                    sku_orientation_map[d_name] = d_ori 

                    for sku_idx in target_sku_indices:
                        t_name = self.sku_names[sku_idx]
                        _, _, _, t_ori = get_real_dims_local(t_name, force_ori=None) 
                        sku_orientation_map[t_name] = t_ori 

                    candidates = []
                    
                    # A. 干扰物候选
                    d_indices = available_sku_indices[d_name] 
                    max_d = min(len(d_indices), m_max_per_sku)
                    if max_d > 0: 
                        k_d = torch.randint(1, max_d + 1, (1,)).item()
                        sel_d = torch.tensor(d_indices)[torch.randperm(len(d_indices))[:k_d]].tolist()
                        for idx in sel_d: 
                            candidates.append((idx, False, d_name))
                            available_sku_indices[d_name].remove(idx) 
                    
                    # B. 目标物候选
                    for sku_idx in target_sku_indices:
                        t_name = self.sku_names[sku_idx]
                        t_indices = available_sku_indices[t_name] 
                        max_t = min(len(t_indices), m_max_per_sku)
                        if max_t > 0: 
                            k_t = torch.randint(1, max_t + 1, (1,)).item()
                            sel_t = torch.tensor(t_indices)[torch.randperm(len(t_indices))[:k_t]].tolist()
                            for idx in sel_t: 
                                candidates.append((idx, True, t_name))
                                available_sku_indices[t_name].remove(idx) 
                    
                    grid_H = int(BOX_X_LEN / resolution)
                    grid_W = int(BOX_Y_LEN / resolution)
                    env_map = np.zeros((grid_H, grid_W), dtype=bool)
                    border = 1
                    env_map[:border, :] = 1; env_map[-border:, :] = 1
                    env_map[:, :border] = 1; env_map[:, -border:] = 1
                    GAP = 0.005
                    
                    for obj_idx, is_target, obj_name in candidates:
                        fixed_ori = sku_orientation_map[obj_name]
                        item_x, item_y, item_z, item_ori = get_real_dims_local(obj_name, force_ori=fixed_ori)
                        
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
                                    quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                                    self.saved_relative_quat[env_idx_int, obj_idx] = quat
                                    
                                    self.obj_to_source_id[env_idx_int, obj_idx] = source_box_idx
                                    self.obj_to_target_id[env_idx_int, obj_idx] = 0 if is_target else -1
                                    break
                            if placed: break

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)


    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
            
        num_envs = len(env_ids)
        num_source_box = getattr(self.cfg, "num_source_box", 2)
        tray_config = getattr(self.cfg, "tray_or_not", [0, 0, 0]) # 取出配置
        
        for obj_idx, obj_asset in enumerate(self.object_assets):
            far_position = torch.zeros((num_envs, 3), device=self.device)
            far_position[:, 0] = 100.0 + obj_idx * 10.0
            far_position[:, 1] = 100.0
            quat = torch.zeros((num_envs, 4), device=self.device)
            quat[:, 0] = 1.0 
            set_asset_position(self.env, env_ids, obj_asset, far_position, quat)

        for obj_idx, obj_asset in enumerate(self.object_assets):
            assigned_box_indices = self.obj_to_source_id[env_ids, obj_idx]
            
            mask = (assigned_box_indices != -1)
            if not mask.any(): continue
            
            # [核心修改]：外层先遍历 box_id，针对每个原料箱独立处理
            for box_id in range(num_source_box):
                box_mask = (assigned_box_indices == box_id)
                if not box_mask.any(): continue
                
                # 提取分配到当前箱子的环境索引和对应的位姿
                sub_env_ids = env_ids[box_mask]
                sub_pos = self.saved_relative_pos[sub_env_ids, obj_idx]
                sub_quat = self.saved_relative_quat[sub_env_ids, obj_idx]

                # 判断当前正在处理的这个 box_id 是否开启了托盘模式
                is_tray_mode = False
                if box_id < len(tray_config):
                    is_tray_mode = (tray_config[box_id] == 1)
                
                if is_tray_mode:
                    # 获取分配到托盘的具体 tray_id
                    current_tray_ids = self.cfg.obj_to_tray_id[sub_env_ids, obj_idx]
                    num_trays = 2 
                    
                    # 遍历属于当前 box_id 的托盘 (例如 box_id=1 对应的就是 tray_2 和 tray_3)
                    for tray_offset in range(num_trays):
                        global_tray_idx = box_id * num_trays + tray_offset
                        tray_mask = (current_tray_ids == global_tray_idx)
                        if not tray_mask.any(): continue
                        
                        sub_sub_env_ids = sub_env_ids[tray_mask]
                        sub_sub_pos = self.saved_relative_pos[sub_sub_env_ids, obj_idx]
                        sub_sub_quat = self.saved_relative_quat[sub_sub_env_ids, obj_idx]
                        
                        tray_asset_name = f"tray_{global_tray_idx}" 
                        try:
                            target_tray_asset = self.env.scene[tray_asset_name]
                        except KeyError:
                            continue 
                        
                        set_asset_relative_position(
                            env=self.env,
                            env_ids=sub_sub_env_ids,
                            target_asset=obj_asset,
                            reference_asset=target_tray_asset,
                            relative_pos=sub_sub_pos,
                            relative_quat=sub_sub_quat
                        )
                
                else:
                    # 如果当前箱子没有开启托盘模式，直接相对 box_asset 生成
                    box_asset = self.source_box_assets[box_id] 
                    
                    set_asset_relative_position(
                        env=self.env,
                        env_ids=sub_env_ids,
                        target_asset=obj_asset,
                        reference_asset=box_asset, 
                        relative_pos=sub_pos,
                        relative_quat=sub_quat
                    )
    def _update_spawn_metrics(self):
        pass

    def _update_command(self):
        pass

    def command(self):
        pass