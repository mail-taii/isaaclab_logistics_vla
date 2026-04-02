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

from isaaclab_logistics_vla.tasks.ms_mt_series.Assign_ms_mt_CommandTerm import AssignMSMTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg

class Spawn_ms_mt_dense_CommandTerm(AssignMSMTCommandTerm):

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

        #self.obj_to_target_id[env_ids] = -1
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
        num_target_box = getattr(self.cfg, "num_target_box", 2)

        BOX_X_LEN = WORK_BOX_PARAMS['X_LENGTH']
        BOX_Y_LEN = WORK_BOX_PARAMS['Y_LENGTH']
        TRAY_X_LEN = 0.30
        TRAY_Y_LEN = 0.23
        resolution = 0.005

        def get_real_dims_local(obj_name, force_ori=None):
            # 1. 直接从全局字典 SKU_CONFIG 中精准查找配置
            if obj_name in SKU_CONFIG:
                p = SKU_CONFIG[obj_name]
            else:
                print(f"[警告] 字典中找不到物品: {obj_name}，将使用默认兜底参数！")
                # 兜底方案：随便取字典里的第一个，防止程序直接崩溃
                p = next(iter(SKU_CONFIG.values())) 

            # 2. 获取基础长宽高
            raw_dims = torch.tensor([p['X_LENGTH'], p['Y_LENGTH'], p['Z_LENGTH']], device=self.device)
            
            # 3. 处理缩放 (优先使用配置表自带的 DENSE_SCALE)
            dense_scale = p.get('DENSE_SCALE', 1.0)
            raw_dims *= dense_scale
            
            # 4. 处理朝向
            if force_ori is not None:
                # 情况 A: 强制使用传入的朝向
                ori_deg = force_ori
            else:
                # 情况 B: 没有强制，则查表获取选项并决定
                ori_deg_options = p.get('DENSE_ORIENT', (10, 10, 10))
                print(ori_deg_options)
                
                if isinstance(ori_deg_options, list):
                    # 如果有多个朝向选项（如 [(0,90,0), (0,0,0)]），随机选一个
                    idx = torch.randint(0, len(ori_deg_options), (1,)).item()
                    ori_deg = ori_deg_options[idx]
                else:
                    # 如果只有一个元组（如 (0,0,0)）
                    ori_deg = ori_deg_options

            # 5. 计算旋转后的真实包围盒尺寸
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
                            pos_z = TRAY_THICK + (item_z / 2.0) + 0.02
                            
                            self.saved_relative_pos[env_idx_int, obj_idx] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)
                            quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                            self.saved_relative_quat[env_idx_int, obj_idx] = quat

                            self.obj_to_source_id[env_idx_int, obj_idx] = source_box_idx 
                            #self.obj_to_target_id[env_idx_int, obj_idx] = 0 
                            
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
            
                    
                    sku_orientation_map = {}

                    for sku_idx in selected_sku_indices:
                        sku_name = self.sku_names[sku_idx]
                        _, _, _, sku_ori = get_real_dims_local(sku_name, force_ori=None) 
                        sku_orientation_map[sku_name] = sku_ori

                    candidates = []
                    
                    for sku_idx in selected_sku_indices:
                        sku_name = self.sku_names[sku_idx]
                        curr_indices = available_sku_indices[sku_name] 
                        
                        actual_max = min(len(curr_indices), m_max_per_sku)
                        
                        if actual_max > 0: 
                            # 随机决定抽取数量 k
                            k = torch.randint(1, actual_max + 1, (1,)).item()
                            sel_instances = torch.tensor(curr_indices)[torch.randperm(len(curr_indices))[:k]].tolist()
                            
                            for idx in sel_instances: 
                                # 将中间的标志位统一设为 True，保持 (idx, is_target, name) 结构不破坏下方的解包
                                candidates.append((idx, True, sku_name))
                                # 从池中移除，防止后续箱子用到同一个物理实例
                                available_sku_indices[sku_name].remove(idx)
                    
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
                                    z_pos = (item_z / 2.0) + 0.015 + 0.02
                                    
                                    self.saved_relative_pos[env_idx_int, obj_idx] = torch.tensor([pos_x, pos_y, z_pos], device=self.device)
                                    quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                                    self.saved_relative_quat[env_idx_int, obj_idx] = quat
                                    
                                    self.obj_to_source_id[env_idx_int, obj_idx] = source_box_idx
                                    #self.obj_to_target_id[env_idx_int, obj_idx] = 0 if is_target else -1
                                    break
                            if placed: break

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        #self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)

                # =========================================================
        # 1. 填充 object_states (物品当前状态)
        # =========================================================
        # 规则说明：
        # obj_to_source_id: -1(不生成), 0(原料箱1), 1(原料箱2), 2(原料箱3)
        # object_states:   -1(不出现),  1(原料箱1), 2(原料箱2), 3(原料箱3)
        # 可见规律是：只要 obj_to_source_id != -1，它的状态就是 obj_to_source_id + 1

        self.object_states = torch.where(
            self.obj_to_source_id != -1,
            self.obj_to_source_id + 1,  # 如果在原料箱中，状态值 = source_id + 1
            torch.tensor(-1, dtype=torch.long, device=self.device)  # 否则保持 -1
        )

        # =========================================================
        # 2. 填充 target_need_sku_num (订单需求量)
        # =========================================================

        # 先将该目标箱的所有需求清零，方便后续覆盖写入
        self.target_need_sku_num[:, :, :] = 0

        # 遍历每种 SKU，统计在各个环境的原料箱中实际生成了多少个
        for sku_idx, sku_name in enumerate(self.sku_names):
            # 取出该 SKU 对应的所有物理实例在全局 object_assets 中的索引
            curr_indices = self.sku_to_indices[sku_name]
            curr_indices_tensor = torch.tensor(curr_indices, dtype=torch.long, device=self.device)

            # 提取这些实例的 obj_to_source_id，判断它们是否被生成 (!= -1)
            # valid_mask 的 shape: (num_envs, 当前SKU的实例总数)
            valid_mask = (self.obj_to_source_id[:, curr_indices_tensor] != -1)

            # 按环境(dim=1)求和，得出这个 SKU 在每个环境里真正可用的最大库存量
            # max_available 的 shape: (num_envs,)
            max_available = valid_mask.sum(dim=1)

            # 为每个目标箱独立生成随机需求
            for target_idx in range(num_target_box):
                rand_vals = torch.rand(self.num_envs, device=self.device)
                
                if target_idx == num_target_box - 1:
                    # 如果是最后一个目标箱，直接把剩余的库存全分配掉，避免出现“空订单”
                    need_num = (rand_vals * (max_available + 1).float()).long()
                else:
                    need_num = (rand_vals * (max_available).float()).long()
                # 保证需求量不为负数，且不超过当前剩余库存
                need_num = torch.clamp(need_num, min=0)
                need_num = torch.minimum(need_num, max_available)
                
                self.target_need_sku_num[:, target_idx, sku_idx] = need_num
                
                # 扣除已分配的数量，剩下的才能给下一个订单箱
                max_available -= need_num


        # =========================================================
        # 3. 防止出现“空订单”
        # =========================================================
        # 因为上面的算法允许需要 0 个物品，极端情况下某环境可能所有 SKU 都恰好被随机成了 0，导致该局无需抓取直接结束。
        # 下面这段逻辑用于：如果某环境订单总量为 0，但场景里确实有物品，则强行随机指定 1 个需要的物品。

        # =========================================================
        # 3. 防止出现“空订单” (支持多订单箱)
        # =========================================================
        for target_idx in range(num_target_box):
            total_needed = self.target_need_sku_num[:, target_idx, :].sum(dim=1)
            empty_order_envs = (total_needed == 0)

            if empty_order_envs.any():
                total_available_all_skus = (self.obj_to_source_id != -1).sum(dim=1)
                needs_fix = empty_order_envs & (total_available_all_skus > 0)

                if needs_fix.any():
                    fix_env_indices = needs_fix.nonzero(as_tuple=True)[0]
                    for env_i in fix_env_indices:
                        available_sku_mask = (self.obj_to_source_id[env_i] != -1)
                        for s_idx, s_name in enumerate(self.sku_names):
                            s_indices = self.sku_to_indices[s_name]
                            if (self.obj_to_source_id[env_i, s_indices] != -1).any():
                                # 强制给当前为空的订单箱塞 1 个需求
                                self.target_need_sku_num[env_i, target_idx, s_idx] += 1
                                break

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
        """
        利用 object_states 的坐标判定结果，计算多订单箱全局 Metrics
        """
        # 1. 直接获取完整的 3D 张量 (num_envs, num_targets, num_skus)
        target_needs = self.target_need_sku_num
        actual_in_target = self.target_contain_sku_num

        # 2. 核心得分计算 (注意这里维度用 dim=-1，代表在 SKU 维度上求和)
        correct_picks_per_target = torch.minimum(actual_in_target, target_needs).sum(dim=-1)
        wrong_picks_per_target = torch.clamp(actual_in_target - target_needs, min=0).sum(dim=-1)
        dropped_count = (self.object_states == 10).sum(dim=1)
        total_needed_per_target = target_needs.sum(dim=-1)

        # 3. 导出每个订单箱独立指标与全局综合指标
        completion_rate_per_target = torch.where(
            total_needed_per_target > 0,
            correct_picks_per_target.float() / total_needed_per_target.float(),
            torch.tensor(1.0, device=self.device) 
        )
        
        # 全局各项总和
        total_correct_picks = correct_picks_per_target.sum(dim=-1)
        total_needed_all = total_needed_per_target.sum(dim=-1)
        total_wrong_picks = wrong_picks_per_target.sum(dim=-1)
        
        global_completion_rate = torch.where(
            total_needed_all > 0,
            total_correct_picks.float() / total_needed_all.float(),
            torch.tensor(1.0, device=self.device)
        )
        
        is_success_per_target = (correct_picks_per_target == total_needed_per_target) & (wrong_picks_per_target == 0)
        global_is_success = is_success_per_target.all(dim=-1)

        self.metrics = {
            "completion_rate_per_target": completion_rate_per_target, 
            "global_completion_rate": global_completion_rate,
            "wrong_pick_count": total_wrong_picks,          
            "dropped_count": dropped_count,           
            "is_success": global_is_success.float(),
            "correct_picks": total_correct_picks,
            "total_needed": total_needed_all
        }
        
        return self.metrics

    def _update_command(self):
        pass

    def command(self):
        pass