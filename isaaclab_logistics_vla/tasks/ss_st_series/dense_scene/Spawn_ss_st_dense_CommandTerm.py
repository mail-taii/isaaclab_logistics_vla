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

class Spawn_ss_st_dense_CommandTerm(BaseOrderCommandTerm):

    
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
        #self.obj_to_target_id[env_ids] = -1
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

        BOX_X_LEN = WORK_BOX_PARAMS['X_LENGTH']
        BOX_Y_LEN = WORK_BOX_PARAMS['Y_LENGTH']
        TRAY_X_LEN = 0.30
        TRAY_Y_LEN = 0.23
        resolution = 0.005

        # --- 内部辅助函数: 支持传入 force_ori ---
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

            # ---------------------------------------------------
            # 分支 A: 托盘模式
            # ---------------------------------------------------
            if is_tray_scene:
                num_trays = 2
                num_to_sample = min(num_trays, self.num_skus) # 先选出要用到的 SKU num
                selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist() # 从所有 SKU 中随机选 num_to_sample 个

                for tray_idx, sku_idx in enumerate(selected_sku_indices):
                    sku_name = self.sku_names[sku_idx]
                    global_indices = self.sku_to_indices[sku_name]
                    
                    # 托盘模式相对简单，我们在计算容量前先确定好朝向
                    # 调用一次 helper，不传参，让它随机选一个，然后存下来
                    _, _, _, fixed_ori = get_real_dims_local(sku_name, force_ori=None)

                    # 再次调用 helper，这次强制使用刚才选好的 fixed_ori
                    item_x, item_y, item_z, item_ori = get_real_dims_local(sku_name, force_ori=fixed_ori)
                    
                    # ... (后续容量计算逻辑保持不变) ...
                    margin = 0.01 # 物品间隙
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
                        pos_z = TRAY_THICK + (item_z / 2.0) + 0.02
                        
                        
                        self.saved_relative_pos[env_idx_int, obj_idx] = torch.tensor([pos_x, pos_y, pos_z], device=self.device)

                        if env_idx_int == 0 and rank == 0: 
                            print("-" * 40)
                            print(f"[Debug Pos] 当前物品: {sku_name}")
                            print(f"[Debug Pos] 物品真实尺寸 (X, Y, Z): ({item_x:.4f}, {item_y:.4f}, {item_z:.4f})")
                            print(f"[Debug] 物品 {sku_name} 计算出的相对坐标: X={pos_x:.3f}, Y={pos_y:.3f}, Z={pos_z:.3f}")
                            print(f"[Debug Pos] 托盘内部可用空间 (inner_x, inner_y): ({inner_x:.4f}, {inner_y:.4f})")
                            print(f"[Debug Pos] 网格左下角起点 (start_x_local, start_y_local): ({start_x_local:.4f}, {start_y_local:.4f})")
                            print("-" * 40)

                        
                        
                        # [注意] 这里使用的是 item_ori (即 fixed_ori)
                        quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                        self.saved_relative_quat[env_idx_int, obj_idx] = quat

                        self.obj_to_source_id[env_idx_int, obj_idx] = 0 
                        #self.obj_to_target_id[env_idx_int, obj_idx] = 0 
                        if hasattr(self.cfg, "obj_to_tray_id"):
                            self.cfg.obj_to_tray_id[env_idx_int, obj_idx] = tray_idx

            # ---------------------------------------------------
            # 分支 B: 无托盘模式 (No Tray Mode)
            # ---------------------------------------------------
            else:
                num_to_sample = min(n_active_skus, self.num_skus)
                selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()
                
                #distractor_sku_idx = selected_sku_indices[0]
                #target_sku_indices = selected_sku_indices[1:]
                
                # --- 预先为本环境的每种 SKU 决定一个朝向 ---
                sku_orientation_map = {} # 字典：sku_name -> (r, p, y)

                # 处理所有selected SKU
                for sku_idx in selected_sku_indices:
                    sku_name = self.sku_names[sku_idx]
                    # 随机选一次朝向，存入字典
                    _, _, _, sku_ori = get_real_dims_local(sku_name, force_ori=None) 
                    sku_orientation_map[sku_name] = sku_ori

                # --- 收集候选物体 ---
                candidates = []
                
                # 统一为所有被选中的 SKU 抽取对应数量的物理实例
                for sku_idx in selected_sku_indices:
                    sku_name = self.sku_names[sku_idx]
                    
                    # 获取该种类在当前环境下的所有可用实例索引
                    # (注: 如果你沿用了前面的多原料箱防重用逻辑，这里应该是 available_sku_indices[sku_name])
                    curr_indices = self.sku_to_indices[sku_name] 
                    
                    # 随机决定当前 SKU 要放几个 (1 到 m_max_per_sku 之间)
                    k = torch.randint(1, min(len(curr_indices), m_max_per_sku) + 1, (1,)).item()
                    sel_instances = torch.tensor(curr_indices)[torch.randperm(len(curr_indices))[:k]]
                    
                    for idx in sel_instances: 
                        # [修改] 统一将中间的布尔值 (之前代表 is_target) 设为 True 或默认状态
                        # 保持三元组结构 (idx, is_target, sku_name) 不变，以免破坏下方解包逻辑
                        candidates.append((idx.item(), True, sku_name))
                
                # 打乱放置顺序（可选，保持随机性）
                #random.shuffle(candidates)
                
                # --- 初始化 Occupancy Map ---
                grid_H = int(BOX_X_LEN / resolution)
                grid_W = int(BOX_Y_LEN / resolution)
                env_map = np.zeros((grid_H, grid_W), dtype=bool)
                border = 1
                env_map[:border, :] = 1; env_map[-border:, :] = 1
                env_map[:, :border] = 1; env_map[:, -border:] = 1
                GAP = 0.01 # 
                
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
                                z_pos = (item_z / 2.0) + 0.015 + 0.02
                                
                                self.saved_relative_pos[env_idx_int, obj_idx] = torch.tensor([pos_x, pos_y, z_pos], device=self.device)
                                
                                # 使用 fixed_ori (item_ori)
                                quat = euler_to_quat_isaac(item_ori[0], item_ori[1], item_ori[2])
                                self.saved_relative_quat[env_idx_int, obj_idx] = quat
                                
                                self.obj_to_source_id[env_idx_int, obj_idx] = 0
                                #self.obj_to_target_id[env_idx_int, obj_idx] = 0 if is_target else -1
                                break
                        if placed: break
                    
                    if not placed:
                        self.obj_to_source_id[env_idx_int, obj_idx] = -1
                        #self.obj_to_target_id[env_idx_int, obj_idx] = -1

        # 更新 Masks
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
        # 单订单情况，所以目标箱索引恒为 0 (target_idx = 0)
        target_idx = 0

        # 先将该目标箱的所有需求清零，方便后续覆盖写入
        self.target_need_sku_num[:, target_idx, :] = 0

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

            # 生成 0 ~ max_available 之间的随机整数作为需求量
            # 原理：torch.rand 产生 [0, 1) 的随机浮点数，乘以 (max+1) 后向下取整，恰好是 [0, max] 的整数
            rand_vals = torch.rand(self.num_envs, device=self.device)
            need_num = (rand_vals * (max_available + 1).float()).long()

            # 将计算出的随机需求量赋值给订单箱
            self.target_need_sku_num[:, target_idx, sku_idx] = need_num


        # =========================================================
        # 3. 防止出现“空订单”
        # =========================================================
        # 因为上面的算法允许需要 0 个物品，极端情况下某环境可能所有 SKU 都恰好被随机成了 0，导致该局无需抓取直接结束。
        # 下面这段逻辑用于：如果某环境订单总量为 0，但场景里确实有物品，则强行随机指定 1 个需要的物品。

        # 计算每个环境需要的物品总数 shape: (num_envs,)
        total_needed = self.target_need_sku_num[:, target_idx, :].sum(dim=1)

        # 找出空订单的环境
        empty_order_envs = (total_needed == 0)

        if empty_order_envs.any():
            # 找出每个环境总共生成了多少可用物体
            total_available_all_skus = (self.obj_to_source_id != -1).sum(dim=1)

            # 只有那些“场景里有东西，但订单却是空”的环境才需要被修正
            needs_fix = empty_order_envs & (total_available_all_skus > 0)

            if needs_fix.any():
                # 获取需要修正的环境的索引
                fix_env_indices = needs_fix.nonzero(as_tuple=True)[0]

                for env_i in fix_env_indices:
                    # 找到该环境里数量 > 0 的有效 SKU 种类
                    available_sku_mask = (self.obj_to_source_id[env_i] != -1)
                    # 逆向反推该环境拥有的有效 SKU indices (直接遍历给某一个 +1)
                    for s_idx, s_name in enumerate(self.sku_names):
                        s_indices = self.sku_to_indices[s_name]
                        if (self.obj_to_source_id[env_i, s_indices] != -1).any():
                            # 只要发现这个 SKU 在场上有存货，就强行让订单要 1 个，然后跳出
                            self.target_need_sku_num[env_i, target_idx, s_idx] = 1
                            break


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
        """
        利用 object_states 的坐标判定结果，计算全局 Metrics
        """
        target_idx = 0 
        
        # =========================================================
        # 1. 直接读取已更新好的需求矩阵与实际包含矩阵
        # =========================================================
        # shape: (num_envs, num_skus)
        target_needs = self.target_need_sku_num[:, target_idx, :] 
        actual_in_target = self.target_contain_sku_num[:, target_idx, :]

        # =========================================================
        # 2. 核心得分项计算
        # =========================================================
        # A. 正确抓取量：实际放进去的，且没有超过需求上限的部分 (多放的不算正分)
        correct_picks = torch.minimum(actual_in_target, target_needs).sum(dim=1)
        
        # B. 错误/多余抓取量：放错了 SKU，或者放对了 SKU 但超过了需求数量
        wrong_picks = torch.clamp(actual_in_target - target_needs, min=0).sum(dim=1)
        
        # C. 物理掉落计数：状态被标记为 10 的物品总数
        dropped_count = (self.object_states == 10).sum(dim=1)
        
        # D. 订单总需求量
        total_needed = target_needs.sum(dim=1)

        # =========================================================
        # 3. 最终复合指标导出
        # =========================================================
        # 订单完成度: 0.0 ~ 1.0 (防止 total_needed 为 0 导致除以 0 报错)
        completion_rate = torch.where(
            total_needed > 0,
            correct_picks.float() / total_needed.float(),
            torch.tensor(1.0, device=self.device) 
        )
        
        # 完美成功标志: 需求全满，且没有任何错拿杂物
        is_success = (correct_picks == total_needed) & (wrong_picks == 0)

        # 组装 Metrics 字典，供 Reward(奖励函数) 或 Logger(日志) 直接读取
        self.metrics = {
            "completion_rate": completion_rate,       # float: [0.0, 1.0] 适合做 Dense Reward
            "wrong_pick_count": wrong_picks,          # long: 错拿个数 适合做大惩罚项
            "dropped_count": dropped_count,           # long: 掉落个数 适合触发 done 截断
            "is_success": is_success.float(),         # float: 1.0/0.0 适合做最终评估指标
            "correct_picks": correct_picks,
            "total_needed": total_needed
        }
        
        return self.metrics

    def _update_command(self):
        pass

    def command(self):
        pass

    def __str__(self) -> str:
        return "ss_st_dense"