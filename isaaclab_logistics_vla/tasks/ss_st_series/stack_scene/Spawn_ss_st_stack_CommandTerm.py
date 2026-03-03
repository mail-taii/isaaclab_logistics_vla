from __future__ import annotations

import math
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
from .scene_cfg import SKU_DEFINITIONS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg


# SKU 名称片段 → constant 参数字典的映射（新增 SKU 时在此处加一行即可）
_SKU_PARAMS_MAP = {
    "cracker": CRACKER_BOX_PARAMS,
    "sugar": SUGER_BOX_PARAMS,
    "plastic_package": PLASTIC_PACKAGE_PARAMS,
    "sf_big": SF_BIG_PARAMS,
    "sf_small": SF_SMALL_PARAMS,
}


class Spawn_ss_st_stack_CommandTerm(AssignSSSTCommandTerm):
    """
    堆叠场景的 CommandTerm（含冗余物品）

    设计要点：
    1. 箱内分为 4 个槽位（2×2 网格），每个槽位可放一摞
    2. 随机选 n_stacks 摞（1~max_stacks），随机选 m 种目标 SKU
    3. 确保订单需求总数 > max_per_stack*(n_stacks-1)，保证必须用到 n_stacks 摞
    4. 物品按 SKU 顺序贪心填充：同种优先填满一摞，不足时下一种接续
    5. 每摞内按底面积从大到小排列（底部面积最大，顶部面积最小）
    6. n 个槽位放摞，随机分配槽位；放置时加入轻微位置抖动和旋转抖动

    冗余物品机制：
    订单需求由 target_need_sku_num 定义。在满足需求之后，会从任意可用 SKU
    （无论是否在订单中）额外随机生成若干冗余物品混入堆叠。
    冗余物品与订单物品的区别完全隐式，由 target_need_sku_num（需求量）与
    is_active_mask（实际生成量）的差值体现，无需额外标记。

    辅助变量 stack_layout:
        shape = [num_envs, num_sources, max_stacks, max_per_stack]
        值为 object index，-1 表示空位
    """

    SCALE = 1.0  # 默认缩放倍率；若 scene_cfg.SKU_DEFINITIONS 中为该 SKU 配置了 scale，则以配置为准

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 预计算堆叠参数缓存
        self._stack_params_cache = self._build_stack_params_cache()

        # 辅助变量：记录每个环境的堆叠布局
        max_stacks = getattr(cfg, 'max_stacks', 4)
        max_per_stack = getattr(cfg, 'max_per_stack', 4)
        self.stack_layout = torch.full(
            (self.num_envs, self.num_sources, max_stacks, max_per_stack),
            -1, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------ #
    #                       堆叠参数缓存                                   #
    # ------------------------------------------------------------------ #

    def _build_stack_params_cache(self) -> dict:
        """构建 object_name → 堆叠参数 的映射"""
        cache = {}
        for obj_name in self.object_names:
            raw = self._get_raw_params(obj_name)
            cache[obj_name] = self._compute_stack_params(raw)
        return cache

    def _get_raw_params(self, obj_name: str) -> dict:
        """
        获取缩放后的 SKU 物理参数（尺寸 + STACK_ORIENT）。
        新增 SKU 时：在 constant.py 加参数字典，在 _SKU_PARAMS_MAP 加映射。
        """
        # 优先从 scene_cfg.SKU_DEFINITIONS 里读取每个 SKU 独立配置的 scale，
        # 若未配置则回退到类属性 SCALE。
        scale = self._get_scale_for_obj(obj_name)
        for key, params in _SKU_PARAMS_MAP.items():
            if key in obj_name:
                return {
                    'X_LENGTH': params['X_LENGTH'] * scale,
                    'Y_LENGTH': params['Y_LENGTH'] * scale,
                    'Z_LENGTH': params['Z_LENGTH'] * scale,
                    'STACK_ORIENT': params['STACK_ORIENT'],
                }
        # 默认回退
        p = CRACKER_BOX_PARAMS
        return {
            'X_LENGTH': p['X_LENGTH'] * scale,
            'Y_LENGTH': p['Y_LENGTH'] * scale,
            'Z_LENGTH': p['Z_LENGTH'] * scale,
            'STACK_ORIENT': p['STACK_ORIENT'],
        }

    def _get_scale_for_obj(self, obj_name: str) -> float:
        """根据物体名称从 scene_cfg.SKU_DEFINITIONS 中获取对应的缩放倍率。"""
        for sku_name, (_usd_path, _count, scale) in SKU_DEFINITIONS.items():
            if sku_name in obj_name:
                return float(scale)
        return float(self.SCALE)

    def _compute_stack_params(self, params: dict) -> dict:
        """根据缩放后尺寸计算 base_area / stack_height / stack_orient"""
        x, y, z = params['X_LENGTH'], params['Y_LENGTH'], params['Z_LENGTH']
        dims = sorted([x, y, z], reverse=True)
        return {
            'base_area': dims[0] * dims[1],      # 最大两维组成底面
            'stack_height': dims[2],              # 最小维度作为堆叠高度
            'stack_orient': params['STACK_ORIENT'],  # 来自 constant.py
        }

    # ------------------------------------------------------------------ #
    #                       物品 / 箱子分配（Assign）                       #
    # ------------------------------------------------------------------ #

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        """
        为每个环境分配物品到原料箱，并规划堆叠布局（含冗余物品）。

        流程：
        1. 随机选一个原料箱，随机选摞数 n_stacks
        2. 选目标 SKU，确定每种的订单需求量 → 写入 target_need_sku_num
        3. 激活需求物品
        4. 从所有尚未激活的实例（任意 SKU）中随机补充冗余物品
        5. 全部物品混合堆叠
        6. 写入 obj_to_source_id / stack_layout / is_active_mask
        """
        self.target_need_sku_num[env_ids] = 0
        self.obj_to_source_id[env_ids] = -1
        self.stack_layout[env_ids] = -1

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)
        max_active_skus = getattr(self.cfg, 'max_active_skus', 5)
        max_redundant = getattr(self.cfg, 'max_redundant', 3)

        for env_id in env_ids:
            env_id_val = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            # --- 1. 随机选一个原料箱 ---
            selected_source_box = torch.randint(0, self.num_sources, (1,)).item()

            # --- 2. 随机选摞数 ---
            n_stacks = torch.randint(1, max_stacks + 1, (1,)).item()

            # --- 3. 选目标 SKU 种类、确定订单需求量 ---
            max_target_skus = min(max_active_skus, self.num_skus, n_stacks * max_per_stack)
            if max_target_skus < 1:
                max_target_skus = 1
            m_target_skus = torch.randint(1, max_target_skus + 1, (1,)).item()
            target_sku_indices = torch.randperm(self.num_skus)[:m_target_skus].tolist()

            available_per_target = []
            for sku_idx in target_sku_indices:
                sku_name = self.sku_names[sku_idx]
                available_per_target.append(len(self.sku_to_indices[sku_name]))
            total_target_available = sum(available_per_target)

            min_demand = max_per_stack * (n_stacks - 1) + 1
            max_demand_total = max_per_stack * n_stacks

            while min_demand > total_target_available and n_stacks > 1:
                n_stacks -= 1
                min_demand = max_per_stack * (n_stacks - 1) + 1
                max_demand_total = max_per_stack * n_stacks

            total_capacity = max_per_stack * n_stacks
            min_demand = max(min_demand, m_target_skus)
            actual_max_demand = min(max_demand_total, total_target_available)
            actual_min_demand = min(min_demand, actual_max_demand)

            if actual_min_demand >= actual_max_demand:
                target_total_demand = actual_max_demand
            else:
                target_total_demand = torch.randint(actual_min_demand, actual_max_demand + 1, (1,)).item()

            demand_counts = [0] * m_target_skus
            remaining = target_total_demand
            for i in range(m_target_skus):
                if remaining <= 0:
                    break
                demand_counts[i] = 1
                remaining -= 1
            while remaining > 0:
                candidates = [i for i in range(m_target_skus) if demand_counts[i] < available_per_target[i]]
                if not candidates:
                    break
                idx = candidates[torch.randint(0, len(candidates), (1,)).item()]
                demand_counts[idx] += 1
                remaining -= 1

            # --- 4. 激活订单需求物品、写 target_need_sku_num ---
            sku_obj_groups = []
            slots_used = 0

            for i, sku_idx in enumerate(target_sku_indices):
                if demand_counts[i] == 0:
                    continue
                self.target_need_sku_num[env_id_val, 0, sku_idx] = demand_counts[i]

                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                perm = torch.randperm(len(global_indices))[:demand_counts[i]]
                selected_objs = [global_indices[p] for p in perm.tolist()]

                for obj_idx in selected_objs:
                    self.obj_to_source_id[env_id_val, obj_idx] = selected_source_box

                sku_obj_groups.append((sku_name, selected_objs))
                slots_used += demand_counts[i]

            # --- 5. 冗余物品：从任意尚未激活的实例中随机补充 ---
            remaining_slots = total_capacity - slots_used
            redundant_budget = min(max_redundant, remaining_slots)

            if redundant_budget > 0:
                n_redundant = torch.randint(0, redundant_budget + 1, (1,)).item()

                if n_redundant > 0:
                    available_pool = [
                        idx for idx in range(self.num_objects)
                        if self.obj_to_source_id[env_id_val, idx].item() == -1
                    ]
                    n_redundant = min(n_redundant, len(available_pool))

                    if n_redundant > 0:
                        perm = torch.randperm(len(available_pool))[:n_redundant]
                        redundant_objs = [available_pool[p] for p in perm.tolist()]

                        for obj_idx in redundant_objs:
                            self.obj_to_source_id[env_id_val, obj_idx] = selected_source_box

                        redundant_by_sku: dict[str, list[int]] = {}
                        for obj_idx in redundant_objs:
                            obj_name = self.object_names[obj_idx]
                            matched_sku = obj_name
                            for sn in self.sku_names:
                                if sn in obj_name:
                                    matched_sku = sn
                                    break
                            redundant_by_sku.setdefault(matched_sku, []).append(obj_idx)

                        for sku_name, objs in redundant_by_sku.items():
                            sku_obj_groups.append((sku_name, objs))

            # --- 6. 按 SKU 顺序贪心填充到 n_stacks 摞 ---
            stacks: list[list[int]] = [[] for _ in range(n_stacks)]
            current_stack = 0

            for _sku_name, obj_indices in sku_obj_groups:
                for obj_idx in obj_indices:
                    if current_stack < n_stacks and len(stacks[current_stack]) >= max_per_stack:
                        current_stack += 1
                    if current_stack >= n_stacks:
                        break
                    stacks[current_stack].append(obj_idx)

            # --- 7. 每摞内按底面积从大到小排序（底部最大，顶部最小）---
            for stack in stacks:
                stack.sort(
                    key=lambda idx: self._stack_params_cache[self.object_names[idx]]['base_area'],
                    reverse=True,
                )

            # --- 8. 写入 stack_layout ---
            for stack_idx, stack in enumerate(stacks):
                for pos, obj_idx in enumerate(stack):
                    self.stack_layout[env_id_val, selected_source_box, stack_idx, pos] = obj_idx

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)

    # ------------------------------------------------------------------ #
    #                       物品生成（Spawn）                               #
    # ------------------------------------------------------------------ #

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """
        根据 stack_layout 将物品摆放到原料箱的槽位中。

        assign 已把摞数、每摞放什么都确定了，这里仅遵照执行：
        - 4 个槽位为 2×2 网格，随机分配给各摞
        - 每摞内从底到顶逐个摆放
        - 加入轻微的 XY 位置抖动和 Z 轴旋转抖动
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # Step 0: 先将所有物品移至远处
        self._move_all_objects_far(env_ids)

        # Step 1: 获取 4 个槽位锚点（2×2 网格）
        anchors = self._get_slot_anchors()

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)

        # Step 2: 遍历每个环境
        for env_id in env_ids:
            env_id_item = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            # 找到使用的原料箱
            active_mask = self.obj_to_source_id[env_id_item] != -1
            if not active_mask.any():
                continue

            box_idx = self.obj_to_source_id[env_id_item][active_mask][0].item()
            layout = self.stack_layout[env_id_item, box_idx]  # (max_stacks, max_per_stack)

            # 随机分配摞 → 槽位映射
            slot_perm = torch.randperm(4, device=self.device)

            for stack_idx in range(max_stacks):
                slot_idx = slot_perm[stack_idx % 4].item()
                anchor = anchors[slot_idx]
                z_offset = 0.015  # 箱底厚度

                for pos in range(max_per_stack):
                    obj_idx = layout[stack_idx, pos].item()
                    if obj_idx == -1:
                        break

                    params = self._stack_params_cache[self.object_names[obj_idx]]
                    stack_height = params['stack_height']

                    # 位置：锚点 + 轻微 XY 抖动
                    jitter_xy = 0.005
                    rand_x = (torch.rand(1, device=self.device).item() * 2 - 1) * jitter_xy
                    rand_y = (torch.rand(1, device=self.device).item() * 2 - 1) * jitter_xy

                    relative_pos = torch.tensor([
                        anchor[0].item() + rand_x,
                        anchor[1].item() + rand_y,
                        z_offset + stack_height / 2,
                    ], device=self.device).unsqueeze(0)

                    # 朝向：STACK_ORIENT + 轻微 Z 轴旋转抖动（±5°）
                    orient = params['stack_orient']
                    jitter_yaw = (torch.rand(1, device=self.device).item() * 2 - 1) * 5
                    relative_quat = euler_to_quat_isaac(
                        orient[0], orient[1], orient[2] + jitter_yaw
                    )

                    set_asset_relative_position(
                        env=self.env,
                        env_ids=torch.tensor([env_id_item], device=self.device),
                        target_asset=self.object_assets[obj_idx],
                        reference_asset=self.source_box_assets[box_idx],
                        relative_pos=relative_pos,
                        relative_quat=relative_quat,
                    )

                    z_offset += stack_height

    # ------------------------------------------------------------------ #
    #                       辅助方法                                       #
    # ------------------------------------------------------------------ #

    def _get_slot_anchors(self) -> torch.Tensor:
        """箱内 4 个槽位锚点（2×2 网格），返回 shape (4, 2)"""
        box_x = WORK_BOX_PARAMS['X_LENGTH']
        box_y = WORK_BOX_PARAMS['Y_LENGTH']
        return torch.tensor([
            [-box_x / 4, -box_y / 4],
            [ box_x / 4, -box_y / 4],
            [-box_x / 4,  box_y / 4],
            [ box_x / 4,  box_y / 4],
        ], device=self.device)

    def _move_all_objects_far(self, env_ids: torch.Tensor):
        """将所有物品移到远处（重置用）"""
        num_envs = len(env_ids)
        for obj_asset in self.object_assets:
            far_pos = torch.zeros((num_envs, 3), device=self.device)
            far_pos[:, 0] = 100
            far_pos[:, 1] = 100
            quat = torch.zeros((num_envs, 4), device=self.device)
            quat[:, 0] = 1
            set_asset_position(self.env, env_ids, obj_asset, far_pos, quat)

    # ------------------------------------------------------------------ #
    #                       指标                                           #
    # ------------------------------------------------------------------ #

    def _update_spawn_metrics(self):
        """
        堆叠加权得分：统计所有在 stack_layout 中的物品。

        权重 = stack_size - position（底层 = stack_size，顶层 = 1）
        stack_weighted_score = Σ(weight × success) / Σ(weight)
        """
        current_states = self.object_states

        weighted_success = torch.zeros(self.num_envs, device=self.device)
        total_weight = torch.zeros(self.num_envs, device=self.device)

        for env_id in range(self.num_envs):
            for src_idx in range(self.num_sources):
                layout = self.stack_layout[env_id, src_idx]
                for stack_idx in range(layout.shape[0]):
                    stack_size = (layout[stack_idx] != -1).sum().item()
                    if stack_size == 0:
                        continue
                    for pos in range(stack_size):
                        obj_idx = layout[stack_idx, pos].item()
                        if obj_idx == -1:
                            break
                        weight = float(stack_size - pos)
                        total_weight[env_id] += weight
                        if current_states[env_id, obj_idx] == 3:
                            weighted_success[env_id] += weight

        valid = total_weight > 0
        score = torch.zeros_like(weighted_success)
        score[valid] = weighted_success[valid] / total_weight[valid]
        self.metrics["stack_weighted_score"] = score

    # ------------------------------------------------------------------ #
    #                       接口方法                                       #
    # ------------------------------------------------------------------ #

    def _update_command(self):
        pass

    def command(self):
        pass
