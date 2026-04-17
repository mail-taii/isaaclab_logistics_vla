from __future__ import annotations

import math
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
from .scene_cfg import SKU_DEFINITIONS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg


class Spawn_ss_mt_stack_CommandTerm(BaseOrderCommandTerm):
    """
    单源-多目标 堆叠场景的 CommandTerm（含冗余物品）

    与 ss_st stack_scene 的主要差异：
    1. 使用多笔订单（随机 2~3 个订单箱），而非只用 1 个
    2. 每笔订单独立选定目标 SKU 及需求量
    3. 所有物品仍放入同一个原料箱（单源），混合堆叠
    4. 机器人需要根据多笔订单将物品分拣到对应的订单箱

    冗余物品机制：
    全部订单需求分配完成后，对原料箱剩余空位逐个进行伯努利采样
    （概率 p ~ Uniform(0, max_redundant_ratio)），独立决定是否填入冗余物品。
    订单物品 vs 冗余物品由 is_order_mask 显式标记。

    辅助变量 stack_layout:
        shape = [num_envs, num_sources, max_stacks, max_per_stack]
        值为 object index，-1 表示空位
    """

    SCALE = 1.0

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._stack_params_cache = self._build_stack_params_cache()

        max_stacks = getattr(cfg, 'max_stacks', 4)
        max_per_stack = getattr(cfg, 'max_per_stack', 4)
        self.stack_layout = torch.full(
            (self.num_envs, self.num_sources, max_stacks, max_per_stack),
            -1, dtype=torch.long, device=self.device
        )

        self.is_order_mask = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------------ #
    #                       堆叠参数缓存                                   #
    # ------------------------------------------------------------------ #

    def _build_stack_params_cache(self) -> dict:
        cache = {}
        for obj_name in self.object_names:
            raw = self._get_raw_params(obj_name)
            cache[obj_name] = self._compute_stack_params(raw)
        return cache

    def _get_raw_params(self, obj_name: str) -> dict:
        """
        获取缩放后的 SKU 物理参数（尺寸 + STACK_ORIENT）。
        根据 object_name 中包含的规范化 sku 名在 SKU_CONFIG 中查找。
        """
        params = None
        for sku_name, p in SKU_CONFIG.items():
            if sku_name in obj_name:
                params = p
                break

        if params is None:
            params = CRACKER_BOX_PARAMS

        scale = float(params.get('STACK_SCALE', self.SCALE))

        return {
            'X_LENGTH': params['X_LENGTH'] * scale,
            'Y_LENGTH': params['Y_LENGTH'] * scale,
            'Z_LENGTH': params['Z_LENGTH'] * scale,
            'STACK_ORIENT': params['STACK_ORIENT'],
        }

    def _compute_stack_params(self, params: dict) -> dict:
        x, y, z = params['X_LENGTH'], params['Y_LENGTH'], params['Z_LENGTH']
        dims = sorted([x, y, z], reverse=True)
        return {
            'base_area': dims[0] * dims[1],
            'stack_height': dims[2],
            'stack_orient': params['STACK_ORIENT'],
        }

    # ------------------------------------------------------------------ #
    #                       物品 / 箱子分配（Assign）                       #
    # ------------------------------------------------------------------ #

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        """
        单源-多目标堆叠分配逻辑（含冗余物品）：

        1. 随机选 1 个原料箱
        2. 随机决定 2~3 笔订单
        3. 为每笔订单独立选定目标 SKU 及需求量，写入 target_need_sku_num
        4. 汇总所有订单物品，确保实例不重复分配
        5. 为原料箱剩余空位采样冗余物品
        6. 全部物品在单个原料箱中混合堆叠
        """
        self.target_need_sku_num[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        self.is_order_mask[env_ids] = False
        self.stack_layout[env_ids] = -1

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)
        min_target_orders = getattr(self.cfg, 'min_target_orders', 2)
        max_target_orders = getattr(self.cfg, 'max_target_orders', 3)
        max_skus_per_order = getattr(self.cfg, 'max_skus_per_order', 3)
        max_items_per_order = getattr(self.cfg, 'max_items_per_order', 5)
        max_redundant_ratio = getattr(self.cfg, 'max_redundant_ratio', 0.7)

        total_capacity = max_stacks * max_per_stack

        for env_id in env_ids:
            env_id_val = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            # --- 1. 随机选一个原料箱 ---
            selected_source_box = torch.randint(0, self.num_sources, (1,)).item()

            # --- 2. 随机决定本局订单数 ---
            actual_max_orders = min(max_target_orders, self.num_targets)
            actual_min_orders = min(min_target_orders, actual_max_orders)
            num_orders = torch.randint(actual_min_orders, actual_max_orders + 1, (1,)).item()

            # --- 3. 为每笔订单分配目标 SKU 及需求量 ---
            # 追踪全局已分配的实例，避免同一实例被多笔订单重复使用
            globally_allocated: set[int] = set()
            sku_obj_groups: list[tuple[str, list[int]]] = []
            total_demand = 0

            for order_idx in range(num_orders):
                self.target_need_sku_num[env_id_val, order_idx, :] = 0

                remaining_cap = total_capacity - total_demand
                if remaining_cap <= 0:
                    break

                order_max_items = min(max_items_per_order, remaining_cap)

                num_skus_this_order = torch.randint(
                    1, min(max_skus_per_order, self.num_skus) + 1, (1,)
                ).item()
                sku_indices = torch.randperm(self.num_skus)[:num_skus_this_order].tolist()

                available_per_sku = []
                for sku_idx in sku_indices:
                    sku_name = self.sku_names[sku_idx]
                    all_instances = self.sku_to_indices[sku_name]
                    free_instances = [i for i in all_instances if i not in globally_allocated]
                    available_per_sku.append(free_instances)

                total_available = sum(len(pool) for pool in available_per_sku)
                if total_available == 0:
                    continue

                order_demand = torch.randint(
                    num_skus_this_order,
                    min(order_max_items, total_available) + 1,
                    (1,)
                ).item()

                demand_counts = [0] * num_skus_this_order
                remaining = order_demand

                for i in range(num_skus_this_order):
                    if remaining <= 0:
                        break
                    demand_counts[i] = 1
                    remaining -= 1

                while remaining > 0:
                    candidates = [
                        i for i in range(num_skus_this_order)
                        if demand_counts[i] < len(available_per_sku[i])
                    ]
                    if not candidates:
                        break
                    idx = candidates[torch.randint(0, len(candidates), (1,)).item()]
                    demand_counts[idx] += 1
                    remaining -= 1

                for i, sku_idx in enumerate(sku_indices):
                    if demand_counts[i] == 0:
                        continue
                    self.target_need_sku_num[env_id_val, order_idx, sku_idx] = demand_counts[i]

                    sku_name = self.sku_names[sku_idx]
                    free_pool = available_per_sku[i]
                    perm = torch.randperm(len(free_pool))[:demand_counts[i]]
                    selected_objs = [free_pool[p] for p in perm.tolist()]

                    for obj_idx in selected_objs:
                        self.obj_to_source_id[env_id_val, obj_idx] = selected_source_box
                        self.is_order_mask[env_id_val, obj_idx] = True
                        globally_allocated.add(obj_idx)

                    sku_obj_groups.append((sku_name, selected_objs))
                    total_demand += demand_counts[i]

            # --- 4. 确定实际用的摞数 ---
            n_stacks = min(max_stacks, math.ceil(total_demand / max_per_stack)) if total_demand > 0 else 1
            n_stacks = max(1, n_stacks)
            actual_capacity = n_stacks * max_per_stack

            # --- 5. 冗余物品：逐槽位伯努利采样 ---
            remaining_slots = actual_capacity - total_demand
            if remaining_slots > 0:
                redundant_prob = torch.rand(1, device=self.device).item() * max_redundant_ratio
                n_redundant = int(
                    (torch.rand(remaining_slots, device=self.device) < redundant_prob).sum().item()
                )

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

            # --- 7. 每摞内按底面积从大到小排序 ---
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
        与 ss_st 相同：遍历单个原料箱（单源），按摞摆放。
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        self._move_all_objects_far(env_ids)

        anchors = self._get_slot_anchors()
        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)

        for env_id in env_ids:
            env_id_item = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            active_mask = self.obj_to_source_id[env_id_item] != -1
            if not active_mask.any():
                continue

            box_idx = self.obj_to_source_id[env_id_item][active_mask][0].item()
            layout = self.stack_layout[env_id_item, box_idx]

            slot_perm = torch.randperm(4, device=self.device)

            for stack_idx in range(max_stacks):
                slot_idx = slot_perm[stack_idx % 4].item()
                anchor = anchors[slot_idx]
                z_offset = 0.025

                for pos in range(max_per_stack):
                    obj_idx = layout[stack_idx, pos].item()
                    if obj_idx == -1:
                        break

                    params = self._stack_params_cache[self.object_names[obj_idx]]
                    stack_height = params['stack_height']

                    jitter_xy = 0.00
                    rand_x = (torch.rand(1, device=self.device).item() * 2 - 1) * jitter_xy
                    rand_y = (torch.rand(1, device=self.device).item() * 2 - 1) * jitter_xy

                    relative_pos = torch.tensor([
                        anchor[0].item() + rand_x,
                        anchor[1].item() + rand_y,
                        z_offset + stack_height / 2,
                    ], device=self.device).unsqueeze(0)

                    orient = params['stack_orient']
                    jitter_yaw = (torch.rand(1, device=self.device).item() * 2 - 1) * 0
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

                    z_offset += stack_height + 0.02

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
        多目标版本的指标计算：遍历所有活跃订单，汇总各订单的完成情况。

        与 ss_st / ms_st 的差异：
        - 遍历所有 target（不只是 target_idx=0）
        - 分别计算每笔订单的 correct / wrong，再汇总
        - 堆叠加权得分同样按 is_order_mask 筛选
        """
        # -----------------------------------------------------------------
        # 1. 需求矩阵与实际包含矩阵（跨所有订单聚合）
        # -----------------------------------------------------------------
        need = self.target_need_sku_num          # (E, T, S)
        contain = self.target_contain_sku_num    # (E, T, S)

        order_active = (need != -1).any(dim=-1)  # (E, T) 哪些订单是活跃的
        safe_need = torch.where(need == -1, torch.zeros_like(need), need)

        correct_per_order = torch.minimum(contain, safe_need).sum(dim=-1)   # (E, T)
        wrong_per_order = torch.clamp(contain - safe_need, min=0).sum(dim=-1)  # (E, T)

        correct_per_order = correct_per_order * order_active.long()
        wrong_per_order = wrong_per_order * order_active.long()

        correct_picks = correct_per_order.sum(dim=-1)          # (E,)
        wrong_picks = wrong_per_order.sum(dim=-1)              # (E,)
        total_needed = safe_need.sum(dim=(1, 2))               # (E,)
        dropped_count = (self.object_states == 10).sum(dim=1)  # (E,)

        completion_rate = torch.where(
            total_needed > 0,
            correct_picks.float() / total_needed.float(),
            torch.tensor(1.0, device=self.device),
        )

        need_per_order = safe_need.sum(dim=-1)   # (E, T)
        is_order_complete = (correct_per_order == need_per_order) & (wrong_per_order == 0) & order_active
        is_success = is_order_complete.all(dim=-1)  # (E,) 所有活跃订单都完成

        # -----------------------------------------------------------------
        # 2. 堆叠加权得分
        # -----------------------------------------------------------------
        current_states = self.object_states
        target_state_min = self.num_sources + 1
        target_state_max = self.num_sources + self.num_targets

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
                        if not self.is_order_mask[env_id, obj_idx]:
                            continue
                        weight = float(stack_size - pos)
                        total_weight[env_id] += weight
                        state_val = current_states[env_id, obj_idx].item()
                        if target_state_min <= state_val <= target_state_max:
                            weighted_success[env_id] += weight

        valid = total_weight > 0
        stack_weighted_score = torch.zeros(self.num_envs, device=self.device)
        stack_weighted_score[valid] = weighted_success[valid] / total_weight[valid]

        self.metrics = {
            "completion_rate": completion_rate,
            "wrong_pick_count": wrong_picks,
            "dropped_count": dropped_count,
            "is_success": is_success.float(),
            "correct_picks": correct_picks,
            "total_needed": total_needed,
            "stack_weighted_score": stack_weighted_score,
        }
        return self.metrics

    # ------------------------------------------------------------------ #
    #                       接口方法                                       #
    # ------------------------------------------------------------------ #

    def _update_command(self):
        pass

    def command(self):
        pass

    def __str__(self) -> str:
        return "ss_mt_stack"
