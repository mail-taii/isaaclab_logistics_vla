from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import BaseOrderCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *
from .scene_cfg import SKU_DEFINITIONS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg


class Spawn_ms_mt_stack_CommandTerm(BaseOrderCommandTerm):
    """
    多源-多目标 堆叠场景 CommandTerm（复用 ms_st + ss_mt）

    核心差异组合：
    - 多源：订单物品与冗余物品分布在多个原料箱（2~3 个，随机）
    - 多目标：同时生成多笔订单（2~3 笔，随机），每个订单箱有独立需求矩阵 target_need_sku_num
    - 约束：同一物品实例不会被多笔订单重复分配

    stack_layout:
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

        # 标记哪些物品是订单物品（True）vs 冗余物品（False）
        self.is_order_mask = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
        )

        # 多目标映射：每个物品实例对应哪个订单箱（-1 代表不属于任何订单）
        self.obj_to_target_id = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
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
        组合 ms_st(多源) + ss_mt(多订单) 的分配逻辑：

        1) 随机决定使用 N 个原料箱（2~3）
        2) 随机决定生成 M 笔订单（2~3），每笔订单独立随机 SKU 与需求量
        3) 汇总所有订单物品（不重复实例），并为每个物品绑定目标订单箱 obj_to_target_id
        4) 将所有订单物品 round-robin 分配到 N 个原料箱 obj_to_source_id
        5) 对每个原料箱剩余空位逐槽位伯努利采样，补充冗余物品
        6) 每个原料箱独立堆叠（按 SKU 贪心填充到多摞；每摞按 base_area 从大到小排序）
        """
        self.target_need_sku_num[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        self.obj_to_target_id[env_ids] = -1
        self.is_order_mask[env_ids] = False
        self.stack_layout[env_ids] = -1

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)

        min_source_box = getattr(self.cfg, 'min_source_box', 2)
        max_source_box = getattr(self.cfg, 'max_source_box', 3)

        min_target_orders = getattr(self.cfg, 'min_target_orders', 2)
        max_target_orders = getattr(self.cfg, 'max_target_orders', 3)
        max_skus_per_order = getattr(self.cfg, 'max_skus_per_order', 3)
        max_items_per_order = getattr(self.cfg, 'max_items_per_order', 5)

        max_redundant_ratio = getattr(self.cfg, 'max_redundant_ratio', 0.7)

        box_capacity = max_stacks * max_per_stack

        for env_id in env_ids:
            env_id_val = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            # --- 1) 选用的原料箱数量 ---
            actual_max_sources = min(max_source_box, self.num_sources)
            actual_min_sources = min(min_source_box, actual_max_sources)
            num_sources_used = torch.randint(actual_min_sources, actual_max_sources + 1, (1,)).item()

            # --- 2) 选用的订单数量 ---
            actual_max_orders = min(max_target_orders, self.num_targets)
            actual_min_orders = min(min_target_orders, actual_max_orders)
            num_orders = torch.randint(actual_min_orders, actual_max_orders + 1, (1,)).item()

            # --- 3) 生成订单需求 + 绑定订单物品实例（不重复）---
            globally_allocated: set[int] = set()
            order_items: list[tuple[int, int]] = []  # (obj_idx, order_idx)

            total_order_demand = 0
            for order_idx in range(num_orders):
                self.target_need_sku_num[env_id_val, order_idx, :] = 0

                # 控制总体规模，避免需求无限膨胀
                remaining_global_cap = num_sources_used * box_capacity - total_order_demand
                if remaining_global_cap <= 0:
                    break

                order_max_items = min(max_items_per_order, remaining_global_cap)

                # 每单随机选 SKU 子集
                num_skus_this_order = torch.randint(
                    1, min(max_skus_per_order, self.num_skus) + 1, (1,)
                ).item()
                sku_indices = torch.randperm(self.num_skus)[:num_skus_this_order].tolist()

                available_pools: list[list[int]] = []
                for sku_idx in sku_indices:
                    sku_name = self.sku_names[sku_idx]
                    all_instances = self.sku_to_indices[sku_name]
                    free_instances = [i for i in all_instances if i not in globally_allocated]
                    available_pools.append(free_instances)

                total_available = sum(len(p) for p in available_pools)
                if total_available == 0:
                    continue

                # 单笔订单需求：至少每个 SKU 1 个，最多 order_max_items，且不超过可用实例数
                order_demand = torch.randint(
                    num_skus_this_order,
                    min(order_max_items, total_available) + 1,
                    (1,)
                ).item()

                demand_counts = [0] * num_skus_this_order
                remaining = order_demand

                # 先保证每个 SKU 至少 1 件
                for i in range(num_skus_this_order):
                    if remaining <= 0:
                        break
                    demand_counts[i] = 1
                    remaining -= 1

                # 再把剩余需求随机分配到仍有余量的 SKU 上
                while remaining > 0:
                    candidates = [
                        i for i in range(num_skus_this_order)
                        if demand_counts[i] < len(available_pools[i])
                    ]
                    if not candidates:
                        break
                    idx = candidates[torch.randint(0, len(candidates), (1,)).item()]
                    demand_counts[idx] += 1
                    remaining -= 1

                # 落盘到 target_need_sku_num，并选取实例绑定到该订单
                for i, sku_idx in enumerate(sku_indices):
                    if demand_counts[i] == 0:
                        continue

                    self.target_need_sku_num[env_id_val, order_idx, sku_idx] = demand_counts[i]

                    free_pool = available_pools[i]
                    perm = torch.randperm(len(free_pool))[:demand_counts[i]]
                    selected_objs = [free_pool[p] for p in perm.tolist()]

                    for obj_idx in selected_objs:
                        self.is_order_mask[env_id_val, obj_idx] = True
                        self.obj_to_target_id[env_id_val, obj_idx] = order_idx
                        globally_allocated.add(obj_idx)
                        order_items.append((obj_idx, order_idx))

                    total_order_demand += demand_counts[i]

            # --- 4) 订单物品分配到多个原料箱（round-robin）---
            if order_items:
                perm = torch.randperm(len(order_items))
                shuffled = [order_items[i] for i in perm.tolist()]
            else:
                shuffled = []

            box_contents: list[list[int]] = [[] for _ in range(num_sources_used)]
            for i, (obj_idx, _order_idx) in enumerate(shuffled):
                box_idx = i % num_sources_used
                box_contents[box_idx].append(obj_idx)

            for box_idx in range(num_sources_used):
                for obj_idx in box_contents[box_idx]:
                    self.obj_to_source_id[env_id_val, obj_idx] = box_idx

            # --- 5) 逐箱冗余补充（伯努利逐槽位）---
            for box_idx in range(num_sources_used):
                remaining_slots = box_capacity - len(box_contents[box_idx])
                if remaining_slots <= 0:
                    continue

                redundant_prob = torch.rand(1, device=self.device).item() * max_redundant_ratio
                n_redundant = int((torch.rand(remaining_slots, device=self.device) < redundant_prob).sum().item())
                if n_redundant <= 0:
                    continue

                available_pool = [
                    idx for idx in range(self.num_objects)
                    if self.obj_to_source_id[env_id_val, idx].item() == -1
                ]
                n_redundant = min(n_redundant, len(available_pool))
                if n_redundant <= 0:
                    continue

                perm = torch.randperm(len(available_pool))[:n_redundant]
                redundant_objs = [available_pool[p] for p in perm.tolist()]

                for obj_idx in redundant_objs:
                    self.obj_to_source_id[env_id_val, obj_idx] = box_idx
                box_contents[box_idx].extend(redundant_objs)

            # --- 6) 每个原料箱独立构建堆叠布局 ---
            for box_idx in range(num_sources_used):
                if not box_contents[box_idx]:
                    continue

                total_in_box = len(box_contents[box_idx])
                n_stacks = min(max_stacks, math.ceil(total_in_box / max_per_stack))
                n_stacks = max(1, n_stacks)

                # 按 SKU 分组（保持与现有实现一致：按插入顺序遍历 dict）
                sku_groups: dict[str, list[int]] = {}
                for obj_idx in box_contents[box_idx]:
                    obj_name = self.object_names[obj_idx]
                    matched_sku = obj_name
                    for sn in self.sku_names:
                        if sn in obj_name:
                            matched_sku = sn
                            break
                    sku_groups.setdefault(matched_sku, []).append(obj_idx)

                stacks: list[list[int]] = [[] for _ in range(n_stacks)]
                current_stack = 0
                for _sku, objs in sku_groups.items():
                    for obj_idx in objs:
                        if current_stack < n_stacks and len(stacks[current_stack]) >= max_per_stack:
                            current_stack += 1
                        if current_stack >= n_stacks:
                            break
                        stacks[current_stack].append(obj_idx)

                for stack in stacks:
                    stack.sort(
                        key=lambda idx: self._stack_params_cache[self.object_names[idx]]['base_area'],
                        reverse=True,
                    )

                for stack_idx, stack in enumerate(stacks):
                    for pos, obj_idx in enumerate(stack):
                        self.stack_layout[env_id_val, box_idx, stack_idx, pos] = obj_idx

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)

    # ------------------------------------------------------------------ #
    #                       物品生成（Spawn）                               #
    # ------------------------------------------------------------------ #

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """
        多源摆放：遍历所有原料箱，按照各自 stack_layout 在箱内 2x2 槽位堆叠摆放。
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        self._move_all_objects_far(env_ids)

        anchors = self._get_slot_anchors()
        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)

        for env_id in env_ids:
            env_id_item = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            for box_idx in range(self.num_sources):
                layout = self.stack_layout[env_id_item, box_idx]
                if (layout == -1).all():
                    continue

                slot_perm = torch.randperm(4, device=self.device)

                for stack_idx in range(max_stacks):
                    slot_idx = slot_perm[stack_idx % 4].item()
                    anchor = anchors[slot_idx]
                    z_offset = 0.015

                    for pos in range(max_per_stack):
                        obj_idx = layout[stack_idx, pos].item()
                        if obj_idx == -1:
                            break

                        params = self._stack_params_cache[self.object_names[obj_idx]]
                        stack_height = params['stack_height']

                        jitter_xy = 0.005
                        rand_x = (torch.rand(1, device=self.device).item() * 2 - 1) * jitter_xy
                        rand_y = (torch.rand(1, device=self.device).item() * 2 - 1) * jitter_xy

                        relative_pos = torch.tensor([
                            anchor[0].item() + rand_x,
                            anchor[1].item() + rand_y,
                            z_offset + stack_height / 2,
                        ], device=self.device).unsqueeze(0)

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
        复用 ss_mt 的多订单汇总逻辑 + ms_st 的多源堆叠加权得分计算方式。
        """
        need = self.target_need_sku_num          # (E, T, S)
        contain = self.target_contain_sku_num    # (E, T, S)

        order_active = (need != -1).any(dim=-1)  # (E, T)
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

        need_per_order = safe_need.sum(dim=-1)
        is_order_complete = (correct_per_order == need_per_order) & (wrong_per_order == 0) & order_active
        is_success = is_order_complete.all(dim=-1)

        # -----------------------------------------------------------------
        # 额外统计：wrong_place / failure 等（对齐 reward_cfg / termination 里的 key）
        # -----------------------------------------------------------------
        # wrong_place：订单物品被放入错误的目标箱
        wrong_place_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # non_order_in_target：冗余/干扰物被放进任一目标箱
        non_order_in_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        target_state_min = self.num_sources + 1
        target_state_max = self.num_sources + self.num_targets

        for env_id in range(self.num_envs):
            for obj_idx in range(self.num_objects):
                state_val = int(current_states[env_id, obj_idx].item())
                if not (target_state_min <= state_val <= target_state_max):
                    continue

                inferred_target = state_val - target_state_min  # 0..num_targets-1

                if self.is_order_mask[env_id, obj_idx]:
                    assigned_target = int(self.obj_to_target_id[env_id, obj_idx].item())
                    if assigned_target != -1 and inferred_target != assigned_target:
                        wrong_place_count[env_id] += 1
                else:
                    non_order_in_target[env_id] += 1

        total_picks = correct_picks + wrong_picks
        denom_needed = torch.clamp(total_needed, min=1).float()
        denom_picks = torch.clamp(total_picks, min=1).float()

        wrong_pick_rate = (wrong_picks.float() + non_order_in_target.float()) / denom_picks
        wrong_place_rate = wrong_place_count.float() / denom_picks
        failure_rate = dropped_count.float() / denom_needed
        mean_action_time = torch.zeros(self.num_envs, device=self.device)

        # 堆叠加权得分（仅统计订单物品）
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
            # 对齐现有 reward/termination 约定的 key
            "order_completion_rate": completion_rate,
            "mean_action_time": mean_action_time,
            "failure_rate": failure_rate,
            "wrong_pick_rate": wrong_pick_rate,
            "wrong_place_rate": wrong_place_rate,
            # 额外保留调试信息
            "completion_rate": completion_rate,
            "wrong_pick_count": wrong_picks,
            "wrong_place_count": wrong_place_count,
            "dropped_count": dropped_count,
            "non_order_in_target_count": non_order_in_target,
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
        return "ms_mt_stack"
