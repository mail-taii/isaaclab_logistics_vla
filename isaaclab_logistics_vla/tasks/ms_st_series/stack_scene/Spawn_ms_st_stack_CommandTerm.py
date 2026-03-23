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


class Spawn_ms_st_stack_CommandTerm(BaseOrderCommandTerm):
    """
    多源-单目标 堆叠场景的 CommandTerm（含冗余物品）

    与 ss_st stack_scene 的主要差异：
    1. 使用多个原料箱（随机 2~3 个），而非只用 1 个
    2. 目标物在箱子间尽量均匀分配（round-robin）
    3. 每个箱子独立进行伯努利采样，为剩余空位填充冗余物品

    冗余物品机制：
    订单需求由 target_need_sku_num 定义。目标物 round-robin 分配到各箱后，
    对每个箱子的剩余空槽位逐个进行伯努利采样（概率 p ~ Uniform(0, max_redundant_ratio)），
    独立决定是否填入冗余物品。冗余物品从全局未分配实例池中随机选取。
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

        # 标记哪些物品是订单物品（True）vs 冗余物品（False）
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
        多源堆叠分配逻辑（含冗余物品）：

        1. 随机决定使用 2~3 个原料箱
        2. 随机选 n_active_skus 种 SKU，全部作为目标物
        3. 采样各 SKU 实例
        4. 目标物 round-robin 均匀分配到各箱，写入映射
        5. 逐箱伯努利采样，为剩余空位填充冗余物品
        6. 逐箱按 SKU 贪心填充到多摞，每摞按 base_area 从大到小排序
        """
        self.target_need_sku_num[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        self.is_order_mask[env_ids] = False
        self.stack_layout[env_ids] = -1

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)
        n_active_skus = getattr(self.cfg, 'num_active_skus', 3)
        max_instances_per_sku = getattr(self.cfg, 'max_instances_per_sku', 3)
        min_source_box = getattr(self.cfg, 'min_source_box', 2)
        max_source_box = getattr(self.cfg, 'max_source_box', 3)
        max_redundant_ratio = getattr(self.cfg, 'max_redundant_ratio', 0.7)

        for env_id in env_ids:
            env_id_val = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            # --- 1. 随机决定使用几个原料箱 ---
            num_boxes = torch.randint(min_source_box, max_source_box + 1, (1,)).item()

            # --- 2. 随机选 SKU 种类，全部作为目标物 ---
            num_to_sample = min(n_active_skus, self.num_skus)
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            # --- 3. 采样所有目标物实例 ---
            # 先将 order_0 整行置 0（标记"此订单存在，各 SKU 默认需求为 0"），
            # 其余 order 保持 -1（"订单不存在"）。随后逐 SKU 覆盖具体需求量。
            self.target_need_sku_num[env_id_val, 0, :] = 0

            target_objs = []
            for sku_idx in selected_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                k = torch.randint(1, min(len(global_indices), max_instances_per_sku) + 1, (1,)).item()
                selected = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]
                target_objs.extend(selected.tolist())
                self.target_need_sku_num[env_id_val, 0, sku_idx] = k

            # --- 4. 目标物尽量均匀分配到各箱 (round-robin)，写入映射 ---
            shuffled_targets = [target_objs[i] for i in torch.randperm(len(target_objs)).tolist()]
            box_contents: list[list[int]] = [[] for _ in range(num_boxes)]
            for i, obj_idx in enumerate(shuffled_targets):
                box_contents[i % num_boxes].append(obj_idx)

            for box_idx in range(num_boxes):
                for obj_idx in box_contents[box_idx]:
                    self.obj_to_source_id[env_id_val, obj_idx] = box_idx
                    self.is_order_mask[env_id_val, obj_idx] = True

            # --- 5. 逐箱伯努利采样，为剩余空位填充冗余物品 ---
            box_capacity = max_stacks * max_per_stack
            for box_idx in range(num_boxes):
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

            # --- 6. 逐箱构建堆叠布局 ---
            for box_idx in range(num_boxes):
                if not box_contents[box_idx]:
                    continue

                total_in_box = len(box_contents[box_idx])
                n_stacks = min(max_stacks, math.ceil(total_in_box / max_per_stack))
                n_stacks = max(1, n_stacks)

                # 按 SKU 分组
                sku_groups: dict[str, list[int]] = {}
                for obj_idx in box_contents[box_idx]:
                    obj_name = self.object_names[obj_idx]
                    matched_sku = obj_name
                    for sn in self.sku_names:
                        if sn in obj_name:
                            matched_sku = sn
                            break
                    sku_groups.setdefault(matched_sku, []).append(obj_idx)

                # 贪心填充：同种 SKU 优先填满当前摞
                stacks: list[list[int]] = [[] for _ in range(n_stacks)]
                current_stack = 0
                for _sku, objs in sku_groups.items():
                    for obj_idx in objs:
                        if current_stack < n_stacks and len(stacks[current_stack]) >= max_per_stack:
                            current_stack += 1
                        if current_stack >= n_stacks:
                            break
                        stacks[current_stack].append(obj_idx)

                # 每摞内按 base_area 从大到小排序（底部最大，顶部最小）
                for stack in stacks:
                    stack.sort(
                        key=lambda idx: self._stack_params_cache[self.object_names[idx]]['base_area'],
                        reverse=True,
                    )

                # 写入 stack_layout
                for stack_idx, stack in enumerate(stacks):
                    for pos, obj_idx in enumerate(stack):
                        self.stack_layout[env_id_val, box_idx, stack_idx, pos] = obj_idx

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)

    # ------------------------------------------------------------------ #
    #                       物品生成（Spawn）                               #
    # ------------------------------------------------------------------ #

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """
        根据 stack_layout 将物品摆放到各原料箱的槽位中。

        与 ss_st stack 的差异：遍历所有原料箱（多源），而非只处理 1 个。
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
                    z_offset = 0.015  # 箱底厚度

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
        """箱内 4 个槽位锚点（2x2 网格），返回 shape (4, 2)"""
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
        利用 target_need_sku_num / target_contain_sku_num 计算与基类一致的 Metrics，
        并在此基础上增加堆叠加权得分（仅统计订单物品，底层权重大于顶层）。
        """
        target_idx = 0

        # -----------------------------------------------------------------
        # 1. 需求矩阵与实际包含矩阵
        # -----------------------------------------------------------------
        target_needs = self.target_need_sku_num[:, target_idx, :]
        actual_in_target = self.target_contain_sku_num[:, target_idx, :]

        correct_picks = torch.minimum(actual_in_target, target_needs).sum(dim=1)
        wrong_picks = torch.clamp(actual_in_target - target_needs, min=0).sum(dim=1)
        dropped_count = (self.object_states == 10).sum(dim=1)
        total_needed = target_needs.sum(dim=1)

        completion_rate = torch.where(
            total_needed > 0,
            correct_picks.float() / total_needed.float(),
            torch.tensor(1.0, device=self.device),
        )
        is_success = (correct_picks == total_needed) & (wrong_picks == 0)

        # -----------------------------------------------------------------
        # 2. 堆叠加权得分：仅统计订单物品，权重 = stack_size - position
        #    成功判定：object_states 落在目标箱区间 (num_sources, num_sources + num_targets]
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
        return "ms_st_stack"