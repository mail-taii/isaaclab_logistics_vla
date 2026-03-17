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


class Spawn_ss_st_stack_CommandTerm(BaseOrderCommandTerm):
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
    订单需求由 target_need_sku_num 定义。在满足需求之后，对剩余空槽位
    逐个进行伯努利采样（概率 p ~ Uniform(0, max_redundant_ratio)），
    独立决定是否填入冗余物品。冗余数量自然服从 Binomial(remaining, p)，
    无固定上限，跨 episode 多样性更高。
    订单物品 vs 冗余物品由 is_order_mask 显式标记。

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

        # 标记哪些物品是订单物品（True）vs 冗余物品（False）
        self.is_order_mask = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
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
        方案 B：在 constant.py 中为每个 SKU 定义独立的参数字典，并统一挂在 SKU_CONFIG 中。
        这里根据 object_name 中包含的规范化 sku 名（如 "cracker_box"）在 SKU_CONFIG 中查找。
        """
        # 优先：根据 SKU_CONFIG 里的规范名匹配
        params = None
        for sku_name, p in SKU_CONFIG.items():
            if sku_name in obj_name:
                params = p
                break

        # 若未匹配到，则回退到默认 CRACKER_BOX_PARAMS，以保持行为稳定
        if params is None:
            params = CRACKER_BOX_PARAMS

        # 缩放倍率优先使用参数里的 STACK_SCALE，否则退回类属性 SCALE
        scale = float(params.get('STACK_SCALE', self.SCALE))

        return {
            'X_LENGTH': params['X_LENGTH'] * scale,
            'Y_LENGTH': params['Y_LENGTH'] * scale,
            'Z_LENGTH': params['Z_LENGTH'] * scale,
            'STACK_ORIENT': params['STACK_ORIENT'],
        }

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
        self.target_need_sku_num[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        self.is_order_mask[env_ids] = False
        self.stack_layout[env_ids] = -1

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)
        max_active_skus = getattr(self.cfg, 'max_active_skus', 5)
        max_redundant_ratio = getattr(self.cfg, 'max_redundant_ratio', 0.7)

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
            # 先将 order_0 整行置 0（标记"此订单存在，各 SKU 默认需求为 0"），
            # 其余 order 保持 -1（"订单不存在"）。随后逐 SKU 覆盖具体需求量。
            self.target_need_sku_num[env_id_val, 0, :] = 0

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
                    self.is_order_mask[env_id_val, obj_idx] = True

                sku_obj_groups.append((sku_name, selected_objs))
                slots_used += demand_counts[i]

            # --- 5. 冗余物品：逐槽位伯努利采样，随机填充剩余空位 ---
            remaining_slots = total_capacity - slots_used

            if remaining_slots > 0:
                redundant_prob = torch.rand(1, device=self.device).item() * max_redundant_ratio
                n_redundant = int((torch.rand(remaining_slots, device=self.device) < redundant_prob).sum().item())

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
                z_offset = 0.025  # 箱底厚度

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

                    z_offset += stack_height+0.01

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
        return "ss_st_stack"