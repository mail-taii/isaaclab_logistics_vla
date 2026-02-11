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
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg


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
    堆叠场景的 CommandTerm

    设计要点：
    1. 箱内分为 4 个槽位（2×2 网格），每个槽位可放一摞
    2. 随机选 n_stacks 摞（1~max_stacks），随机选 m 种 SKU
    3. 确保目标物总数 > max_per_stack*(n_stacks-1)，保证必须用到 n_stacks 摞
    4. 目标物按 SKU 顺序贪心填充：同种优先填满一摞，不足时下一种接续
    5. 每摞内按底面积从小到大排列（底部面积最小，顶部面积最大）
    6. n 个槽位放目标摞，随机分配槽位；放置时加入轻微位置抖动和旋转抖动
    7. 本版本暂不生成干扰物（后续可简单扩展：在剩余 4-n 个空槽中随机散放即可）

    辅助变量 stack_layout:
        shape = [num_envs, num_sources, max_stacks, max_per_stack]
        值为 object index，-1 表示空位
        示例：[[[0,1,2,3], [4,5,6,7], [8,9,-1,-1], [-1,-1,-1,-1]]]
        表示 0 号环境、0 号原料箱有 3 摞，第 1 摞 4 个，第 2 摞 4 个，第 3 摞 2 个
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
        为每个环境分配物品到原料箱，并规划堆叠布局。

        流程：
        1. 随机选一个原料箱
        2. 随机选摞数 n_stacks ∈ [1, max_stacks]
        3. 随机选 m 种 SKU，随机决定每种激活几个实例
           约束：总实例数 ∈ (max_per_stack*(n-1), max_per_stack*n]
        4. 按 SKU 顺序贪心填充：同种 SKU 优先填满一摞，放不下则去下一摞
        5. 每摞内按底面积从小到大排序
        6. 写入 stack_layout 辅助张量
        """
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1
        self.stack_layout[env_ids] = -1

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)
        max_active_skus = getattr(self.cfg, 'max_active_skus', 5)

        for env_id in env_ids:
            env_id_val = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            # --- 1. 随机选一个原料箱 ---
            selected_source_box = torch.randint(0, self.num_sources, (1,)).item()

            # --- 2. 随机选摞数 n_stacks ∈ [1, max_stacks] ---
            n_stacks = torch.randint(1, max_stacks + 1, (1,)).item()

            # --- 3. 随机选 SKU 种类数 ---
            # m_skus 不能超过 max_per_stack*n_stacks（否则至少1个/SKU就放不下）
            max_skus = min(max_active_skus, self.num_skus, n_stacks * max_per_stack)
            if max_skus < 1:
                max_skus = 1
            m_skus = torch.randint(1, max_skus + 1, (1,)).item()
            selected_sku_indices = torch.randperm(self.num_skus)[:m_skus].tolist()

            # --- 4. 计算总实例数约束 ---
            # 每种 SKU 可用实例数
            available_per_sku = []
            for sku_idx in selected_sku_indices:
                sku_name = self.sku_names[sku_idx]
                available_per_sku.append(len(self.sku_to_indices[sku_name]))
            total_available = sum(available_per_sku)

            min_total = max_per_stack * (n_stacks - 1) + 1  # 保证必须用到 n_stacks 摞
            max_total = max_per_stack * n_stacks

            # 如果可用实例不够，降低摞数
            while min_total > total_available and n_stacks > 1:
                n_stacks -= 1
                min_total = max_per_stack * (n_stacks - 1) + 1
                max_total = max_per_stack * n_stacks

            # 还需保证每种 SKU 至少 1 个
            min_total = max(min_total, m_skus)
            actual_max = min(max_total, total_available)
            actual_min = min(min_total, actual_max)

            # 随机采样目标总数
            if actual_min >= actual_max:
                target_total = actual_max
            else:
                target_total = torch.randint(actual_min, actual_max + 1, (1,)).item()

            # --- 5. 分配每种 SKU 的实例数 ---
            counts = [0] * m_skus
            remaining = target_total

            # 先保证每种至少 1 个
            for i in range(m_skus):
                if remaining <= 0:
                    break
                counts[i] = 1
                remaining -= 1

            # 随机补满剩余
            while remaining > 0:
                candidates = [i for i in range(m_skus) if counts[i] < available_per_sku[i]]
                if not candidates:
                    break
                idx = candidates[torch.randint(0, len(candidates), (1,)).item()]
                counts[idx] += 1
                remaining -= 1

            # --- 6. 激活物品并按 SKU 分组记录 ---
            sku_obj_groups = []  # [(sku_name, [obj_indices]), ...]

            for i, sku_idx in enumerate(selected_sku_indices):
                if counts[i] == 0:
                    continue
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                k = counts[i]
                perm = torch.randperm(len(global_indices))[:k]
                selected_objs = [global_indices[p] for p in perm.tolist()]

                for obj_idx in selected_objs:
                    self.obj_to_source_id[env_id_val, obj_idx] = selected_source_box
                    self.obj_to_target_id[env_id_val, obj_idx] = 0  # 全部为目标物

                sku_obj_groups.append((sku_name, selected_objs))

            # --- 7. 按 SKU 顺序贪心填充到 n_stacks 摞 ---
            # 规则：同种 SKU 优先填满当前摞；摞满则换下一摞；下一种 SKU 接续当前摞
            stacks: list[list[int]] = [[] for _ in range(n_stacks)]
            current_stack = 0

            for _sku_name, obj_indices in sku_obj_groups:
                for obj_idx in obj_indices:
                    # 当前摞满了，换下一摞
                    if current_stack < n_stacks and len(stacks[current_stack]) >= max_per_stack:
                        current_stack += 1
                    if current_stack >= n_stacks:
                        break  # 理论上不会发生（总数 <= max_per_stack * n_stacks）
                    stacks[current_stack].append(obj_idx)

            # --- 8. 每摞内按底面积从大到小排序（底部最大，顶部最小）---
            for stack in stacks:
                # 底层为底面积最大的物体，越往上越小
                stack.sort(
                    key=lambda idx: self._stack_params_cache[self.object_names[idx]]['base_area'],
                    reverse=True,
                )

            # --- 9. 写入 stack_layout ---
            for stack_idx, stack in enumerate(stacks):
                for pos, obj_idx in enumerate(stack):
                    self.stack_layout[env_id_val, selected_source_box, stack_idx, pos] = obj_idx

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)

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
        堆叠加权得分：每摞中越靠下层的目标物成功放置，分数越高。

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
                        weight = float(stack_size - pos)  # 底层权重最高
                        total_weight[env_id] += weight
                        if current_states[env_id, obj_idx] == 3:  # 成功放置
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
