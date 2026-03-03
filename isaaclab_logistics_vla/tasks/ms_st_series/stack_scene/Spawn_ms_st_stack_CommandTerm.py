from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

from isaaclab_logistics_vla.tasks.ms_st_series.Assign_ms_st_CommandTerm import AssignDSSTCommandTerm

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *
from .scene_cfg import SKU_DEFINITIONS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg


# SKU 名称片段 → constant 参数字典的映射
_SKU_PARAMS_MAP = {
    "cracker": CRACKER_BOX_PARAMS,
    "sugar": SUGER_BOX_PARAMS,
    "plastic_package": PLASTIC_PACKAGE_PARAMS,
    "sf_big": SF_BIG_PARAMS,
    "sf_small": SF_SMALL_PARAMS,
}


class Spawn_ms_st_stack_CommandTerm(AssignDSSTCommandTerm):
    """
    多源-单目标 堆叠场景的 CommandTerm

    与 ss_st stack_scene 的主要差异：
    1. 使用多个原料箱（随机 2~3 个），而非只用 1 个
    2. 所有激活物品均为目标物（暂不生成干扰物）
    3. 目标物在箱子间尽量均匀分配（round-robin）
    4. 继承 AssignDSSTCommandTerm（含源箱清理率、均衡度等指标）

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
        scale = self._get_scale_for_obj(obj_name)
        for key, params in _SKU_PARAMS_MAP.items():
            if key in obj_name:
                return {
                    'X_LENGTH': params['X_LENGTH'] * scale,
                    'Y_LENGTH': params['Y_LENGTH'] * scale,
                    'Z_LENGTH': params['Z_LENGTH'] * scale,
                    'STACK_ORIENT': params['STACK_ORIENT'],
                }
        p = CRACKER_BOX_PARAMS
        return {
            'X_LENGTH': p['X_LENGTH'] * scale,
            'Y_LENGTH': p['Y_LENGTH'] * scale,
            'Z_LENGTH': p['Z_LENGTH'] * scale,
            'STACK_ORIENT': p['STACK_ORIENT'],
        }

    def _get_scale_for_obj(self, obj_name: str) -> float:
        for sku_name, (_usd_path, _count, scale) in SKU_DEFINITIONS.items():
            if sku_name in obj_name:
                return float(scale)
        return float(self.SCALE)

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
        多源堆叠分配逻辑（全目标，无干扰物）：

        1. 随机决定使用 2~3 个原料箱
        2. 随机选 n_active_skus 种 SKU，全部作为目标物
        3. 采样各 SKU 实例
        4. 目标物 round-robin 均匀分配到各箱
        5. 每个箱子内部按 SKU 贪心填充到多摞，每摞按 base_area 从大到小排序
        """
        self.obj_can_to_targets_ids[env_ids] = -1
        self.target_need_sku_num[env_ids] = 0
        self.obj_to_source_id[env_ids] = -1
        self.stack_layout[env_ids] = -1

        max_stacks = getattr(self.cfg, 'max_stacks', 4)
        max_per_stack = getattr(self.cfg, 'max_per_stack', 4)
        n_active_skus = getattr(self.cfg, 'num_active_skus', 3)
        max_instances_per_sku = getattr(self.cfg, 'max_instances_per_sku', 3)
        min_source_box = getattr(self.cfg, 'min_source_box', 2)
        max_source_box = getattr(self.cfg, 'max_source_box', 3)

        for env_id in env_ids:
            env_id_val = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)

            # --- 1. 随机决定使用几个原料箱 ---
            num_boxes = torch.randint(min_source_box, max_source_box + 1, (1,)).item()

            # --- 2. 随机选 SKU 种类，全部作为目标物 ---
            num_to_sample = min(n_active_skus, self.num_skus)
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            # --- 3. 采样所有目标物实例 ---
            target_objs = []
            for sku_idx in selected_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]
                k = torch.randint(1, min(len(global_indices), max_instances_per_sku) + 1, (1,)).item()
                selected = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]
                target_objs.extend(selected.tolist())
                self.target_need_sku_num[env_id_val, 0, sku_idx] = k

            # --- 4. 目标物尽量均匀分配到各箱 (round-robin) ---
            shuffled_targets = [target_objs[i] for i in torch.randperm(len(target_objs)).tolist()]
            box_contents: list[list[int]] = [[] for _ in range(num_boxes)]
            for i, obj_idx in enumerate(shuffled_targets):
                box_contents[i % num_boxes].append(obj_idx)

            # --- 5. 写入映射并构建堆叠布局 ---
            for box_idx in range(num_boxes):
                for obj_idx in box_contents[box_idx]:
                    self.obj_to_source_id[env_id_val, obj_idx] = box_idx
                    self.obj_can_to_targets_ids[env_id_val, obj_idx, :] = 0
                    self.obj_can_to_targets_ids[env_id_val, obj_idx, 0] = 1

                if not box_contents[box_idx]:
                    continue

                # 确定该箱摞数
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
        self.is_target_mask[env_ids] = (self.obj_can_to_targets_ids[env_ids] == 1).any(dim=-1)

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
        堆叠加权得分：每摞中越靠下层的物体成功放置，分数越高。
        权重 = stack_size - position（底层 = stack_size，顶层 = 1）
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
