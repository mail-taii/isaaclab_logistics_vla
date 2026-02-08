from __future__ import annotations

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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg


class Spawn_ss_st_stack_CommandTerm(AssignSSSTCommandTerm):
    """
    堆叠场景的 CommandTerm
    
    核心逻辑：
    1. 目标物和干扰物分开成不同的摞
    2. 按底面积从大到小排序（最大面朝下）
    3. 从3个原料箱中随机选一个生成（仅方盒类物品）
    """

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 预计算堆叠参数缓存
        self._stack_params_cache = self._build_stack_params_cache()

    def _build_stack_params_cache(self) -> dict:
        """构建 SKU 名称到堆叠参数的映射（动态计算）"""
        cache = {}
        for obj_name in self.object_names:
            raw_params = self._get_raw_params(obj_name)
            cache[obj_name] = self._compute_stack_params(raw_params)
        return cache

    def _get_raw_params(self, obj_name: str) -> dict:
        """获取原始 SKU 参数"""
        if "cracker" in obj_name:
            return CRACKER_BOX_PARAMS
        elif "sugar" in obj_name:
            return SUGER_BOX_PARAMS
        elif "plastic_package" in obj_name:
            # 塑料包裹缩放到0.5倍，需要返回缩放后的尺寸
            scale = 0.5
            return {
                'USD_PATH': PLASTIC_PACKAGE_PARAMS['USD_PATH'],
                'X_LENGTH': PLASTIC_PACKAGE_PARAMS['X_LENGTH'] * scale,
                'Y_LENGTH': PLASTIC_PACKAGE_PARAMS['Y_LENGTH'] * scale,
                'Z_LENGTH': PLASTIC_PACKAGE_PARAMS['Z_LENGTH'] * scale,
                'SPARSE_ORIENT': PLASTIC_PACKAGE_PARAMS['SPARSE_ORIENT'],
            }
        elif "sf_big" in obj_name:
            scale = 0.5
            return {
                'USD_PATH': SFBIG_PARAMS['USD_PATH'],
                'X_LENGTH': SFBIG_PARAMS['X_LENGTH'] * scale,
                'Y_LENGTH': SFBIG_PARAMS['Y_LENGTH'] * scale,
                'Z_LENGTH': SFBIG_PARAMS['Z_LENGTH'] * scale,
                'SPARSE_ORIENT': SFBIG_PARAMS['SPARSE_ORIENT'],
            }
        elif "sf_small" in obj_name:
            scale = 0.4
            return {
                'USD_PATH': SFSMALL_PARAMS['USD_PATH'],
                'X_LENGTH': SFSMALL_PARAMS['X_LENGTH'] * scale,
                'Y_LENGTH': SFSMALL_PARAMS['Y_LENGTH'] * scale,
                'Z_LENGTH': SFSMALL_PARAMS['Z_LENGTH'] * scale,
                'SPARSE_ORIENT': SFSMALL_PARAMS['SPARSE_ORIENT'],
            }
        else:
            return CRACKER_BOX_PARAMS  # 默认

    def _compute_stack_params(self, params: dict) -> dict:
        """根据现有参数动态计算堆叠参数（仅方盒）"""
        x, y, z = params['X_LENGTH'], params['Y_LENGTH'], params['Z_LENGTH']

        # 选择最大的两个维度作为底面，最小维度作为高度
        dims = sorted([x, y, z], reverse=True)
        base_area = dims[0] * dims[1]  # 最大面积
        stack_height = dims[2]  # 最小维度作为高度

        # 计算需要的旋转角度使最小维度朝上
        stack_orient = self._compute_orient_for_min_height(x, y, z)

        return {
            'base_area': base_area,
            'stack_height': stack_height,
            'stack_orient': stack_orient,
        }

    def _compute_orient_for_min_height(self, x: float, y: float, z: float) -> tuple:
        """计算使最小维度朝上的欧拉角"""
        dims = [(x, 'x'), (y, 'y'), (z, 'z')]
        dims.sort(key=lambda d: d[0])
        min_axis = dims[0][1]  # 最小维度对应的轴

        if min_axis == 'z':
            return (0, 0, 0)
        elif min_axis == 'y':
            return (90, 0, 0)
        else:  # min_axis == 'x'
            return (0, 90, 0)

    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        """分配物品到箱子（复用父类逻辑）"""
        self.obj_to_target_id[env_ids] = -1
        self.obj_to_source_id[env_ids] = -1

        n_active_skus = getattr(self.cfg, "num_active_skus", 3)
        m_max_per_sku = getattr(self.cfg, "max_instances_per_sku", 2)

        for env_id in env_ids:
            # 随机选择一个原料箱
            selected_source_box = torch.randint(0, self.num_sources, (1,)).item()

            num_to_sample = min(n_active_skus, self.num_skus)
            selected_sku_indices = torch.randperm(self.num_skus)[:num_to_sample].tolist()

            # 约定：第一个选中的是干扰物，剩下的是目标
            distractor_sku_idx = selected_sku_indices[0]
            target_sku_indices = selected_sku_indices[1:]

            # --- 处理干扰物 ---
            sku_name = self.sku_names[distractor_sku_idx]
            global_indices = self.sku_to_indices[sku_name]

            k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
            selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]

            # 写入：干扰物 (Source=selected_source_box, Target=-1)
            self.obj_to_source_id[env_id, selected_objs] = selected_source_box
            self.obj_to_target_id[env_id, selected_objs] = -1

            # --- 处理目标物 ---
            for sku_idx in target_sku_indices:
                sku_name = self.sku_names[sku_idx]
                global_indices = self.sku_to_indices[sku_name]

                k = torch.randint(1, min(len(global_indices), m_max_per_sku) + 1, (1,)).item()
                selected_objs = torch.tensor(global_indices)[torch.randperm(len(global_indices))[:k]]

                # 写入：目标物 (Source=selected_source_box, Target=0) (SSST模式)
                self.obj_to_source_id[env_id, selected_objs] = selected_source_box
                self.obj_to_target_id[env_id, selected_objs] = 0

        self.is_active_mask[env_ids] = (self.obj_to_source_id[env_ids] != -1)
        self.is_target_mask[env_ids] = (self.obj_to_target_id[env_ids] != -1)

    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """在原料箱中生成堆叠物品"""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        num_envs = len(env_ids)

        # Step 0: 先将所有物品移至远处
        self._move_all_objects_far(env_ids)

        # Step 1: 获取箱内槽位锚点（2个槽位：目标摞 + 干扰摞）
        anchors = self._get_stack_anchors()

        # Step 2: 为每个环境处理堆叠
        for env_idx, env_id in enumerate(env_ids):
            env_id_item = env_id.item() if isinstance(env_id, torch.Tensor) else env_id

            # 获取该环境使用的原料箱
            active_mask = self.obj_to_source_id[env_id_item] != -1
            if not active_mask.any():
                continue

            # 找到使用的原料箱（应该都是同一个）
            active_source_ids = self.obj_to_source_id[env_id_item][active_mask]
            box_idx = active_source_ids[0].item()

            # 2.1 分离目标物和干扰物
            target_objs, distractor_objs = self._split_objects(env_id_item)

            # 2.2 目标物排序并堆叠
            if target_objs:
                self._spawn_stack(
                    env_id=env_id_item,
                    obj_indices=target_objs,
                    anchor=anchors[0],
                    box_idx=box_idx
                )

            # 2.3 干扰物处理
            if distractor_objs:
                distractor_mode = getattr(self.cfg, "distractor_mode", "stack")
                if distractor_mode == "stack":
                    self._spawn_stack(
                        env_id=env_id_item,
                        obj_indices=distractor_objs,
                        anchor=anchors[1],
                        box_idx=box_idx
                    )
                else:
                    self._spawn_scattered(env_id_item, distractor_objs, box_idx)

    def _split_objects(self, env_id: int) -> tuple[list, list]:
        """将活跃物品分为目标物和干扰物"""
        targets = []
        distractors = []

        for obj_idx in range(self.num_objects):
            if self.obj_to_source_id[env_id, obj_idx] == -1:
                continue  # 非活跃物品

            if self.is_target_mask[env_id, obj_idx]:
                targets.append(obj_idx)
            else:
                distractors.append(obj_idx)

        return targets, distractors

    def _sort_by_base_area(self, obj_indices: list) -> list:
        """按底面积从大到小排序"""
        def get_area(idx):
            return self._stack_params_cache[self.object_names[idx]]['base_area']
        return sorted(obj_indices, key=get_area, reverse=True)

    def _get_stack_anchors(self) -> torch.Tensor:
        """获取箱内堆叠槽位锚点（2个位置），沿箱子长边 Y 方向分成两份，避免挤飞"""
        box_x = WORK_BOX_PARAMS['X_LENGTH']
        box_y = WORK_BOX_PARAMS['Y_LENGTH']

        # 沿长边 Y 分为前后两区（X 居中）
        return torch.tensor([
            [0, -box_y / 4],   # 前侧（目标摞）
            [0, box_y / 4],   # 后侧（干扰摞）
        ], device=self.device)

    def _get_scatter_anchors(self) -> list:
        """获取散放物品的锚点（箱子角落位置）"""
        box_x = WORK_BOX_PARAMS['X_LENGTH']
        box_y = WORK_BOX_PARAMS['Y_LENGTH']
        margin = 0.05
        corner_positions = [
            (box_x / 2 - margin, box_y / 2 - margin),
            (-box_x / 2 + margin, box_y / 2 - margin),
            (box_x / 2 - margin, -box_y / 2 + margin),
            (-box_x / 2 + margin, -box_y / 2 + margin),
        ]
        return [torch.tensor([p[0], p[1]], device=self.device) for p in corner_positions]

    def _spawn_stack(self, env_id: int, obj_indices: list, anchor, box_idx: int):
        """
        堆叠方盒物品：按底面积排序，最多堆叠 max_stack_height 个，其余散放在角落。
        """
        max_stack_height = getattr(self.cfg, "max_stack_height", 4)
        sorted_objs = self._sort_by_base_area(obj_indices)
        objs_to_stack = sorted_objs[:max_stack_height]
        objs_to_scatter = sorted_objs[max_stack_height:]

        z_offset = 0.015  # 箱底厚度
        for obj_idx in objs_to_stack:
            z_offset = self._place_single_object(env_id, obj_idx, anchor, box_idx, z_offset)

        if objs_to_scatter:
            scatter_anchors = self._get_scatter_anchors()
            for i, obj_idx in enumerate(objs_to_scatter):
                scatter_anchor = scatter_anchors[i % len(scatter_anchors)]
                self._place_single_object(env_id, obj_idx, scatter_anchor, box_idx, 0.015)

    def _place_single_object(self, env_id: int, obj_idx: int, anchor, box_idx: int, z_offset: float) -> float:
        """放置单个方盒物品，返回新的 z_offset"""
        params = self._stack_params_cache[self.object_names[obj_idx]]
        stack_height = params['stack_height']

        jitter = 0.01
        rand_x = (torch.rand(1, device=self.device) * 2 - 1).item() * jitter
        rand_y = (torch.rand(1, device=self.device) * 2 - 1).item() * jitter

        relative_pos = torch.tensor([
            anchor[0].item() + rand_x,
            anchor[1].item() + rand_y,
            z_offset + stack_height / 2
        ], device=self.device).unsqueeze(0)

        orient = params['stack_orient']
        relative_quat = euler_to_quat_isaac(orient[0], orient[1], orient[2])

        # 放置物品
        set_asset_relative_position(
            env=self.env,
            env_ids=torch.tensor([env_id], device=self.device),
            target_asset=self.object_assets[obj_idx],
            reference_asset=self.source_box_assets[box_idx],
            relative_pos=relative_pos,
            relative_quat=relative_quat
        )

        # 返回新的高度偏移
        return z_offset + stack_height

    def _spawn_scattered(self, env_id: int, obj_indices: list, box_idx: int):
        """散放物品（简化版，复用 sparse_scene 的槽位逻辑）"""
        box_x = WORK_BOX_PARAMS['X_LENGTH']
        box_y = WORK_BOX_PARAMS['Y_LENGTH']

        # 简单的散放锚点
        scatter_anchors = torch.tensor([
            [-box_x / 4, -box_y / 4],
            [box_x / 4, -box_y / 4],
            [-box_x / 4, box_y / 4],
            [box_x / 4, box_y / 4],
        ], device=self.device)

        for i, obj_idx in enumerate(obj_indices):
            anchor_idx = i % len(scatter_anchors)
            anchor = scatter_anchors[anchor_idx]

            params = self._stack_params_cache[self.object_names[obj_idx]]
            stack_height = params['stack_height']
            jitter = 0.01
            rand_x = (torch.rand(1, device=self.device) * 2 - 1).item() * jitter
            rand_y = (torch.rand(1, device=self.device) * 2 - 1).item() * jitter

            relative_pos = torch.tensor([
                anchor[0].item() + rand_x,
                anchor[1].item() + rand_y,
                0.015 + stack_height / 2
            ], device=self.device).unsqueeze(0)

            orient = params['stack_orient']
            relative_quat = euler_to_quat_isaac(orient[0], orient[1], orient[2])

            set_asset_relative_position(
                env=self.env,
                env_ids=torch.tensor([env_id], device=self.device),
                target_asset=self.object_assets[obj_idx],
                reference_asset=self.source_box_assets[box_idx],
                relative_pos=relative_pos,
                relative_quat=relative_quat
            )

    def _move_all_objects_far(self, env_ids):
        """将所有物品移到远处"""
        num_envs = len(env_ids)
        for obj_asset in self.object_assets:
            far_position = torch.zeros((num_envs, 3), device=self.device)
            far_position[:, 0] = 100
            far_position[:, 1] = 100

            quat = torch.zeros((num_envs, 4), device=self.device)
            quat[:, 0] = 1

            set_asset_position(self.env, env_ids, obj_asset, far_position, quat)

    def _update_spawn_metrics(self):
        """堆叠特定指标（暂不实现）"""
        pass

    def _update_command(self):
        pass

    def command(self):
        pass
