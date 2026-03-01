from __future__ import annotations
import os
import time
import json
import glob
import random

import torch
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg


from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *
from isaaclab_logistics_vla.utils.path_utils import *

from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import BaseOrderCommandTerm

class STOrderCommandTerm(BaseOrderCommandTerm):
    def __init__(self, cfg, env, is_multi_target = False):
        super().__init__(cfg, env, is_multi_target)

    def _update_object_states(self):
        env_ids = torch.tensor(range(self.num_envs),dtype =torch.int32, device=self.device)
        # A. 在正确的原料箱里? (Pending 的必要条件) (所有变量 Shape 均为 [N, O])
        in_source = self._check_objects_in_source_box(env_ids)

        # B. 在正确的目标箱里? (Success 的必要条件)
        in_correct_target = self._check_objects_in_correct_target_box(env_ids)

        is_physically_failed = self._check_objects_failure(env_ids)

        # 读取当前状态 (N, O)
        current_states = self.object_states[env_ids]
        new_states = current_states.clone()

        active_mask = self.is_active_mask[env_ids]

        # --- 优先级1 判定失败（4）
        condition_fail = is_physically_failed & active_mask
        new_states[condition_fail] = 4

        # --- 优先级 2: 判定 COMPLETED (3) ---
        condition_success = in_correct_target & active_mask & (new_states != 4)
        new_states[condition_success] = 3

        # --- 优先级 3: 判定 TRANSIT (2) ---
        # 条件：不在源 AND 不在终 AND (没成功 且 没失败)
        condition_transit = (~in_source) & (~in_correct_target) & (new_states != 3) & (new_states != 4) & active_mask
        new_states[condition_transit] = 2

        # --- 优先级 4: 判定 PENDING (1) ---
        # 条件：在源
        condition_pending = in_source & active_mask
        new_states[condition_pending] = 1

        self.object_states[env_ids] = new_states

    def _check_objects_in_correct_target_box(self, env_ids):
        """
            return (N,O) O = len(objects),N = len(env_ids)
            N个环境中,每个物体是否在其应该去的订单箱内
            如某物体是干扰物，依然为False
        """
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        results = []
        target_ids_all = self.obj_to_target_id[env_ids]

        for i, obj_asset in enumerate(self.object_assets):
            expected_target_id = target_ids_all[:, i]
            obj_is_success = torch.zeros_like(expected_target_id, dtype=torch.bool)

            for k in range(self.num_targets):
                # 1. 筛选：目标是 k 的环境
                relevant_mask = (expected_target_id == k)
                
                # 2. 剪枝
                if not relevant_mask.any():
                    continue

                # 3. 子集计算
                subset_env_ids = env_ids[relevant_mask]
                
                is_in_subset = check_object_in_box(
                    subset_env_ids, 
                    obj_asset, 
                    self.target_box_assets[k],
                    box_size=self.box_size_tensor
                )
                
                # 4. 回填
                obj_is_success[relevant_mask] = is_in_subset
            
            results.append(obj_is_success)

        return torch.stack(results, dim=1)
    
    def _check_objects_failure(self, env_ids):
        is_failed = torch.zeros((len(env_ids), self.num_objects), dtype=torch.bool, device=self.device)

        for i, obj_asset in enumerate(self.object_assets):
            z_pos = obj_asset.data.root_pos_w[env_ids, 2]
            # 阈值设为 0.3 (原料箱在 0.75，稍微掉下来一点不算，必须是掉到地上)
            is_failed[:, i] |= (z_pos < 0.3)

        for i, obj_asset in enumerate(self.object_assets):
            # 获取该物体"本该去"的 ID
            correct_s_id = self.obj_to_source_id[env_ids, i] # (N,)
            correct_t_id = self.obj_to_target_id[env_ids, i] # (N,)

            # 1. 检查所有原料箱
            for k in range(self.num_sources):
                # 只有当物体真的在箱子 k 里时才触发
                in_box = check_object_in_box(
                    env_ids, obj_asset, self.source_box_assets[k], self.box_size_tensor
                )
                # 如果在箱子里 且 箱子号不对 -> 失败
                # (注意：干扰物的 correct_s_id=0，如果跑到 s_box_1(k=1) 就算失败)
                is_failed[:, i] |= (in_box & (correct_s_id != k))

            # 2. 检查所有订单箱
            for k in range(self.num_targets):
                in_box = check_object_in_box(
                    env_ids, obj_asset, self.target_box_assets[k], self.box_size_tensor
                )
                # 如果在箱子里 且 箱子号不对 -> 失败
                # (注意：干扰物的 correct_t_id=-1，进任何订单箱都算失败)
                is_failed[:, i] |= (in_box & (correct_t_id != k))

        return is_failed
