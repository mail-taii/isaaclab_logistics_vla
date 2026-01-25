from __future__ import annotations
import os
import time
import json

import torch
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_logistics_vla.tasks.test_tasks.order_series.OrderCommandCfg import OrderCommandCfg


from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *
from isaaclab_logistics_vla.utils.path_utils import *

class OrderCommand(CommandTerm):
    cfg: OrderCommandCfg

    def __init__(self, cfg: OrderCommandCfg, env: ManagerBasedRLEnvCfg):
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.object_names = cfg.objects
        self.object_assets = [env.scene[name] for name in cfg.objects]
        self.source_box_assets = [env.scene[name] for name in cfg.source_boxes]
        self.target_box_assets = [env.scene[name] for name in cfg.target_boxes]

        self.num_objects = len(cfg.objects)      # SKU 总数
        self.num_sources = len(cfg.source_boxes) # 原料箱总数
        self.num_targets = len(cfg.target_boxes) # 订单箱总数 (订单数)

        # [核心映射 1] 物品 -> 应该去哪个订单箱？
        # 值范围：0 ~ num_targets-1。如果值为 -1 代表该物品本局不需要处理。
        self.obj_to_target_id = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
        )

        # [核心映射 2] 物品 -> 应该从哪个原料箱生成？
        # 值范围：0 ~ num_sources-1
        self.obj_to_source_id = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
        )

        # 记录每个物品的状态：0=待生成, 1=待处理, 2=抓取中, 3=已完成, 4=失败
        self.object_states = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.long, device=self.device
        )

        self.order_completion = torch.zeros(
            (self.num_envs, self.num_targets), dtype=torch.bool, device=self.device
        )

        self.log_path = f"{get_logs_path()}/{int(time.time())}.jsonl"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)


    def __str__(self) -> str: 
        msg = f"该次任务共有{self.order_num}笔订单，{self.object_num}个SKU"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        self._save_dynamic_metrics()
        self._assign_objects_boxes(env_ids)

        # 重置物品状态为 0 (待处理)
        self.object_states[env_ids] = 0 
        # 重置订单箱完成状态
        self.order_completion[env_ids] = False
        
        self._spawn_items_in_source_boxes(env_ids)

    @abstractmethod
    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        raise NotImplementedError
        
    @abstractmethod
    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        raise NotImplementedError

    def _update_metrics(self):
        self._update_assign_metrics()
        self._update_spawn_metrics()
        
    @abstractmethod
    def _update_assign_metrics(self):
        raise NotImplementedError
    
    @abstractmethod
    def _update_spawn_metrics(self):
        raise NotImplementedError

    def _save_dynamic_metrics(self, env_ids):
        if len(env_ids) == 0:
            return
        
        extracted_data = {}
        ids_list = env_ids.tolist()
        num_resets = len(ids_list)

        for key, value in self.metrics.items():
            # 防御性编程：确保只处理 Tensor 类型
            if isinstance(value, torch.Tensor):
                # .tolist() 会自动把 GPU 数据拉回 CPU 并转为 Python 浮点数/列表
                # 如果 value 是 (N, 3)，这里就会变成 [[x,y,z], [x,y,z]...]，JSON 也能存
                extracted_data[key] = value[env_ids].tolist()

        with open(self.log_path, "a", encoding='utf-8') as f:
            for i in range(num_resets):
                # 1. 构建基础信息
                row_record = {
                    "timestamp": time.time(),
                    "env_id": ids_list[i],
                }

                # 2. 动态注入 metrics
                for key, val_list in extracted_data.items():
                    # val_list[i] 就是第 i 个被重置的环境对应的 metric 值
                    row_record[key] = val_list[i]

                # 3. 写入文件
                f.write(json.dumps(row_record) + "\n")

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])
