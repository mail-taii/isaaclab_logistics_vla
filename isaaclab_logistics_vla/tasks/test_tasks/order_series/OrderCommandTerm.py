from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab_logistics_vla.tasks.test_tasks.order_series.OrderCommandCfg import OrderCommandCfg


from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

class OrderCommand(CommandTerm):
    cfg: OrderCommandCfg

    def __init__(self, cfg: OrderCommandCfg, env: ManagerBasedEnv):
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

        self.metrics['object_success_rate'] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['order_success_rate'] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str: 
        msg = f"该次任务共有{self.order_num}笔订单，{self.object_num}个SKU"
        return msg
    
    @property
    def command(self)->torch.Tensor:
        return self.object_states

    def _resample_command(self, env_ids: Sequence[int]):
        """
        数据流打通MVP测试版：
        1. 原料箱 <-> 目标箱 一一对应 (Random Permutation)
        2. 物品 -> 原料箱 随机分配 (Random Assignment)
        3. 物品 -> 目标箱 自动推导
        """
        assert self.num_sources == self.num_targets, "测试模式下，原料箱和目标箱数量必须一致！"
        assert self.num_objects >= self.num_sources, "物品数量太少，不够分配！"

        num_resample = len(env_ids)

        rand_noise = torch.rand((num_resample, self.num_sources), device=self.device)
        #shape (num_resample, num_sources) 例如: [[2, 0, 1], [1, 2, 0]] 表示 Env0 中: 原料箱0->目标箱2, 原料箱1->目标箱0...
        source_to_target_map = torch.argsort(rand_noise, dim=1) 

        # shape: (num_resample, num_objects)
        obj_source_idx = torch.randint(
            0, self.num_sources, 
            (num_resample, self.num_objects), 
            device=self.device
        )

        # shape: (num_resample, num_objects)
        obj_target_idx = torch.gather(
            input=source_to_target_map, 
            dim=1, 
            index=obj_source_idx
        )

        self.obj_to_source_id[env_ids] = obj_source_idx
        self.obj_to_target_id[env_ids] = obj_target_idx

        # 重置物品状态为 0 (待处理)
        self.object_states[env_ids] = 0
        
        # 重置订单箱完成状态
        self.order_completion[env_ids] = False
        
        if num_resample == 1:
            print(f"\n[Resample Env {env_ids[0]}]")
            print(f"Map (Src->Tgt): {source_to_target_map[0].tolist()}")
            print(f"Obj Sources:    {obj_source_idx[0].tolist()}")
            print(f"Obj Targets:    {obj_target_idx[0].tolist()}")
        
        self._spawn_items_in_source_boxes(env_ids)
        
    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        """
        利用 set_asset_relative_position 将物品放置在对应的原料箱上方。
        """
        # 1. 准备数据
        # 确保 env_ids 是 Tensor
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        num_envs = len(env_ids)

        def get_params_and_dims(obj_name):
            # 返回参数字典，并统一转化为 X, Y, Z, ORI
            if "cracker" in obj_name: p = CRACKER_BOX_PARAMS
            elif "sugar" in obj_name: p = SUGER_BOX_PARAMS
            elif "soup" in obj_name:  p = TOMATO_SOUP_CAN_PARAMS
            else: p = CRACKER_BOX_PARAMS # 默认
            
            # 统一处理汤罐头的半径 -> 直径
            if 'RADIUS' in p:
                return p['RADIUS']*2, p['RADIUS']*2, p['Z_LENGTH'], p['STANDARD_ORI']
            else:
                return p['X_LENGTH'], p['Y_LENGTH'], p['Z_LENGTH'], p['STANDARD_ORI']

        box_x = WORK_BOX_PARAMS['X_LENGTH'] # 0.36
        box_y = WORK_BOX_PARAMS['Y_LENGTH'] # 0.56
        box_z = WORK_BOX_PARAMS['Z_LENGTH'] # 0.23

        cell_x = box_x / 2.0
        cell_y = box_y / 2.0

        anchors = torch.tensor([
            [-cell_x/2, -cell_y/2], 
            [-cell_x/2,  cell_y/2],
            [ cell_x/2, -cell_y/2],
            [ cell_x/2,  cell_y/2]
        ], device=self.device)

        # === 3. 随机分配槽位 ===
        # 生成随机排列 (N, 4)，例如 [3, 0, 1, 2]
        # 保证同一箱内不重叠
        slot_perms = torch.rand(num_envs, 4, device=self.device).argsort(dim=1)
        
        for obj_idx, obj_asset in enumerate(self.object_assets):
            # 获取当前物品的物理参数
            item_x, item_y, item_z, item_ori = get_params_and_dims(self.object_names[obj_idx])
            # 获取需要处理的环境 ID
            assigned_box_indices = self.obj_to_source_id[env_ids, obj_idx]

            for box_idx, box_asset in enumerate(self.source_box_assets):
                # Mask 筛选
                mask = (assigned_box_indices == box_idx)
                if not mask.any(): continue
                
                active_env_ids = env_ids[mask]
                num_active = len(active_env_ids)

                # --- A. 确定姿态 ---
                # 直接读取预设的四元数
                relative_quat = euler_to_quat_isaac(item_ori[0],item_ori[1],item_ori[2], return_tensor=True).repeat(num_active, 1)

                # --- B. 确定位置 (Grid + Jitter) ---
                # 取出分配给当前物品的槽位锚点
                # 使用 obj_idx % 4 防止物品超过4个时报错 (虽然你暂定不超过4)
                current_slots = slot_perms[mask, obj_idx % 4]
                batch_anchors = anchors[current_slots] # (N, 2)

                # 计算 Margin (可移动的空隙)
                # 既然 item_x 已经是旋转后的 X 轴长度，直接相减即可
                margin_x = (cell_x - item_x) / 2.0 - 0.015 # 留 5mm 缝隙
                margin_y = (cell_y - item_y) / 2.0 - 0.015
                
                # 生成随机抖动 (-margin 到 +margin)
                rand_x = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_x
                rand_y = (torch.rand(num_active, device=self.device) * 2 - 1) * margin_y
                
                # --- C. 确定 Z 轴 ---
                # 箱底 + 物品半高 + 微小缓冲
                z_pos = (item_z / 2.0) +0.015

                # 组合最终位置
                relative_pos = torch.stack([
                    batch_anchors[:, 0] + rand_x,
                    batch_anchors[:, 1] + rand_y,
                    torch.full((num_active,), z_pos, device=self.device)
                ], dim=-1)

                # 5. 写入仿真
                set_asset_relative_position(
                    env=self.env,
                    env_ids=active_env_ids,
                    target_asset=obj_asset,
                    reference_asset=box_asset,
                    relative_pos=relative_pos,
                    relative_quat=relative_quat
                )

    def _update_metrics(self):
        pass

    def _update_command(self):
        #self._spawn_items_in_source_boxes([i for i in range(self.num_envs)])
        pass

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
