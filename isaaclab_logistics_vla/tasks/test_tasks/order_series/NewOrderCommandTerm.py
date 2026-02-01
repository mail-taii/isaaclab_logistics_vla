from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab_logistics_vla.tasks.test_tasks.order_series.NewOrderCommandCfg import OrderCommandCfg

class OrderCommand(CommandTerm):
    cfg: OrderCommandCfg

    def __init__(self, cfg: OrderCommandCfg, env: ManagerBasedEnv):
        # [关键修复] 在调用父类初始化前，先初始化可视化相关的属性
        # 因为父类 __init__ 会调用 _set_debug_vis_impl，那时这些属性必须存在
        self.goal_pose_visualizer = None
        self.current_pose_visualizer = None

        # 调用父类初始化
        super().__init__(cfg, env)

        # 保存环境引用的副本，方便后续访问场景数据
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # === 动态获取对象句柄 ===
        # 从配置中读取物品的名字列表（例如 ['o_item_0', 'o_item_1', ...]）
        self.object_names = cfg.objects
        self.object_assets = []
        
        for name in cfg.objects:
            if name in env.scene.keys():
                self.object_assets.append(env.scene[name])
            else:
                available_keys = list(env.scene.keys())
                print(f"[ERROR] Object '{name}' not found in Scene!")
        
        # 批量获取原料箱（Source Boxes）的物理资产句柄
        # 列表推导式：遍历 cfg.source_boxes 中的名字，从 env.scene 中取出对象
        self.source_box_assets = [env.scene[name] for name in cfg.source_boxes]
        self.target_box_assets = [env.scene[name] for name in cfg.target_boxes]

        # 记录关键的数量信息，用于后续定义 Tensor 的形状
        self.num_objects = len(cfg.objects)         # SKU 总数
        self.num_sources = len(cfg.source_boxes)    # 原料箱总数
        self.num_targets = len(cfg.target_boxes)    # 目标箱总数

        # [状态1] 物品 -> 目标箱的映射
        # 初始化为 -1，表示尚未分配。Shape: (环境数, 物品数)
        self.obj_to_target_id = torch.full((self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device)

        # [状态2] 物品 -> 原料箱的映射
        # 初始化为 -1。Shape: (环境数, 物品数)
        self.obj_to_source_id = torch.full((self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device)

        # [状态3] 物品的处理状态
        # 0=待处理, 1=处理中, 2=完成。初始化为 0。Shape: (环境数, 物品数)
        self.object_states = torch.zeros((self.num_envs, self.num_objects), dtype=torch.long, device=self.device)

        # [状态4] 订单完成状态
        # 记录每个目标箱是否已装满。True/False。Shape: (环境数, 目标箱数)
        self.order_completion = torch.zeros((self.num_envs, self.num_targets), dtype=torch.bool, device=self.device)
        
        # 初始化评估指标 (Metrics) 字典
        # 记录任务成功率，用于 TensorBoard 记录
        self.metrics['order_success_rate'] = torch.zeros(self.num_envs, device=self.device)

        # 如果 Config 里开启了 debug，则初始化可视化标记
        # 注意：这里我们再次调用一次，确保资源加载后可视化正确
        if self.cfg.debug_vis:
            self._set_debug_vis_impl(True)

    def __str__(self) -> str: 
        return f"NewOrderCommandTerm: {self.num_objects} Items -> {self.num_sources} Boxes"
    
    @property
    def command(self) -> torch.Tensor:
        return self.object_states

    # === 任务重置函数 ===
    # 当环境需要重置（Reset）时，系统会自动调用此函数
    # env_ids: 指定哪些环境需要重置 (通常是一个索引列表)
    def _resample_command(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 1. Assign Hook：负责计算“谁去哪”，更新 obj_to_source_id 和 obj_to_target_id
        if self.cfg.assign_objects_hook:
            self.cfg.assign_objects_hook(self, env_ids)
        
        # 2. 重置内部状态：将这些环境中的所有物品状态重置为 0 (待处理)，将这些环境中的订单完成状态重置为 False
        self.object_states[env_ids] = 0
        self.order_completion[env_ids] = False
        
        # 3. Spawn Hook
        if self.cfg.spawn_objects_hook:
            self.cfg.spawn_objects_hook(self, env_ids)

    def _update_metrics(self): pass
    def _update_command(self): pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # 这个函数可能会被父类 __init__ 调用，所以必须做好判空检查
        if debug_vis:
            # 只有当还没有创建 visualize 时才创建
            if not self.goal_pose_visualizer:
                # 只有在 cfg 已经赋值后才能创建 (父类初始化后 cfg 就有了)
                if hasattr(self, 'cfg') and self.cfg is not None:
                    try:
                        self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                        self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
                        self.goal_pose_visualizer.set_visibility(True)
                        self.current_pose_visualizer.set_visibility(True)
                    except Exception as e:
                        print(f"[Warning] Failed to initialize visualizers: {e}")
            else:
                self.goal_pose_visualizer.set_visibility(True)
                self.current_pose_visualizer.set_visibility(True)
        else:
            if self.goal_pose_visualizer:
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event): 
        pass