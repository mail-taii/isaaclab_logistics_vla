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

class BaseOrderCommandTerm(CommandTerm):
    cfg: OrderCommandTermCfg

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv,is_multi_target:bool):
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.is_multi_target = is_multi_target

        self.object_names = []    #该任务实际含有的所有SKU的名称
        self.object_assets:list[RigidObject | Articulation] = []    #该任务实际含有的所有SKU对象
        self.sku_to_indices :dict[str,list] = {}    #某类SKU对应的实例对象的索引列表（在object_assets中）

        # 输入: cfg.objects 列表只包含 SKU 名称，例如 ["cracker_box", "sugar_box"]
        self.sku_names = cfg.objects # 记录原始 SKU 清单

        self._discover_object_instances(env, cfg.objects)    # 填充object_names object_assets sku_to_indices

        self.num_objects = len(self.object_names) # 总实例数
        self.num_skus = len(self.sku_names)     # SKU 种类数

        self.source_box_assets = [env.scene[name] for name in cfg.source_boxes]
        self.target_box_assets = [env.scene[name] for name in cfg.target_boxes]

        self.num_sources = len(cfg.source_boxes) # 原料箱总数
        self.num_targets = len(cfg.target_boxes) # 订单箱总数 (订单数)

        self.box_size_tensor = torch.tensor(
            [WORK_BOX_PARAMS['X_LENGTH'], WORK_BOX_PARAMS['Y_LENGTH'], WORK_BOX_PARAMS['Z_LENGTH']], 
            device=self.device
        )

        #核心映射
        #每个订单箱中，每种SKU最多需要几个
        self.target_need_sku_num = torch.full(
            (self.num_envs,self.num_targets,self.num_skus), 0, dtype=torch.long, device=self.device
        )

        # [核心映射 2] 物品 -> 应该从哪个原料箱生成？
        # 值范围：0 ~ num_sources-1  -1代表该物品本局不考虑
        self.obj_to_source_id = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
        )

        # is_active: 本局是否出现 (ID != -1)
        self.is_active_mask = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
        )
        # is_target: 本局是否为目标 (是 active 且不是干扰物)
        self.is_target_mask = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
        )

        # 记录每个物品的状态：0=待生成, 1=待处理, 2=抓取中, 3=已完成, 4=失败
        self.object_states = torch.zeros(
            (self.num_envs, self.num_objects),  dtype=torch.long, device=self.device
        )

        self.order_completion = torch.zeros(
            (self.num_envs, self.num_targets), dtype=torch.bool, device=self.device
        )

        self.log_path = f"{get_logs_path()}/{int(time.time())}.jsonl"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        #--------待check------ 场景持久化相关代码
        # --- 1. 基础属性初始化---
        self.from_json = getattr(cfg, "from_json", 2)
        self.obstacle_names = getattr(cfg, "obstacles", [])
        self.obstacle_assets = [self.env.scene[name] for name in self.obstacle_names if name in self.env.scene.keys()]
        
        # 预定义回放相关变量，确保在所有模式下都存在这些属性
        self.env_replay_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.replay_data_pool = []
        self.num_replay_configs = 0
        self._current_obs_scales = {} # 用于存储当前生成的 scale，解决 JSON 记录延迟问题

        # --- 2. 动态获取任务对应的唯一文件名 ---
        self.task_name = env.cfg.__class__.__name__
        self.task_filename = f"{self.task_name}.jsonl"
        self.session_file = get_env_order_info_path().joinpath(self.task_filename)

        # --- 3. 根据模式执行特定初始化 ---
        if self.from_json == 1:
            # --- 消费者模式 (Replay) ---
            if not self.session_file.exists():
                raise FileNotFoundError(f"未找到回放文件: {self.session_file}")
            
            with open(self.session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.replay_data_pool.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            self.num_replay_configs = len(self.replay_data_pool)
            if self.num_replay_configs == 0:
                raise ValueError(f"回放文件 {self.session_file} 内容为空！")
                
            print(f"[Info]顺序回放模式: 已加载 {self.num_replay_configs} 条场景记录")
            
        elif self.from_json == 0:
            # --- 生产者模式 (Record) ---
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            # 清空旧文件
            with open(self.session_file, 'w', encoding='utf-8') as f:
                pass 
            print(f"[Info]生产者模式: 已重置并清空文件 {self.task_filename}，开始重新生成...")
            
        else:
            # --- 纯随机模式 (Default) ---
            print(f"[Info]纯随机模式: from_json={self.from_json}，不记录也不读取 JSONL")


    def _discover_object_instances(self, env: ManagerBasedRLEnv, sku_list: Sequence[str]):
        """
        遍历 env.scene.keys()，寻找以 sku_name 开头的所有物体实例。
        假设实例命名规范为: "{sku_name}_{index}" 或 "{sku_name}_X"
        """
        scene_keys = list(env.scene.keys())
        current_global_idx = 0

        for sku_name in sku_list:
            self.sku_to_indices[sku_name] = []

            found_instances = []
            for key in scene_keys:
                # 我们要求 key 必须是 "{sku_name}" 或者 "{sku_name}_" 开头
                if key == sku_name or key.startswith(f"{sku_name}_"):
                    found_instances.append(key)

            found_instances.sort()

            if not found_instances:
                print(f"[Warning] No instances found for SKU: {sku_name} in env.scene!")
                continue

            for instance_name in found_instances:
                # 添加名字
                self.object_names.append(instance_name)
                
                # 添加资产引用
                self.object_assets.append(env.scene[instance_name])
                
                # 记录映射关系 (Global Index)
                self.sku_to_indices[sku_name].append(current_global_idx)
                
                current_global_idx += 1

    def __str__(self) -> str: 
        msg = f"该次任务共有{self.num_targets}笔订单，{self.num_objects}个SKU"
        return msg
    
    def _resample_command(self, env_ids: Sequence[int]):
        # 保存动态指标
        self._save_dynamic_metrics(env_ids)
        
        #-----check 场景持久化 ------
        if self.from_json == 1:
            # 只有模式 1 才执行回放
            self._resample_from_json(env_ids)
            # 重置状态
            self.order_completion[env_ids] = False
            self.object_states[env_ids] = 0
            active_mask = self.is_active_mask[env_ids]
            self.object_states[env_ids] = self.object_states[env_ids].masked_fill(active_mask, 1)
        else:
            # 模式 0 (Record) 和 模式 2 (Random) 都走正常生成流程
            self._assign_objects_boxes(env_ids)
            self.order_completion[env_ids] = False
            self._spawn_items_in_source_boxes(env_ids)
            
            self.object_states[env_ids] = 0
            active_mask = self.is_active_mask[env_ids]
            self.object_states[env_ids] = self.object_states[env_ids].masked_fill(active_mask, 1)
            
            # 只有模式 0 才执行记录
            if self.from_json == 0:
                self._record_order_env_info(env_ids)

    @abstractmethod
    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        raise NotImplementedError
        
    @abstractmethod
    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        raise NotImplementedError

    #----check 场景持久化 -----
    def _record_order_env_info(self, env_ids: Sequence[int]):
        """
        记录环境信息：将坐标转换为环境局部坐标并存入任务专属的 JSONL。
        存储逻辑：Local_Pos = World_Pos - Env_Origin，增加对障碍物 Scale 的记录以适配随机化。
        """

        # --- 1. 准备环境 ID 列表 ---
        if isinstance(env_ids, torch.Tensor):
            ids_list = env_ids.tolist()
            env_ids_tensor = env_ids
        else:
            ids_list = list(env_ids)
            env_ids_tensor = torch.tensor(env_ids, device=self.device)

        # 获取当前批次环境的世界原点坐标 (Shape: [len(ids_list), 3])
        batch_origins = self.env.scene.env_origins[env_ids_tensor]

        # --- 2. 以追加模式打开文件 ---
        with open(self.session_file, 'a', encoding='utf-8') as f:
            for i, env_id in enumerate(ids_list):
                current_origin = batch_origins[i]

                # --- 3. 构造基础数据结构 ---
                record = {
                    "timestamp": time.time(),
                    "env_id": env_id,
                    "task_name": self.task_name,
                    "obj_to_source_id": self.obj_to_source_id[env_id].tolist(),
                    "obj_to_target_id": self.obj_to_target_id[env_id].tolist(),
                    "containers": {
                        "source_boxes": [
                            {
                                "name": self.cfg.source_boxes[j],
                                "pos": (self.source_box_assets[j].data.root_pos_w[env_id] - current_origin).tolist(),
                                "rot": self.source_box_assets[j].data.root_quat_w[env_id].tolist()
                            } for j in range(len(self.source_box_assets))
                        ],
                        "target_boxes": [
                            {
                                "name": self.cfg.target_boxes[j],
                                "pos": (self.target_box_assets[j].data.root_pos_w[env_id] - current_origin).tolist(),
                                "rot": self.target_box_assets[j].data.root_quat_w[env_id].tolist()
                            } for j in range(len(self.target_box_assets))
                        ]
                    },
                    "items": [],
                    "obstacles": [], # 占位符
                    "orders": []
                }

                # --- 4. 填充物品明细 ---
                for obj_idx, obj_name in enumerate(self.object_names):
                    is_active = self.is_active_mask[env_id, obj_idx].item()
                    s_idx = self.obj_to_source_id[env_id, obj_idx].item()
                    t_idx = self.obj_to_target_id[env_id, obj_idx].item()
                    
                    role = "target" if (is_active and t_idx != -1) else ("distractor" if is_active else "none")

                    if is_active:
                        world_pos = self.object_assets[obj_idx].data.root_pos_w[env_id]
                        local_pos = (world_pos - current_origin).tolist()
                        local_rot = self.object_assets[obj_idx].data.root_quat_w[env_id].tolist()
                    else:
                        local_pos = None
                        local_rot = None

                    record["items"].append({
                        "instance_name": obj_name,
                        "is_active": is_active,
                        "logic_role": {"type": role, "source_box_idx": s_idx, "target_box_idx": t_idx},
                        "spawn_pose": {
                            "pos": local_pos,
                            "rot": local_rot
                        } if is_active else None
                    })

                # --- 5. 记录障碍物 (适配随机化尺寸) ---
                # 只有当子类定义了障碍物资产时才进行记录
                if hasattr(self, "obstacle_assets") and self.obstacle_assets:
                    for j, asset in enumerate(self.obstacle_assets):
                        # 获取世界位姿
                        w_pos = asset.data.root_pos_w[env_id]
                        w_quat = asset.data.root_quat_w[env_id]
                        
                        # 如果资产尚未完全初始化或不支持 scale，则默认为 [1.0, 1.0, 1.0]
                        if hasattr(asset.data, "root_scale") and asset.data.root_scale is not None:
                            current_scale = asset.data.root_scale[env_id].tolist()
                        else:
                            current_scale = [1.0, 1.0, 1.0]

                        record["obstacles"].append({
                            "instance_name": self.obstacle_names[j],
                            "scale": current_scale, # 存储缩放信息，用于 resample 还原
                            "spawn_pose": {
                                "pos": (w_pos - current_origin).tolist(),
                                "rot": w_quat.tolist()
                            }
                        })

                # --- 6. 填充订单明细 ---
                for t_idx in range(self.num_targets):
                    mask = (self.obj_to_target_id[env_id] == t_idx)
                    if mask.any():
                        match_indices = torch.where(mask)[0].tolist()
                        s_idx = self.obj_to_source_id[env_id, match_indices[0]].item()
                        record["orders"].append({
                            "target_box_name": self.cfg.target_boxes[t_idx],
                            "source_box_index": s_idx,
                            "target_box_index": t_idx,
                            "required_items": [self.object_names[idx] for idx in match_indices]
                        })

                # --- 7. 写入 JSONL ---
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _resample_from_json(self, env_ids: Sequence[int]):
        """
        核心回放函数：从预加载的 JSON/JSONL 数据池中顺序读取配置，
        并精确还原环境中的物体状态（包含随机化后的障碍物尺寸）。
        """
        # 确保 env_ids 是 Tensor 格式
        if not isinstance(env_ids, torch.Tensor):
            env_ids_batch = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_batch = env_ids

        for env_id in env_ids_batch:
            # --- 1. 顺序从池子中抽取索引 ---
            current_idx = self.env_replay_ptr[env_id].item()
            data = self.replay_data_pool[current_idx]

            # --- 2. 恢复逻辑映射 (Tensor) ---
            self.obj_to_source_id[env_id] = torch.tensor(data["obj_to_source_id"], device=self.device)
            self.obj_to_target_id[env_id] = torch.tensor(data["obj_to_target_id"], device=self.device)
            self.is_active_mask[env_id] = (self.obj_to_source_id[env_id] != -1)

            # 构造用于底层接口的批次 ID (Shape: [1])
            single_env_id = env_id.unsqueeze(0) if env_id.dim() == 0 else env_id

            # --- 3. 还原物品 (is_active/is_not_active) ---
            for obj_idx, item_data in enumerate(data["items"]):
                asset = self.object_assets[obj_idx]
                
                if item_data["is_active"] and item_data["spawn_pose"]:
                    pos = torch.tensor(item_data["spawn_pose"]["pos"], device=self.device).unsqueeze(0)
                    rot = torch.tensor(item_data["spawn_pose"]["rot"], device=self.device).unsqueeze(0)
                    
                    set_asset_position(
                        env=self.env,
                        env_ids=single_env_id,
                        asset=asset,
                        position=pos,
                        quat=rot
                    )
                else:
                    # 非活跃物品：隐藏到地下深处并清空速度
                    far_away_pos = torch.tensor([[0.0, 0.0, -100.0]], device=self.device)
                    set_asset_position(self.env, single_env_id, asset, far_away_pos)
                    if hasattr(asset, "write_root_velocity_to_sim"):
                        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=single_env_id)

            # --- 4. 还原障碍物 ---
            obstacles_data = data.get("obstacles", [])
            for obs in obstacles_data:
                obs_name = obs["instance_name"]
                if obs_name in self.env.scene.keys():
                    asset = self.env.scene[obs_name]
                    
                    # A. 还原缩放 (Scale) - 必须在位姿还原之前或同时进行，以确保物理边界正确
                    if "scale" in obs and hasattr(asset, "write_root_scale_to_sim"):
                        target_scale = torch.tensor(obs["scale"], device=self.device).unsqueeze(0)
                        asset.write_root_scale_to_sim(target_scale, env_ids=single_env_id)
                    
                    # B. 还原位姿 (Pose)
                    pos = torch.tensor(obs["spawn_pose"]["pos"], device=self.device)
                    rot = torch.tensor(obs["spawn_pose"]["rot"], device=self.device)
                    world_pos = pos + self.env.scene.env_origins[env_id]
                    root_pose = torch.cat([world_pos, rot]).unsqueeze(0)
                    
                    asset.write_root_pose_to_sim(root_pose, env_ids=single_env_id)
                    
                    # C. 速度清零：防止上一局残留的动量影响新一局
                    asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=single_env_id)
                    
                    # D. 强制同步内部数据缓冲区 (data.root_pos_w 等)
                    if hasattr(asset, "reset"):
                        asset.reset(env_ids=single_env_id)
            
            # --- 5. 更新指针：指向下一个场景配置 ---
            self.env_replay_ptr[env_id] = (current_idx + 1) % self.num_replay_configs

    @abstractmethod
    def _update_object_states(self):
        raise NotImplementedError
    
    def _check_objects_in_source_box(self, env_ids):
        """
            return (N,O) O = len(objects),N = len(env_ids)
            N个环境中,每个物体是否在其原来分配的原料箱内
            如某物体未出现，依然为False
        """
        results = []

        # 获取分配规则 (N, O)
        # source_ids_all[:, i] 代表第 i 个物体应该在哪个原料箱
        source_ids_all = self.obj_to_source_id[env_ids]

        # --- 外层循环：遍历每一个物体 (SKU) ---
        for i, obj_asset in enumerate(self.object_assets):
            # 1. 拿到该物体"应该"去的原料箱 ID (N, 1) -> (N,)
            target_source_id = source_ids_all[:, i]
            
            # 初始化该物体的结果列 (N,)
            # 默认 False
            obj_in_correct_source = torch.zeros_like(target_source_id, dtype=torch.bool)

            # --- 内层循环：遍历所有原料箱 ---
            for k in range(self.num_sources):
                relevant_mask = (target_source_id == k)
                # 剪枝：如果没有任何一个环境符合条件，完全跳过
                if not relevant_mask.any():
                    continue

                # 只拿出那些 "确实需要检测" 的环境 ID
                subset_env_ids = env_ids[relevant_mask]

                # A. 物理判定：物体 i 是否在 原料箱 k 里？返回 (N`,)
                is_in_subset = check_object_in_box(
                    subset_env_ids, 
                    obj_asset,                 # 单个物体资产
                    self.source_box_assets[k], # 单个箱子资产
                    box_size = self.box_size_tensor
                )
                
                obj_in_correct_source[relevant_mask] = is_in_subset

            # 将该物体在所有环境的结果存入列表
            results.append(obj_in_correct_source)

        # --- 最终拼接 ---
        # 列表里有 O 个 (N,) 的张量 -> 堆叠成 (N, O)
        return torch.stack(results, dim=1)
    
    def _update_metrics(self):
        self._update_object_states()
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

    @abstractmethod
    def _update_command(self):
        raise NotImplementedError

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
