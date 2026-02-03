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

#全局开关（是否需要根据JSON文件生成场景）
FROM_JSON = 0

class OrderCommandTerm(CommandTerm):
    cfg: OrderCommandTermCfg

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.object_names = []
        self.object_assets:list[RigidObject | Articulation] = []
        self.sku_to_indices :dict[str,list] = {} 

        # 输入: cfg.objects 列表只包含 SKU 名称，例如 ["cracker_box", "sugar_box"]
        self.sku_names = cfg.objects # 记录原始 SKU 清单

        self._discover_object_instances(env, cfg.objects)

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

        # [核心映射 1] 物品 -> 应该去哪个订单箱？
        # 值范围：0 ~ num_targets-1。如果值为 -1 代表该物品本局是干扰物。
        self.obj_to_target_id = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
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

        #---1. 动态获取任务对应的唯一文件名---
        self.task_name = env.cfg.__class__.__name__
        self.task_filename = f"{self.task_name}.jsonl"
        self.session_file = get_env_order_info_path().joinpath(self.task_filename)

        self.from_json = FROM_JSON
        if self.from_json == 1:
            #---消费者模式：读取现有文件---
            if not self.session_file.exists():
                raise FileNotFoundError(f"未找到回放文件: {self.session_file}")
            
            self.replay_data_pool = []
            with open(self.session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.replay_data_pool.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            self.num_replay_configs = len(self.replay_data_pool)
            print(f"[Info]顺序回放模式: 已加载{self.num_replay_configs}条场景记录")
            self.env_replay_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            
        else:
            #---生产者模式：每次运行代码时覆盖旧文件---
            #确保父目录存在
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            
            #使用'w'模式打开并立即关闭，这会清空文件内容
            with open(self.session_file, 'w', encoding='utf-8') as f:
                pass 
        
            print(f"[Info]生产者模式: 已重置并清空文件{self.task_filename}，开始重新生成...")

    
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
        self._save_dynamic_metrics(env_ids)
        if self.from_json == 0:
            self._assign_objects_boxes(env_ids)

            #重置订单箱完成状态
            self.order_completion[env_ids] = False
            self._spawn_items_in_source_boxes(env_ids)
            self.object_states[env_ids] = 0

            #[修正]只把Active的设为1 (Pending)
            active_mask = self.is_active_mask[env_ids]
            self.object_states[env_ids] = self.object_states[env_ids].masked_fill(active_mask, 1)
            self._record_order_env_info(env_ids) 

        else:
            self._resample_from_json(env_ids)
            self.order_completion[env_ids] = False
            self.object_states[env_ids] = 0
            active_mask = self.is_active_mask[env_ids]
            self.object_states[env_ids] = self.object_states[env_ids].masked_fill(active_mask, 1)


    @abstractmethod
    def _assign_objects_boxes(self, env_ids: Sequence[int]):
        raise NotImplementedError
        
    @abstractmethod
    def _spawn_items_in_source_boxes(self, env_ids: Sequence[int]):
        raise NotImplementedError
    
    def _resample_from_json(self, env_ids: Sequence[int]):
        """
        核心回放函数：从预加载的JSON/JSONL数据池中顺序读取配置，并精确还原环境中的物体状态。
        """
        # 确保env_ids是Tensor格式，方便后续计算
        if not isinstance(env_ids, torch.Tensor):
            env_ids_batch = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_batch = env_ids

        for env_id in env_ids_batch:
            #---1. 顺序从池子中抽取索引---
            #获取当前环境应该回放的索引
            current_idx = self.env_replay_ptr[env_id].item()
            data = self.replay_data_pool[current_idx]

            #---2. 恢复逻辑映射(Tensor)---
            #必须直接赋值给对应的env_id行
            self.obj_to_source_id[env_id] = torch.tensor(data["obj_to_source_id"], device=self.device)
            self.obj_to_target_id[env_id] = torch.tensor(data["obj_to_target_id"], device=self.device)
            self.is_active_mask[env_id] = (self.obj_to_source_id[env_id] != -1)

            #---3. 恢复物理位姿---
            #构造一个形状为(1,)的Tensor，因为后续功能函数底层期望接收一个批次的ID
            single_env_id = torch.tensor([env_id], device=self.device, dtype=torch.long)

            for obj_idx, item_data in enumerate(data["items"]):
                asset = self.object_assets[obj_idx]
                
                if item_data["is_active"] and item_data["spawn_pose"]:
                    #准备位置和旋转Tensor，形状为(1, 3)和(1, 4)
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
                    #对于本局不需要出现的非活跃物体，将其位置设在地下深处(z=-10.0)，避免碰撞干扰
                    far_away_pos = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
                    set_asset_position(
                        env=self.env,
                        env_ids=single_env_id,
                        asset=asset,
                        position=far_away_pos
                    )
            #---4. 更新指针：确保下一次重置时指向池子里的下一个场景---
            #使用当前索引加1并对总场景数取模，实现循环顺序回放(1, 2, ..., N, 1, ...)
            self.env_replay_ptr[env_id] = (current_idx + 1) % self.num_replay_configs
                

    def _record_order_env_info(self, env_ids: Sequence[int]):
        """
        高效记录环境信息：通过追加模式写入任务专属的 JSONL 文件。
        此函数捕获逻辑 ID 映射及资产的绝对世界位姿，实现 100% 场景还原。
        """
        #--- 1. 准备环境 ID 列表---
        if isinstance(env_ids, torch.Tensor):
            ids_list = env_ids.tolist()
        else:
            ids_list = list(env_ids)

        # --- 2. 以追加模式 ('a') 打开文件 ---
        # 注意：__init__ 中已通过 'w' 模式清空过旧运行数据，
        # 这里的 'a' 确保在同一次代码运行过程中的多次 reset 记录能按顺序累积。
        with open(self.session_file, 'a', encoding='utf-8') as f:
            for env_id in ids_list:
                # --- 3. 构造基础数据结构 (Snapshot) ---
                record = {
                    "timestamp": time.time(),
                    "env_id": env_id,
                    "task_name": self.task_name,
                    # 核心逻辑 ID 映射：还原物品归属的关键
                    "obj_to_source_id": self.obj_to_source_id[env_id].tolist(),
                    "obj_to_target_id": self.obj_to_target_id[env_id].tolist(),
                    # 箱体物理位姿记录（用于验证或回放时校验箱子是否移动）
                    "containers": {
                        "source_boxes": [
                            {
                                "name": self.cfg.source_boxes[i],
                                "pos": self.source_box_assets[i].data.root_pos_w[env_id].tolist(),
                                "rot": self.source_box_assets[i].data.root_quat_w[env_id].tolist()
                            } for i in range(len(self.source_box_assets))
                        ],
                        "target_boxes": [
                            {
                                "name": self.cfg.target_boxes[i],
                                "pos": self.target_box_assets[i].data.root_pos_w[env_id].tolist(),
                                "rot": self.target_box_assets[i].data.root_quat_w[env_id].tolist()
                            } for i in range(len(self.target_box_assets))
                        ]
                    },
                    "items": [],
                    "orders": []
                }

                # --- 4. 填充物品明细 (绝对位姿提取) ---
                for obj_idx, obj_name in enumerate(self.object_names):
                    is_active = self.is_active_mask[env_id, obj_idx].item()
                    s_idx = self.obj_to_source_id[env_id, obj_idx].item()
                    t_idx = self.obj_to_target_id[env_id, obj_idx].item()
                    
                    # 确定逻辑角色：target (目标物), distractor (干扰物), none (不活跃)
                    role = "none"
                    if is_active:
                        role = "target" if t_idx != -1 else "distractor"

                    # 提取该环境下的资产位姿
                    asset = self.object_assets[obj_idx]
                    # 获取绝对世界坐标 (root_pos_w / root_quat_w)
                    pos_w = asset.data.root_pos_w[env_id].tolist()
                    quat_w = asset.data.root_quat_w[env_id].tolist()

                    record["items"].append({
                        "instance_name": obj_name,
                        "sku_type": obj_name.rsplit('_', 1)[0] if '_' in obj_name else obj_name,
                        "is_active": is_active,
                        "logic_role": {"type": role, "source_box_idx": s_idx, "target_box_idx": t_idx},
                        "spawn_pose": {
                            "pos": pos_w,
                            "rot": quat_w
                        }
                    })

                # --- 5. 填充订单明细 ---
                for t_idx in range(self.num_targets):
                    mask = (self.obj_to_target_id[env_id] == t_idx)
                    if mask.any():
                        match_indices = torch.where(mask)[0].tolist()
                        # 取第一个物品的源箱作为该订单的关联源箱
                        s_idx = self.obj_to_source_id[env_id, match_indices[0]].item()
                        record["orders"].append({
                            "target_box_name": self.cfg.target_boxes[t_idx],
                            "target_box_index": t_idx,
                            "source_box_index": s_idx,
                            "required_items": [self.object_names[idx] for idx in match_indices]
                        })

                # --- 6. 核心写入操作 ---
                # 将 record 字典序列化为单行 JSON 并追加到文件，确保 JSONL 格式正确
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        f.flush()


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
