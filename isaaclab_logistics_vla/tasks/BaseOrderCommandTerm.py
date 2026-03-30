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
    from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg


from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *
from isaaclab_logistics_vla.utils.path_utils import *

def get_image_from_sensor(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, data_type: str):
    """
    从指定的相机传感器获取图像张量，返回的Tensor形状为(num_envs, H, W, C)
    """
    return env.scene.sensors[sensor_cfg.name].data.output[data_type]

class BaseOrderCommandTerm(CommandTerm):
    cfg: OrderCommandTermCfg

    def __init__(self, cfg: OrderCommandTermCfg, env: ManagerBasedRLEnv,is_multi_target:bool = False):
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
            (self.num_envs,self.num_targets,self.num_skus), -1, dtype=torch.long, device=self.device
        )

        #每个订单的状态 每个订单箱中，每种SKU实际上现有几个
        self.target_contain_sku_num = torch.full(
            (self.num_envs,self.num_targets,self.num_skus), 0, dtype=torch.long, device=self.device
        )
        self.last_target_contain_sku_num = torch.zeros_like(self.target_contain_sku_num)
            
        # 值范围：0 ~ num_sources-1  -1代表该物品本局不考虑
        self.obj_to_source_id = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
        )

        # is_active: 本局是否出现 (ID != -1)
        self.is_active_mask = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
        )

        # 每个物品的状态：-1 = 不出现 0 = 在抓取中 1 2 3 = 原料箱123 4 5 6 = 目标箱123 10 = 物理上失败
        self.object_states = torch.full(
            (self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device
        )

        self.last_object_states = torch.full_like(self.object_states, -1)

        self.order_completion = torch.zeros(
            (self.num_envs, self.num_targets), dtype=torch.bool, device=self.device
        )
        self.eval_metrics = {}
        self.log_path = f"{get_logs_path()}/{self.__str__()}_{int(time.time())}.jsonl"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        #--------待check------ 场景持久化相关代码
        # --- 1. 基础属性初始化---
        self.from_json = getattr(cfg, "from_json", 2)
        self.obstacle_names = getattr(cfg, "obstacles", []) or []
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

    @abstractmethod
    def __str__(self) -> str:     #当前任务的名称
        pass
    
    def _resample_command(self, env_ids: Sequence[int]):
        self.target_need_sku_num[env_ids] = -1
        self.target_contain_sku_num[env_ids] = 0
        self.order_completion[env_ids] = False
        self.object_states[env_ids] = -1
        #-----check 场景持久化 ------
        if self.from_json == 1:
            # 只有模式 1 才执行回放
            self._resample_from_json(env_ids)
            # 重置状态
            
            active_mask = self.is_active_mask[env_ids]
            self.object_states[env_ids] = self.object_states[env_ids].masked_fill(active_mask, 1)
        else:
            # 模式 0 (Record) 和 模式 2 (Random) 都走正常生成流程
            self._assign_objects_boxes(env_ids)      
            self._spawn_items_in_source_boxes(env_ids)
            
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

    #----check 场景持久化----
    def _record_order_env_info(self, env_ids: Sequence[int]):
        """
            记录环境信息：将坐标转换为环境局部坐标并存入任务专属的JSONL
            移除obj_to_target_id依赖，基于target_need_sku_num进行SKU级的精准追踪
        """
        if isinstance(env_ids, torch.Tensor):
            ids_list = env_ids.tolist()
            env_ids_tensor = env_ids
        else:
            ids_list = list(env_ids)
            env_ids_tensor = torch.tensor(env_ids, device=self.device)

        batch_origins = self.env.scene.env_origins[env_ids_tensor]

        #预构建obj_idx到sku_idx的反向映射字典，将时间复杂度降为O(1)
        obj_idx_to_sku_idx = {}
        for s_idx, sku_name in enumerate(self.sku_names):
            for o_idx in self.sku_to_indices.get(sku_name, []):
                obj_idx_to_sku_idx[o_idx] = s_idx

        with open(self.session_file, 'a', encoding='utf-8') as f:
            for i, env_id in enumerate(ids_list):
                current_origin = batch_origins[i]

                #---1. 构造基础数据结构---
                record = {
                    "timestamp": time.time(),
                    "env_id": env_id,
                    "task_name": self.task_name,
                    "obj_to_source_id": self.obj_to_source_id[env_id].tolist(),
                    "target_need_sku_num": self.target_need_sku_num[env_id].tolist(),
                    "containers": {
                        "source_boxes": [
                            {
                                "name": self.cfg.source_boxes[j],
                                "pos": (self.source_box_assets[j].data.root_pos_w[env_id] - current_origin).tolist(),
                                "rot": self.source_box_assets[j].data.root_quat_w[env_id].tolist()
                            } for j in range(self.num_sources)
                        ],
                        "target_boxes": [
                            {
                                "name": self.cfg.target_boxes[j],
                                "pos": (self.target_box_assets[j].data.root_pos_w[env_id] - current_origin).tolist(),
                                "rot": self.target_box_assets[j].data.root_quat_w[env_id].tolist()
                            } for j in range(self.num_targets)
                        ]
                    },
                    "items": [],
                    "obstacles": [],
                    "orders": []
                }

                #获取该环境对所有SKU的总需求量(Shape: [num_skus])
                env_total_sku_needs = self.target_need_sku_num[env_id].sum(dim=0)

                #---2. 填充物品明细---
                for obj_idx, obj_name in enumerate(self.object_names):
                    is_active = self.is_active_mask[env_id, obj_idx].item()
                    s_idx = self.obj_to_source_id[env_id, obj_idx].item()
                    
                    #逻辑判定：如果该物体激活，且它所属的SKU在本环境有任何需求(>0)，它就是target
                    #否则，它就是纯粹的distractor(干扰物)
                    if is_active:
                        sku_idx = obj_idx_to_sku_idx.get(obj_idx, -1)
                        if sku_idx != -1 and env_total_sku_needs[sku_idx].item() > 0:
                            role = "target"
                        else:
                            role = "distractor"
                            
                        world_pos = self.object_assets[obj_idx].data.root_pos_w[env_id]
                        local_pos = (world_pos - current_origin).tolist()
                        local_rot = self.object_assets[obj_idx].data.root_quat_w[env_id].tolist()
                    else:
                        role = "none"
                        local_pos = None
                        local_rot = None

                    record["items"].append({
                        "instance_name": obj_name,
                        "is_active": is_active,
                        #注意：target_box_idx已废弃，因为实例不再绑定单一目标箱，统一设为-1
                        "logic_role": {"type": role, "source_box_idx": s_idx, "target_box_idx": -1},
                        "spawn_pose": {
                            "pos": local_pos,
                            "rot": local_rot
                        } if is_active else None
                    })

                #---3. 记录障碍物---
                if hasattr(self, "obstacle_assets") and self.obstacle_assets:
                    for j, asset in enumerate(self.obstacle_assets):
                        w_pos = asset.data.root_pos_w[env_id]
                        w_quat = asset.data.root_quat_w[env_id]
                        
                        if hasattr(asset.data, "root_scale") and asset.data.root_scale is not None:
                            current_scale = asset.data.root_scale[env_id].tolist()
                        else:
                            current_scale = [1.0, 1.0, 1.0]

                        record["obstacles"].append({
                            "instance_name": self.obstacle_names[j],
                            "scale": current_scale,
                            "spawn_pose": {
                                "pos": (w_pos - current_origin).tolist(),
                                "rot": w_quat.tolist()
                            }
                        })

                #---4. 填充订单明细---
                for t_idx in range(self.num_targets):
                    need_array = self.target_need_sku_num[env_id, t_idx]
                    
                    if (need_array > 0).any():
                        required_skus = {}
                        for s_idx, sku_name in enumerate(self.sku_names):
                            count = need_array[s_idx].item()
                            if count > 0:
                                required_skus[sku_name] = count

                        record["orders"].append({
                            "target_box_name": self.cfg.target_boxes[t_idx],
                            "target_box_index": t_idx,
                            "required_skus": required_skus
                        })

                #---5. 写入JSONL---
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _resample_from_json(self, env_ids: Sequence[int]):
        """
            核心回放函数：从预加载的JSON/JSONL数据池中顺序读取配置，
            完美还原target_need_sku_num以及物理状态
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids_batch = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_batch = env_ids

        for env_id in env_ids_batch:
            current_idx = self.env_replay_ptr[env_id].item()
            data = self.replay_data_pool[current_idx]

            #---1. 恢复核心逻辑映射(完全剔除obj_to_target_id)---
            self.obj_to_source_id[env_id] = torch.tensor(data["obj_to_source_id"], device=self.device)
            self.is_active_mask[env_id] = (self.obj_to_source_id[env_id] != -1)
            
            #关键恢复：将JSON里的多维列表转换回GPU Tensor，确保后续Metrics评分正常！
            if "target_need_sku_num" in data:
                self.target_need_sku_num[env_id] = torch.tensor(
                    data["target_need_sku_num"], device=self.device, dtype=torch.long
                )
            else:
                #兼容非常老的JSONL，防止崩溃
                self.target_need_sku_num[env_id] = 0

            single_env_id = env_id.unsqueeze(0) if env_id.dim() == 0 else env_id

            #---2. 还原物品位姿---
            for obj_idx, item_data in enumerate(data["items"]):
                asset = self.object_assets[obj_idx]
                
                if item_data["is_active"] and item_data["spawn_pose"]:
                    pos = torch.tensor(item_data["spawn_pose"]["pos"], device=self.device).unsqueeze(0)
                    rot = torch.tensor(item_data["spawn_pose"]["rot"], device=self.device).unsqueeze(0)
                    
                    set_asset_position(self.env, single_env_id, asset, pos, rot)
                else:
                    far_away_pos = torch.tensor([[0.0, 0.0, -100.0]], device=self.device)
                    set_asset_position(self.env, single_env_id, asset, far_away_pos)
                    if hasattr(asset, "write_root_velocity_to_sim"):
                        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=single_env_id)

            #---3. 还原障碍物---
            obstacles_data = data.get("obstacles", [])
            for obs in obstacles_data:
                obs_name = obs["instance_name"]
                if obs_name in self.env.scene.keys():
                    asset = self.env.scene[obs_name]
                    
                    if "scale" in obs and hasattr(asset, "write_root_scale_to_sim"):
                        target_scale = torch.tensor(obs["scale"], device=self.device).unsqueeze(0)
                        asset.write_root_scale_to_sim(target_scale, env_ids=single_env_id)
                    
                    pos = torch.tensor(obs["spawn_pose"]["pos"], device=self.device)
                    rot = torch.tensor(obs["spawn_pose"]["rot"], device=self.device)
                    world_pos = pos + self.env.scene.env_origins[env_id]
                    root_pose = torch.cat([world_pos, rot]).unsqueeze(0)
                    
                    asset.write_root_pose_to_sim(root_pose, env_ids=single_env_id)
                    asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=single_env_id)
                    
                    if hasattr(asset, "reset"):
                        asset.reset(env_ids=single_env_id)
            
            #---4. 更新指针---
            self.env_replay_ptr[env_id] = (current_idx + 1) % self.num_replay_configs

    def _update_object_states(self):
        env_ids = torch.tensor(range(self.num_envs),dtype =torch.int32, device=self.device)

        # 1. 基础掩码获取 (Shape: [N, O])
        active_mask = self.is_active_mask
        failed_mask = self._check_objects_failure(env_ids)
        valid_mask = active_mask & (~failed_mask)

        # 2. 初始化状态数组
        # 默认全部为 -1 (未出现)
        new_states = torch.full((self.num_envs, self.num_objects), -1, dtype=torch.long, device=self.device)

        # 激活且未失败的物体，默认状态设为 0 (抓取中/运输中)
        new_states[valid_mask] = 0

        # 物理失败的物体，状态直接设为 10
        failed_active_mask = active_mask & failed_mask
        new_states[failed_active_mask] = 10

        # 3. 组织所有的箱子资产和它们对应的状态 ID
        all_boxes_info = []
        for k in range(self.num_sources):
            all_boxes_info.append((self.source_box_assets[k], k + 1))
        for k in range(self.num_targets):
            all_boxes_info.append((self.target_box_assets[k], self.num_sources + k + 1))

        # 4. 遍历物体进行物理判定 (调用你的原函数)
        for i, obj_asset in enumerate(self.object_assets):
            # 取出第 i 个物体在所有环境中的“有效”掩码 (N,)
            obj_valid_mask = valid_mask[:, i]

            # 极速剪枝：如果这个物体在所有环境中都没生成或都掉落了，直接跳过，节省算力
            if not obj_valid_mask.any():
                continue

            # 仅仅提取出该物体依然有效的环境 ID (N`,)
            valid_env_ids = env_ids[obj_valid_mask]

            for box_asset, state_id in all_boxes_info:
                # 调用你验证过的原函数，返回布尔值 (N`,)
                is_in_box = check_object_in_box(
                    valid_env_ids,
                    obj_asset,
                    box_asset,
                    self.box_size_tensor
                )

                # 如果在某些环境里，这个物体确实在这个箱子里
                if is_in_box.any():
                    # 映射回原始的 env_ids
                    hit_env_ids = valid_env_ids[is_in_box]
                    # 覆盖这些环境下的物体状态
                    new_states[hit_env_ids, i] = state_id
            
        # 5. 更新全局状态
        self.object_states = new_states

    def _check_objects_failure(self, env_ids):
        is_failed = torch.zeros((len(env_ids), self.num_objects), dtype=torch.bool, device=self.device)

        for i, obj_asset in enumerate(self.object_assets):
            z_pos = obj_asset.data.root_pos_w[env_ids, 2]
            # 阈值设为 0.3 (原料箱在 0.75，稍微掉下来一点不算，必须是掉到地上)
            is_failed[:, i] |= (z_pos < 0.3)

        return is_failed

    def _update_metrics(self):
        # 将上一步的状态备份
        self.last_object_states = self.object_states.clone()
        self.last_target_contain_sku_num = self.target_contain_sku_num.clone()

        self._update_object_states()
        self._update_target_states()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            # 如果是 None (全局初始化)，直接显式地生成所有环境的 ID 列表
            env_ids = list(range(self.num_envs))

        if len(env_ids) > 0 and self.env.episode_length_buf[env_ids][0]>1:    #不是最初初始化的reset
            if 'episode_physics_steps' not in self.eval_metrics:
                self.eval_metrics['episode_physics_steps'] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.eval_metrics['episode_physics_steps'][env_ids] = self.env.episode_length_buf[env_ids].float()
            self._compute_total_metrics(env_ids)

            # self._update_spawn_metrics(env_ids)
            self._save_dynamic_metrics(env_ids)
        info = super().reset(env_ids)

        # 为了严谨，重置后立刻更新一次当前状态，获取最干净的初始态
        self._update_object_states()
        self._update_target_states()

        # 强行抹平历史差异！这样新回合第一帧的 reward delta 就是 0
        self.last_object_states[env_ids] = self.object_states[env_ids].clone()
        self.last_target_contain_sku_num[env_ids] = self.target_contain_sku_num[env_ids].clone()

        return info

    def _update_target_states(self):
        """
        根据最新的 self.object_states，统计每个目标箱中包含的各个 SKU 数量。
        更新 self.target_contain_sku_num，其形状为 (num_envs, num_targets, num_skus)
        """
        # 1. 每次计算前清零。因为我们要根据当前的物理状态完全重新统计
        self.target_contain_sku_num.zero_()

        # 2. 外层循环：遍历每一种 SKU 类目
        for sku_idx, sku_name in enumerate(self.sku_names):
            
            # 获取属于当前 SKU 的所有物体实例索引
            obj_indices = self.sku_to_indices.get(sku_name, [])
            
            # 极速剪枝：如果当前任务根本没有这种 SKU 的实例，直接跳过
            if not obj_indices:
                continue
                
            # 提取这些属于该 SKU 的实例在所有环境中的状态
            # sku_obj_states 形状: (num_envs, 当前 SKU 的实例数量)
            sku_obj_states = self.object_states[:, obj_indices]

            # 3. 内层循环：遍历每一个目标订单箱
            for target_idx in range(self.num_targets):
                
                # 计算该目标箱对应的状态 ID
                # 状态映射规则: 1~num_sources 为原料箱，后面的为目标箱
                target_state_id = self.num_sources + target_idx + 1

                # 核心并行计算：找出现有实例中，状态正好等于 target_state_id 的掩码
                # is_in_target 形状: (num_envs, 当前 SKU 的实例数量)
                # 里面的值为 True (1) 代表该实例正好在这个目标箱里
                is_in_target = (sku_obj_states == target_state_id)

                # 沿着实例维度 (dim=1) 求和，统计出每个环境中，该目标箱含有多少个该 SKU 的 True
                # count_in_target 形状: (num_envs,)
                count_in_target = is_in_target.long().sum(dim=1)

                # 将统计结果直接赋值给记录张量对应的切片位置
                self.target_contain_sku_num[:, target_idx, sku_idx] = count_in_target
    
    @abstractmethod
    def _update_spawn_metrics(self,env_ids: Sequence[int] | None = None):
        raise NotImplementedError

    def _save_dynamic_metrics(self, env_ids):
        if len(env_ids) == 0:
            return
        
        def _slice_nested_dict(d:dict, target_id):
            sliced_dict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    # 如果还是字典，继续往里钻
                    sliced_dict[k] = _slice_nested_dict(v, target_id)
                elif isinstance(v, torch.Tensor):
                    # 找到 Tensor 了！直接切片，并转为普通 Python 数据给 JSON 用
                    sliced_dict[k] = v[target_id].tolist()
                else:
                    # 如果是普通的标量（如 int, float），直接照抄
                    sliced_dict[k] = v
            return sliced_dict
        
        ids_list = env_ids.tolist()

        with open(self.log_path, "a", encoding='utf-8') as f:
            for env_id in ids_list:
                row_record = _slice_nested_dict(self.eval_metrics, env_id)

                row_record["timestamp"] = time.time()
                row_record["env_id"] = env_id

                f.write(json.dumps(row_record) + "\n")

    def _init_episode_metrics_structure(self):
        """仅仅在第一次被调用时，构建嵌套字典和底层的 Tensor 容器"""
        if "overall" in self.eval_metrics:
            return

        device = self.device
        N = self.num_envs

        # ==========================
        # 1. 整体完成情况 (Overall)
        # ==========================
        self.eval_metrics["overall"] = {
            "total_orders": torch.zeros(N, dtype=torch.long, device=device),
            "completed_orders": torch.zeros(N, dtype=torch.long, device=device),
            "total_needed_items": torch.zeros(N, dtype=torch.long, device=device),
            "correct_items": torch.zeros(N, dtype=torch.long, device=device),
            "correct_ratio": torch.zeros(N, dtype=torch.float32, device=device),
            "missing_items": torch.zeros(N, dtype=torch.long, device=device),
            "missing_ratio": torch.zeros(N, dtype=torch.float32, device=device),
            "extra_items": torch.zeros(N, dtype=torch.long, device=device),
            "extra_ratio": torch.zeros(N, dtype=torch.float32, device=device),
        }

        # ==========================
        # 2. 分订单完成情况 (Per Order)
        # ==========================
        self.eval_metrics["orders"] = {}
        for t in range(self.num_targets):
            order_key = f"order_{t}"
            self.eval_metrics["orders"][order_key] = {
                "total_need": torch.zeros(N, dtype=torch.long, device=device),
                "fulfilled_need": torch.zeros(N, dtype=torch.long, device=device),
                "missing": torch.zeros(N, dtype=torch.long, device=device),
                "extra": torch.zeros(N, dtype=torch.long, device=device),
                "skus": {}
            }
            # 3. 按 SKU 区分的小字典
            for s, sku_name in enumerate(self.sku_names):
                self.eval_metrics["orders"][order_key]["skus"][sku_name] = {
                    "need": torch.zeros(N, dtype=torch.long, device=device),
                    "contain": torch.zeros(N, dtype=torch.long, device=device),
                }

    def _compute_total_metrics(self, env_ids):
        """在 reset() 拦截阶段调用，计算这局的最终得分表现"""
        self._init_episode_metrics_structure()

        # 1. 取出本次需要计算的环境的切片 (E, T, S)
        # E=len(env_ids), T=num_targets, S=num_skus
        need = self.target_need_sku_num[env_ids]
        contain = self.target_contain_sku_num[env_ids]

        order_mask = (need != -1).any(dim=-1)
        need = torch.where(need == -1, torch.zeros_like(need), need)
        contain = torch.where(need == -1, torch.zeros_like(contain), contain)
        # ==========================================
        # 核心矩阵运算 (数学逻辑极度严密)
        # ==========================================
        # 1. 正确放入的数量 = min(需要的, 实际有的)
        correct = torch.minimum(need, contain)  # (E, T, S)
        
        # 2. 漏掉的数量 = max(0, 需要的 - 实际有的)
        missing = torch.clamp(need - contain, min=0)  # (E, T, S)
        
        # 3. 多放/错放的数量 = max(0, 实际有的 - 需要的)
        extra = torch.clamp(contain - need, min=0)  # (E, T, S)

        # --- 按订单 (Target) 维度聚合 (E, T) ---
        order_need = need.sum(dim=-1)
        order_correct = correct.sum(dim=-1)
        order_missing = missing.sum(dim=-1)
        order_extra = extra.sum(dim=-1)
        
        # 定义“完美完成”的订单：没有漏掉的，也没有多余错放的
        is_order_completed = (order_missing == 0) & (order_extra == 0) & order_mask

        # --- 整体 (Overall) 维度聚合 (E,) ---
        total_need_env = order_need.sum(dim=-1)
        total_correct_env = order_correct.sum(dim=-1)
        total_missing_env = order_missing.sum(dim=-1)
        total_extra_env = order_extra.sum(dim=-1)
        completed_orders_env = is_order_completed.long().sum(dim=-1)

        # 为了计算比例，防止除以 0（如果某个环境本身就没有需求）
        denom = torch.clamp(total_need_env, min=1).float()

        # ==========================================
        # 写入指标字典
        # ==========================================
        # 整体
        ov_metrics = self.eval_metrics["overall"]
        ov_metrics["total_orders"][env_ids] = self.num_targets
        ov_metrics["completed_orders"][env_ids] = completed_orders_env
        ov_metrics["total_needed_items"][env_ids] = total_need_env
        ov_metrics["correct_items"][env_ids] = total_correct_env
        ov_metrics["correct_ratio"][env_ids] = total_correct_env.float() / denom
        ov_metrics["missing_items"][env_ids] = total_missing_env
        ov_metrics["missing_ratio"][env_ids] = total_missing_env.float() / denom
        ov_metrics["extra_items"][env_ids] = total_extra_env
        ov_metrics["extra_ratio"][env_ids] = total_extra_env.float() / denom

        # 分订单与分 SKU
        for t in range(self.num_targets):
                order_dict = self.eval_metrics["orders"][f"order_{t}"]
                order_dict["total_need"][env_ids] = order_need[:, t]
                order_dict["fulfilled_need"][env_ids] = order_correct[:, t]
                order_dict["missing"][env_ids] = order_missing[:, t]
                order_dict["extra"][env_ids] = order_extra[:, t]

                for s, sku_name in enumerate(self.sku_names):
                    sku_dict = order_dict["skus"][sku_name]
                    sku_dict["need"][env_ids] = need[:, t, s]
                    sku_dict["contain"][env_ids] = contain[:, t, s]

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
