"""
测试框架核心类
通过瞬移物体来模拟各种测试用例，验证环境指标计算的正确性
"""
from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_logistics_vla.utils.object_position import (
    set_asset_position,
    set_asset_relative_position,
    check_object_in_box
)


@dataclass
class TestConfig:
    """单个环境的测试配置"""
    name: str = "default"
    drop_rate: float = 0.0        # 掉落概率（失败率）
    wrong_pick_rate: float = 0.0  # 错抓概率（抓干扰物）
    wrong_place_rate: float = 0.0 # 错放概率（放错箱子）


# 预定义测试套件
TEST_SUITE = [
    TestConfig(name="all_correct",    drop_rate=0.0, wrong_pick_rate=0.0, wrong_place_rate=0.0),
    TestConfig(name="drop_20%",       drop_rate=0.2, wrong_pick_rate=0.0, wrong_place_rate=0.0),
    TestConfig(name="wrong_pick_30%", drop_rate=0.0, wrong_pick_rate=0.3, wrong_place_rate=0.0),
    TestConfig(name="wrong_place_20%",drop_rate=0.0, wrong_pick_rate=0.0, wrong_place_rate=0.2),
    TestConfig(name="mixed",          drop_rate=0.1, wrong_pick_rate=0.1, wrong_place_rate=0.1),
]


class Tester:
    """
    测试框架类
    
    功能：
    1. 通过瞬移物体模拟抓取和放置过程
    2. 按episode轮换不同的测试配置
    3. 维护"应然"状态，与环境的"实然"指标对比验证
    """
    
    def __init__(
        self, 
        env: ManagerBasedRLEnv, 
        command_term_name: str = "order_info",
        interval_steps: int = 50,
        test_suite: List[TestConfig] = None
    ):
        """
        初始化测试器
        
        Args:
            env: Isaac Lab环境实例
            command_term_name: CommandTerm的名称
            interval_steps: 每隔多少步进行一次瞬移
            test_suite: 测试配置套件，默认使用预定义的TEST_SUITE
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        
        # 获取CommandTerm引用
        self.command_term = env.command_manager.get_term(command_term_name)
        if self.command_term is None:
            raise ValueError(f"Cannot find command term '{command_term_name}'")
        
        self.interval_steps = interval_steps
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 测试配置
        self.test_suite = test_suite if test_suite is not None else TEST_SUITE
        
        # 每个环境当前使用的配置索引
        # 这里初始化为 -1，这样在第一次 reset() 时自增到 0，
        # 确保第一个 episode 使用 TEST_SUITE[0]（例如 "all_correct"）
        self.env_config_idx = torch.full(
            (self.num_envs,),
            fill_value=-1,
            dtype=torch.long,
            device=self.device,
        )
        
        # 每个环境当前处理到的物体索引
        self.current_obj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 追踪每个物体是否已被处理（避免重复采样）
        # 注意：需要在知道num_objects后初始化，这里先设为None
        self._processed_mask = None
        
        # ============ "应然"状态 ============
        # 我们主动控制瞬移，所以知道预期结果
        self.expected_success_count = torch.zeros(self.num_envs, device=self.device)
        self.expected_failure_count = torch.zeros(self.num_envs, device=self.device)
        self.expected_wrong_pick_count = torch.zeros(self.num_envs, device=self.device)
        self.expected_wrong_place_count = torch.zeros(self.num_envs, device=self.device)
        
        # 记录每种配置的统计数据
        self.stats = {
            cfg.name: {
                "episodes": 0,
                "completion_rates": [],      # 每个episode的完成率
                "failure_rates": [],         # 每个episode的失败率
                "wrong_pick_rates": [],      # 每个episode的错抓率
                "wrong_place_rates": [],     # 每个episode的错放率
                "passed": 0,                 # 指标匹配的次数
                "failed": 0,                 # 指标不匹配的次数
                # 详细统计数据（每个episode的记录）
                "env_details": [],           # 每个episode的详细统计 [{env_id, processed, wrong_pick, wrong_place, dropped, ...}, ...]
            } 
            for cfg in self.test_suite
        }
        
        print(f"[Tester] 初始化完成，共 {len(self.test_suite)} 种测试配置")
        for i, cfg in enumerate(self.test_suite):
            print(f"  [{i}] {cfg.name}: drop={cfg.drop_rate:.0%}, wrong_pick={cfg.wrong_pick_rate:.0%}, wrong_place={cfg.wrong_place_rate:.0%}")
    
    def reset(self, env_ids: torch.Tensor):
        """
        环境重置时调用
        
        Args:
            env_ids: 需要重置的环境ID
        """
        if len(env_ids) == 0:
            return
            
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        for env_id in env_ids:
            env_id_int = env_id.item()
            
            # 轮换到下一个配置
            self.env_config_idx[env_id] = (self.env_config_idx[env_id] + 1) % len(self.test_suite)
            new_cfg = self.test_suite[self.env_config_idx[env_id].item()]
            
            print(f"[Tester] Env {env_id_int}: 切换到配置 '{new_cfg.name}'")
        
        # 重置"应然"状态
        self.expected_success_count[env_ids] = 0
        self.expected_failure_count[env_ids] = 0
        self.expected_wrong_pick_count[env_ids] = 0
        self.expected_wrong_place_count[env_ids] = 0
        
        # 重置物体索引和步数计数器
        self.current_obj_idx[env_ids] = 0
        self.step_counter[env_ids] = 0
        
        # 初始化/重置已处理物体追踪
        num_objects = len(self.command_term.object_names)
        if self._processed_mask is None:
            self._processed_mask = torch.zeros(
                (self.num_envs, num_objects), dtype=torch.bool, device=self.device
            )
        self._processed_mask[env_ids] = False
    
    def step(self):
        """
        每帧调用，检查是否到瞬移时机
        """
        self.step_counter += 1
        
        # 检查哪些环境到了瞬移时机
        should_teleport = (self.step_counter % self.interval_steps == 0)
        teleport_env_ids = torch.where(should_teleport)[0]
        
        if len(teleport_env_ids) == 0:
            return
        
        # 对每个需要瞬移的环境执行瞬移
        for env_id in teleport_env_ids:
            self._teleport_next_object(env_id.item())
    
    def get_completed_envs(self) -> torch.Tensor:
        """
        检查哪些环境的所有目标物都已处理完毕
        
        "处理完毕"定义：所有目标物的状态都不是1（待处理）
        即：已成功(3) 或 已失败(4) 或 在途中(2)
        
        Returns:
            需要重置的环境ID列表
        """
        if self._processed_mask is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        completed_envs = []
        
        for env_id in range(self.num_envs):
            # 获取该环境的目标物掩码
            target_mask = self.command_term.is_target_mask[env_id]
            
            # 检查是否有目标物
            if target_mask.sum() == 0:
                continue
            
            # 检查所有目标物是否都已被 Tester 处理
            # （使用 _processed_mask 而不是环境状态，因为环境状态可能有延迟）
            targets_processed = self._processed_mask[env_id] | (~target_mask)
            all_targets_processed = targets_processed.all().item()
            
            if all_targets_processed:
                completed_envs.append(env_id)
        
        return torch.tensor(completed_envs, dtype=torch.long, device=self.device)
    
    def _teleport_next_object(self, env_id: int):
        """
        为指定环境瞬移下一个待处理的物体
        
        设计逻辑：
        1. 先以 wrong_pick_rate 概率决定是否错抓干扰物
        2. 如果不错抓，就处理一个目标物
        3. 目标物处理时，根据 drop_rate 和 wrong_place_rate 决定是正确还是出错
        
        Args:
            env_id: 环境ID
        """
        cfg = self.test_suite[self.env_config_idx[env_id].item()]
        
        # ========== 第一步：决定是否错抓干扰物 ==========
        if torch.rand(1).item() < cfg.wrong_pick_rate:
            distractor_idx = self._sample_distractor(env_id)
            if distractor_idx is not None:
                obj_name = self.command_term.object_names[distractor_idx]
                self._teleport_distractor_out(env_id, distractor_idx)
                self._processed_mask[env_id, distractor_idx] = True
                self.expected_wrong_pick_count[env_id] += 1
                print(f"[Tester] Env {env_id} | Config: {cfg.name} | Object: {obj_name} | Action: wrong_pick")
                return
        
        # ========== 第二步：处理目标物 ==========
        target_idx = self._sample_target(env_id)
        if target_idx is None:
            # 没有待处理的目标物了
            return
        
        obj_name = self.command_term.object_names[target_idx]
        
        # 决定目标物的处理方式
        rand = torch.rand(1).item()
        
        if rand < cfg.drop_rate:
            # 掉落失败
            self._teleport_to_ground(env_id, target_idx)
            action_type = "drop"
            self.expected_failure_count[env_id] += 1
        elif rand < cfg.drop_rate + cfg.wrong_place_rate:
            # 放错箱子
            self._teleport_to_wrong_target(env_id, target_idx)
            action_type = "wrong_place"
            self.expected_wrong_place_count[env_id] += 1
            self.expected_failure_count[env_id] += 1  # 放错也算失败
        else:
            # 正确放置
            self._teleport_to_correct_target(env_id, target_idx)
            action_type = "correct"
            self.expected_success_count[env_id] += 1
        
        # 标记为已处理
        self._processed_mask[env_id, target_idx] = True
        print(f"[Tester] Env {env_id} | Config: {cfg.name} | Object: {obj_name} | Action: {action_type}")
    
    def _sample_target(self, env_id: int) -> Optional[int]:
        """
        采样下一个待处理的目标物
        
        条件：
        1. 是目标物（is_target_mask）
        2. 状态为1（待处理，还在原料箱）
        3. 未被处理过（_processed_mask）
        
        Returns:
            物体索引，如果没有则返回None
        """
        target_mask = self.command_term.is_target_mask[env_id]
        pending_mask = (self.command_term.object_states[env_id] == 1)
        not_processed = ~self._processed_mask[env_id]
        
        available_mask = target_mask & pending_mask & not_processed
        available_indices = torch.where(available_mask)[0]
        
        if len(available_indices) == 0:
            return None
        
        # 按顺序返回第一个
        return available_indices[0].item()
    
    def _sample_distractor(self, env_id: int) -> Optional[int]:
        """
        采样一个待处理的干扰物（用于错抓）
        
        条件：
        1. 是活跃物但不是目标物（干扰物）
        2. 状态为1（还在原料箱）
        3. 未被处理过
        
        Returns:
            物体索引，如果没有则返回None
        """
        active_mask = self.command_term.is_active_mask[env_id]
        target_mask = self.command_term.is_target_mask[env_id]
        distractor_mask = active_mask & (~target_mask)  # 干扰物 = 活跃但不是目标
        
        pending_mask = (self.command_term.object_states[env_id] == 1)
        not_processed = ~self._processed_mask[env_id]
        
        available_mask = distractor_mask & pending_mask & not_processed
        available_indices = torch.where(available_mask)[0]
        
        if len(available_indices) == 0:
            return None
        
        # 随机选一个干扰物
        rand_idx = torch.randint(0, len(available_indices), (1,)).item()
        return available_indices[rand_idx].item()
    
    def _teleport_to_correct_target(self, env_id: int, obj_idx: int):
        """
        将物体瞬移到正确的目标箱
        
        策略：从箱子上方较高位置释放，让物体自然掉落
        这样可以避免与已有物体重叠导致的物理碰撞"炸飞"问题
        """
        env_ids = torch.tensor([env_id], device=self.device)
        obj_asset = self.command_term.object_assets[obj_idx]
        
        # 获取正确的目标箱
        target_box_idx = self.command_term.obj_to_target_id[env_id, obj_idx].item()
        target_box_asset = self.command_term.target_box_assets[target_box_idx]
        
        # 从箱子上方释放，让物体自然掉落
        # z=0.25 足够高，物体会自然掉入箱子并稳定下来
        # xy随机偏移范围增大到 -0.08 ~ 0.08，减少与已有物体碰撞的概率
        rand_offset_x = (torch.rand(1).item() - 0.5) * 0.16  # -0.08 到 0.08
        rand_offset_y = (torch.rand(1).item() - 0.5) * 0.16
        drop_height = 0.25  # 从箱子中心上方0.25m处释放
        
        relative_pos = torch.tensor([[rand_offset_x, rand_offset_y, drop_height]], device=self.device)
        relative_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        
        set_asset_relative_position(
            env=self.env,
            env_ids=env_ids,
            target_asset=obj_asset,
            reference_asset=target_box_asset,
            relative_pos=relative_pos,
            relative_quat=relative_quat
        )
        
        # 调试输出：显示目标箱位置和相对偏移
        box_pos = target_box_asset.data.root_pos_w[env_id]
        print(f"    [DEBUG] 目标箱位置: {box_pos.tolist()}, 释放高度: z={drop_height}, xy偏移: ({rand_offset_x:.3f}, {rand_offset_y:.3f})")
    
    def _teleport_to_ground(self, env_id: int, obj_idx: int):
        """
        将物体瞬移到地面（模拟掉落失败）
        
        策略：获取原料箱的世界位置，在其旁边的地面放置物体
        z < 0.3 会被环境判定为失败
        """
        env_ids = torch.tensor([env_id], device=self.device)
        obj_asset = self.command_term.object_assets[obj_idx]
        
        # 获取物体所在的原料箱的世界位置
        source_box_idx = self.command_term.obj_to_source_id[env_id, obj_idx].item()
        source_box_asset = self.command_term.source_box_assets[source_box_idx]
        source_box_pos = source_box_asset.data.root_pos_w[env_id]  # 世界坐标
        
        # 计算掉落位置：在原料箱x方向偏移0.5m，贴近地面z=0.1
        drop_pos_world = source_box_pos.clone()
        drop_pos_world[0] += 0.5  # x偏移
        drop_pos_world[2] = 0.1   # 贴近地面（z<0.3会被判定为失败）
        
        # 转换为局部坐标（减去env_origin）
        env_origin = self.env.scene.env_origins[env_id]
        drop_pos_local = drop_pos_world - env_origin
        
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        
        set_asset_position(
            env=self.env,
            env_ids=env_ids,
            asset=obj_asset,
            position=drop_pos_local.unsqueeze(0),
            quat=quat
        )
        
        # 调试输出
        print(f"    [DEBUG] 掉落：目标位置(局部)={drop_pos_local.tolist()}, z={drop_pos_local[2].item():.3f}")
    
    def _teleport_to_wrong_target(self, env_id: int, obj_idx: int):
        """
        将物体瞬移到错误的目标箱
        
        策略：同样从高处释放，避免碰撞问题
        """
        env_ids = torch.tensor([env_id], device=self.device)
        obj_asset = self.command_term.object_assets[obj_idx]
        
        # 获取正确的目标箱索引
        correct_target_idx = self.command_term.obj_to_target_id[env_id, obj_idx].item()
        
        # 选择一个错误的目标箱
        num_targets = self.command_term.num_targets
        wrong_target_idx = (correct_target_idx + 1) % num_targets
        
        wrong_box_asset = self.command_term.target_box_assets[wrong_target_idx]
        
        # 从箱子上方释放
        rand_offset_x = (torch.rand(1).item() - 0.5) * 0.16
        rand_offset_y = (torch.rand(1).item() - 0.5) * 0.16
        drop_height = 0.25
        
        relative_pos = torch.tensor([[rand_offset_x, rand_offset_y, drop_height]], device=self.device)
        relative_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        
        set_asset_relative_position(
            env=self.env,
            env_ids=env_ids,
            target_asset=obj_asset,
            reference_asset=wrong_box_asset,
            relative_pos=relative_pos,
            relative_quat=relative_quat
        )
        
        # 调试输出
        print(f"    [DEBUG] 错放：正确箱={correct_target_idx}, 放入箱={wrong_target_idx}, 释放高度={drop_height}")
    
    def _teleport_distractor_out(self, env_id: int, obj_idx: int):
        """
        将干扰物移出原料箱（模拟错抓）
        
        策略：移到原料箱和目标箱之间的过渡区域（不进入任何箱子）
        """
        env_ids = torch.tensor([env_id], device=self.device)
        obj_asset = self.command_term.object_assets[obj_idx]
        
        # 获取原料箱位置作为参考
        source_box_idx = self.command_term.obj_to_source_id[env_id, obj_idx].item()
        source_box_asset = self.command_term.source_box_assets[source_box_idx]
        source_box_pos = source_box_asset.data.root_pos_w[env_id]
        
        # 在原料箱上方/旁边，但不进入任何箱子
        transit_pos_world = source_box_pos.clone()
        transit_pos_world[0] -= 0.3  # x方向偏移
        transit_pos_world[2] += 0.3  # 抬高一点
        
        # 转换为局部坐标
        env_origin = self.env.scene.env_origins[env_id]
        transit_pos_local = transit_pos_world - env_origin
        
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        
        set_asset_position(
            env=self.env,
            env_ids=env_ids,
            asset=obj_asset,
            position=transit_pos_local.unsqueeze(0),
            quat=quat
        )
    
    def check(self, env_ids: torch.Tensor = None) -> bool:
        """
        验证实然指标与应然指标是否一致
        
        Args:
            env_ids: 要检查的环境ID列表，如果为None则检查所有环境
        
        Returns:
            所有检查是否通过
        """
        all_passed = True
        
        if env_ids is None:
            check_env_ids = range(self.num_envs)
        else:
            check_env_ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids
        
        for env_id in check_env_ids:
            cfg = self.test_suite[self.env_config_idx[env_id].item()]
            
            # 获取实然指标
            actual_completion = self.command_term.metrics.get("order_completion_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
            actual_failure = self.command_term.metrics.get("failure_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
            actual_wrong_pick = self.command_term.metrics.get("wrong_pick_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
            actual_wrong_place = self.command_term.metrics.get("wrong_place_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
            
            # 获取环境中实际的统计数据
            num_targets = self.command_term.is_target_mask[env_id].sum().item()
            num_active = self.command_term.is_active_mask[env_id].sum().item()
            num_distractors = num_active - num_targets
            
            # 我们期望的统计数据（基于我们的瞬移操作）
            exp_success = self.expected_success_count[env_id].item()
            exp_failure = self.expected_failure_count[env_id].item()
            exp_wrong_pick = self.expected_wrong_pick_count[env_id].item()
            exp_wrong_place = self.expected_wrong_place_count[env_id].item()
            
            # 计算期望的比率（确保不超过合理范围）
            expected_completion = min(exp_success / max(num_targets, 1), 1.0)
            expected_failure = min(exp_failure / max(num_active, 1), 1.0)
            expected_wrong_pick = min(exp_wrong_pick / max(num_distractors, 1), 1.0) if num_distractors > 0 else 0.0
            
            # 打印详细的调试信息
            print(f"\n[Tester] === Env {env_id} 检查详情 ===")
            print(f"  配置: {cfg.name}")
            print(f"  目标物数量: {num_targets}, 活跃物数量: {num_active}, 干扰物数量: {num_distractors}")
            print(f"  期望统计: success={exp_success:.0f}, failure={exp_failure:.0f}, wrong_pick={exp_wrong_pick:.0f}, wrong_place={exp_wrong_place:.0f}")
            print(f"  实然指标: completion={actual_completion:.3f}, failure={actual_failure:.3f}, wrong_pick={actual_wrong_pick:.3f}")
            print(f"  应然指标: completion={expected_completion:.3f}, failure={expected_failure:.3f}, wrong_pick={expected_wrong_pick:.3f}")
            
            # ===== 调试：打印每个物体的实际位置和状态 =====
            print(f"  --- 物体状态详情 ---")
            object_states = self.command_term.object_states[env_id]
            for obj_idx, obj_name in enumerate(self.command_term.object_names):
                is_target = self.command_term.is_target_mask[env_id, obj_idx].item()
                is_active = self.command_term.is_active_mask[env_id, obj_idx].item()
                state = object_states[obj_idx].item()
                
                if is_active:
                    obj_pos = self.command_term.object_assets[obj_idx].data.root_pos_w[env_id]
                    state_names = {0: "待生成", 1: "待处理", 2: "在途中", 3: "已完成", 4: "已失败"}
                    role = "目标物" if is_target else "干扰物"
                    print(f"    {obj_name}: {role}, 状态={state_names.get(state, state)}, 位置={obj_pos[:3].tolist()}")
            
            # 检查一致性（允许小误差）
            tolerance = 0.05  # 放宽容差
            
            checks = [
                ("completion_rate", actual_completion, expected_completion),
                ("failure_rate", actual_failure, expected_failure),
            ]
            
            # 只有当有干扰物时才检查错抓率
            if num_distractors > 0:
                checks.append(("wrong_pick_rate", actual_wrong_pick, expected_wrong_pick))
            
            env_passed = True
            for name, actual, expected in checks:
                if abs(actual - expected) > tolerance:
                    print(f"  ❌ {name}: 不匹配! actual={actual:.3f}, expected={expected:.3f}")
                    env_passed = False
                else:
                    print(f"  ✓ {name}: 匹配")
            
            if env_passed:
                print(f"[Tester] ✓ Env {env_id} 所有指标匹配!")
            else:
                all_passed = False
                print(f"[Tester] ❌ Env {env_id} 存在不匹配的指标")
        
        return all_passed
    
    def record_stats(self, env_ids: torch.Tensor):
        """
        记录指定环境的统计数据（必须在环境重置前调用）
        
        Args:
            env_ids: 需要记录的环境ID列表
        """
        if env_ids is None or len(env_ids) == 0:
            return
            
        check_env_ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids
        
        for env_id in check_env_ids:
            config_idx = self.env_config_idx[env_id].item()
            config_name = self.test_suite[config_idx].name
            self._record_episode_stats(env_id, config_name)
    
    def _record_episode_stats(self, env_id: int, config_name: str):
        """
        记录一个episode的统计数据
        """
        # 获取实然指标
        actual_completion = self.command_term.metrics.get("order_completion_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
        actual_failure = self.command_term.metrics.get("failure_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
        actual_wrong_pick = self.command_term.metrics.get("wrong_pick_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
        actual_wrong_place = self.command_term.metrics.get("wrong_place_rate", torch.zeros(self.num_envs, device=self.device))[env_id].item()
        
        # ========== 计算详细的物体数量统计 ==========
        object_states = self.command_term.object_states[env_id]
        is_target_mask = self.command_term.is_target_mask[env_id]
        is_active_mask = self.command_term.is_active_mask[env_id]
        distractor_mask = is_active_mask & (~is_target_mask)
        
        # 1. 处理个数（成功完成的目标物数量）
        num_processed = ((object_states == 3) & is_target_mask).sum().item()
        
        # 2. 错抓个数（干扰物被移动的数量：干扰物且状态!=1）
        num_wrong_pick = (distractor_mask & (object_states != 1)).sum().item()
        
        # 3. 错放个数（目标物放到错误箱子的数量）
        # 使用 CommandTerm 的方法计算错放数量
        wrong_place_count_tensor = self.command_term._count_wrong_placements()
        num_wrong_place = wrong_place_count_tensor[env_id].item()
        
        # 4. 掉落个数（状态为4的物体数量）
        num_dropped = ((object_states == 4) & is_active_mask).sum().item()
        
        # 5. 其他统计
        num_targets = is_target_mask.sum().item()
        num_distractors = distractor_mask.sum().item()
        num_total_active = is_active_mask.sum().item()
        
        # 记录到统计数据
        self.stats[config_name]["episodes"] += 1
        self.stats[config_name]["completion_rates"].append(actual_completion)
        self.stats[config_name]["failure_rates"].append(actual_failure)
        self.stats[config_name]["wrong_pick_rates"].append(actual_wrong_pick)
        self.stats[config_name]["wrong_place_rates"].append(actual_wrong_place)
        
        # 记录详细统计
        detail = {
            "env_id": env_id,
            "processed": num_processed,           # 成功处理个数
            "wrong_pick": num_wrong_pick,         # 错抓个数
            "wrong_place": num_wrong_place,       # 错放个数
            "dropped": num_dropped,               # 掉落个数
            "num_targets": num_targets,           # 目标物总数
            "num_distractors": num_distractors,   # 干扰物总数
            "num_total_active": num_total_active, # 活跃物总数
        }
        self.stats[config_name]["env_details"].append(detail)
        
        print(f"[Tester] Episode结束 | Env {env_id} | Config: {config_name}")
        print(f"  Expected: success={self.expected_success_count[env_id].item():.0f}, "
              f"failure={self.expected_failure_count[env_id].item():.0f}, "
              f"wrong_pick={self.expected_wrong_pick_count[env_id].item():.0f}, "
              f"wrong_place={self.expected_wrong_place_count[env_id].item():.0f}")
        print(f"  Actual rates: completion={actual_completion:.1%}, "
              f"failure={actual_failure:.1%}, "
              f"wrong_pick={actual_wrong_pick:.1%}, "
              f"wrong_place={actual_wrong_place:.1%}")
        print(f"  详细统计: 处理={num_processed}, 错抓={num_wrong_pick}, 错放={num_wrong_place}, 掉落={num_dropped} "
              f"(目标物={num_targets}, 干扰物={num_distractors})")
    
    def print_summary(self):
        """
        打印测试总结报告
        """
        print("\n" + "=" * 80)
        print("                         测试报告总结")
        print("=" * 80)
        
        total_episodes = 0
        
        for cfg_name, stat in self.stats.items():
            episodes = stat["episodes"]
            if episodes > 0:
                total_episodes += episodes
                
                # 计算平均值
                avg_completion = sum(stat["completion_rates"]) / episodes
                avg_failure = sum(stat["failure_rates"]) / episodes
                avg_wrong_pick = sum(stat["wrong_pick_rates"]) / episodes if stat["wrong_pick_rates"] else 0
                avg_wrong_place = sum(stat["wrong_place_rates"]) / episodes if stat["wrong_place_rates"] else 0
                
                # 计算详细统计的平均值
                details = stat["env_details"]
                if details:
                    avg_processed = sum(d["processed"] for d in details) / len(details)
                    avg_wrong_pick_count = sum(d["wrong_pick"] for d in details) / len(details)
                    avg_wrong_place_count = sum(d["wrong_place"] for d in details) / len(details)
                    avg_dropped = sum(d["dropped"] for d in details) / len(details)
                else:
                    avg_processed = avg_wrong_pick_count = avg_wrong_place_count = avg_dropped = 0
                
                print(f"\n┌─ 配置: {cfg_name} ({episodes} episodes)")
                print(f"│")
                print(f"│  平均完成率:   {avg_completion:>7.1%}")
                print(f"│  平均失败率:   {avg_failure:>7.1%}")
                print(f"│  平均错抓率:   {avg_wrong_pick:>7.1%}")
                print(f"│  平均错放率:   {avg_wrong_place:>7.1%}")
                print(f"│")
                print(f"│  平均处理个数: {avg_processed:>7.1f}")
                print(f"│  平均错抓个数: {avg_wrong_pick_count:>7.1f}")
                print(f"│  平均错放个数: {avg_wrong_place_count:>7.1f}")
                print(f"│  平均掉落个数: {avg_dropped:>7.1f}")
                print(f"│")
                
                # 每个环境的详细统计
                if details:
                    print(f"│  ┌─ 各环境详细统计 ─┐")
                    print(f"│  │ {'环境':<4} │ {'处理':>6} │ {'错抓':>6} │ {'错放':>6} │ {'掉落':>6} │ {'目标物':>6} │ {'干扰物':>6} │")
                    print(f"│  ├" + "─" * 50 + "┤")
                    for detail in details:
                        print(f"│  │ Env{detail['env_id']:<2} │ {detail['processed']:>6} │ "
                              f"{detail['wrong_pick']:>6} │ {detail['wrong_place']:>6} │ "
                              f"{detail['dropped']:>6} │ {detail['num_targets']:>6} │ "
                              f"{detail['num_distractors']:>6} │")
                    print(f"│  └" + "─" * 50 + "┘")
                
                print(f"└" + "─" * 60)
        
        print(f"\n总计: {total_episodes} 个 episodes")
        print("=" * 80)
        
        # 输出汇总表格（按配置）
        print("\n┌" + "─" * 78 + "┐")
        print(f"│ {'配置名称':<20} │ {'Episodes':>8} │ {'完成率':>10} │ {'失败率':>10} │ {'错抓率':>10} │ {'错放率':>10} │")
        print("├" + "─" * 78 + "┤")
        
        for cfg_name, stat in self.stats.items():
            episodes = stat["episodes"]
            if episodes > 0:
                avg_comp = sum(stat["completion_rates"]) / episodes
                avg_fail = sum(stat["failure_rates"]) / episodes
                avg_pick = sum(stat["wrong_pick_rates"]) / episodes if stat["wrong_pick_rates"] else 0
                avg_place = sum(stat["wrong_place_rates"]) / episodes if stat["wrong_place_rates"] else 0
                print(f"│ {cfg_name:<20} │ {episodes:>8} │ {avg_comp:>9.1%} │ {avg_fail:>9.1%} │ "
                      f"{avg_pick:>9.1%} │ {avg_place:>9.1%} │")
        
        print("└" + "─" * 78 + "┘")
        
        # 输出详细数量汇总表格
        print("\n┌" + "─" * 78 + "┐")
        print(f"│ {'配置名称':<20} │ {'Episodes':>8} │ {'处理个数':>10} │ {'错抓个数':>10} │ {'错放个数':>10} │ {'掉落个数':>10} │")
        print("├" + "─" * 78 + "┤")
        
        for cfg_name, stat in self.stats.items():
            episodes = stat["episodes"]
            if episodes > 0 and stat["env_details"]:
                details = stat["env_details"]
                avg_processed = sum(d["processed"] for d in details) / len(details)
                avg_wrong_pick_count = sum(d["wrong_pick"] for d in details) / len(details)
                avg_wrong_place_count = sum(d["wrong_place"] for d in details) / len(details)
                avg_dropped = sum(d["dropped"] for d in details) / len(details)
                print(f"│ {cfg_name:<20} │ {episodes:>8} │ {avg_processed:>10.1f} │ "
                      f"{avg_wrong_pick_count:>10.1f} │ {avg_wrong_place_count:>10.1f} │ "
                      f"{avg_dropped:>10.1f} │")
        
        print("└" + "─" * 78 + "┘")
