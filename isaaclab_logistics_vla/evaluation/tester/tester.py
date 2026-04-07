"""
测试框架核心类（适配新版 BaseOrderCommandTerm）

通过瞬移物体来模拟各种测试用例，验证环境指标计算的正确性。
完全基于 target_need_sku_num + sku_to_indices 确定放置计划，
不依赖 is_target_mask / obj_to_target_id 等子类特有属性。
"""
from __future__ import annotations

import random
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Dict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_logistics_vla.utils.object_position import (
    set_asset_position,
    set_asset_relative_position,
)


@dataclass
class TestConfig:
    """测试配置"""
    name: str = "default"
    error_rate: float = 0.0
    correction_ratio: float = 0.0


TEST_SUITE = [
    TestConfig(name="all_correct", error_rate=0.0, correction_ratio=0.0),
    TestConfig(name="random_error", error_rate=0.4, correction_ratio=0.3),
]


class Tester:
    """
    测试框架类

    功能：
    1. 通过瞬移物体模拟抓取和放置过程
    2. 按 episode 轮换不同的测试配置（all_correct / random_error）
    3. 直接使用基类 eval_metrics 输出结果

    支持 ss-st, ms-st, ss-mt, ms-mt 所有场景类型。
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        command_term_name: str = "order_info",
        interval_steps: int = 50,
        test_suite: List[TestConfig] = None,
    ):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self.command_term = env.command_manager.get_term(command_term_name)
        if self.command_term is None:
            raise ValueError(f"Cannot find command term '{command_term_name}'")

        self.interval_steps = interval_steps
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.test_suite = test_suite if test_suite is not None else TEST_SUITE

        self.env_config_idx = torch.full(
            (self.num_envs,), fill_value=-1, dtype=torch.long, device=self.device,
        )

        # obj_idx -> sku_idx 反查表
        self._obj_to_sku: Dict[int, int] = {}
        for sku_idx, sku_name in enumerate(self.command_term.sku_names):
            for obj_idx in self.command_term.sku_to_indices[sku_name]:
                self._obj_to_sku[obj_idx] = sku_idx

        # 每个环境的放置计划: {obj_idx: target_box_idx}
        self._placement_plans: Dict[int, Dict[int, int]] = {}

        self._processed_mask: Optional[torch.Tensor] = None

        # 阶段追踪: 0=源箱阶段, 1=纠偏阶段, 2=完成
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._correction_budget = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._corrections_done = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 每种配置的统计数据（从基类 eval_metrics 收集）
        self.stats: Dict[str, Dict] = {
            cfg.name: {"episodes": 0, "snapshots": []}
            for cfg in self.test_suite
        }

        self._action_log: Dict[int, List[Dict]] = {}

        print(f"[Tester] 初始化完成，共 {len(self.test_suite)} 种测试配置")
        for i, cfg in enumerate(self.test_suite):
            print(f"  [{i}] {cfg.name}: error_rate={cfg.error_rate:.0%}, "
                  f"correction_ratio={cfg.correction_ratio}")

    # ================================================================
    #  Reset / Step / Completion
    # ================================================================

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        num_objects = self.command_term.num_objects
        if self._processed_mask is None:
            self._processed_mask = torch.zeros(
                (self.num_envs, num_objects), dtype=torch.bool, device=self.device,
            )

        for env_id in env_ids:
            eid = env_id.item()

            self.env_config_idx[eid] = (self.env_config_idx[eid] + 1) % len(self.test_suite)
            cfg = self.test_suite[self.env_config_idx[eid].item()]
            print(f"[Tester] Env {eid}: 切换到配置 '{cfg.name}'")

            self._placement_plans[eid] = self._build_placement_plan(eid)

            num_order_items = len(self._placement_plans[eid])
            self._correction_budget[eid] = max(1, int(cfg.correction_ratio * num_order_items))
            self._action_log[eid] = []

        self._processed_mask[env_ids] = False
        self.step_counter[env_ids] = 0
        self._phase[env_ids] = 0
        self._corrections_done[env_ids] = 0

    def step(self):
        self.step_counter += 1

        should_teleport = (self.step_counter % self.interval_steps == 0)
        teleport_env_ids = torch.where(should_teleport)[0]

        if len(teleport_env_ids) == 0:
            return

        for env_id in teleport_env_ids:
            eid = env_id.item()
            if self._phase[eid] < 2:
                self._execute_next_action(eid)

    def get_completed_envs(self) -> torch.Tensor:
        if self._processed_mask is None:
            return torch.tensor([], dtype=torch.long, device=self.device)

        completed = (self._phase == 2)
        return torch.where(completed)[0]

    # ================================================================
    #  Placement Plan
    # ================================================================

    def _build_placement_plan(self, env_id: int) -> Dict[int, int]:
        """
        根据 target_need_sku_num 和 sku_to_indices 计算每个订单物品的正确目标箱。
        适用于 ss-st, ms-st, ss-mt, ms-mt 所有场景。

        Returns:
            {obj_idx: target_box_idx}，不在结果中的活跃物品视为冗余
        """
        ct = self.command_term
        plan: Dict[int, int] = {}
        remaining_need = ct.target_need_sku_num[env_id].clone()
        remaining_need = torch.where(remaining_need == -1, torch.zeros_like(remaining_need), remaining_need)

        for sku_idx, sku_name in enumerate(ct.sku_names):
            active_instances = [
                idx for idx in ct.sku_to_indices[sku_name]
                if ct.is_active_mask[env_id, idx].item()
            ]
            for obj_idx in active_instances:
                for t in range(ct.num_targets):
                    if remaining_need[t, sku_idx] > 0:
                        plan[obj_idx] = t
                        remaining_need[t, sku_idx] -= 1
                        break
        return plan

    # ================================================================
    #  Action Execution
    # ================================================================

    def _execute_next_action(self, env_id: int):
        cfg = self.test_suite[self.env_config_idx[env_id].item()]

        if self._phase[env_id] == 0:
            obj_idx = self._sample_unprocessed_order_item(env_id)
            if obj_idx is None:
                if cfg.correction_ratio > 0:
                    self._phase[env_id] = 1
                    self._execute_next_action(env_id)
                else:
                    self._phase[env_id] = 2
                return

            if random.random() < cfg.error_rate:
                self._do_random_source_error(env_id, obj_idx)
            else:
                self._do_correct_placement(env_id, obj_idx)

            self._processed_mask[env_id, obj_idx] = True

        elif self._phase[env_id] == 1:
            if self._corrections_done[env_id] >= self._correction_budget[env_id]:
                self._phase[env_id] = 2
                return

            self._do_random_correction(env_id)
            self._corrections_done[env_id] += 1

            if self._corrections_done[env_id] >= self._correction_budget[env_id]:
                self._phase[env_id] = 2

    def _sample_unprocessed_order_item(self, env_id: int) -> Optional[int]:
        ct = self.command_term
        plan = self._placement_plans.get(env_id, {})

        source_state_min = 1
        source_state_max = ct.num_sources

        for obj_idx in plan:
            if self._processed_mask[env_id, obj_idx]:
                continue
            state = ct.object_states[env_id, obj_idx].item()
            if source_state_min <= state <= source_state_max:
                return obj_idx
        return None

    def _do_correct_placement(self, env_id: int, obj_idx: int):
        plan = self._placement_plans.get(env_id, {})
        target_idx = plan.get(obj_idx)
        if target_idx is None:
            return

        self._teleport_to_target(env_id, obj_idx, target_idx)

        obj_name = self.command_term.object_names[obj_idx]
        self._log_action(env_id, obj_name, f"s->t{target_idx} (correct)")

    def _do_random_source_error(self, env_id: int, obj_idx: int):
        ct = self.command_term
        plan = self._placement_plans.get(env_id, {})
        correct_target = plan.get(obj_idx, 0)
        obj_name = ct.object_names[obj_idx]

        error_types = ["s_to_g"]
        if ct.num_targets >= 2:
            error_types.append("s_to_t_wrong")

        error_type = random.choice(error_types)

        if error_type == "s_to_t_wrong":
            wrong_targets = [t for t in range(ct.num_targets) if t != correct_target]
            wrong_t = random.choice(wrong_targets)
            self._teleport_to_target(env_id, obj_idx, wrong_t)
            self._log_action(env_id, obj_name,
                             f"s->t{wrong_t} (wrong, correct={correct_target})")
        else:
            self._teleport_to_ground(env_id, obj_idx)
            self._log_action(env_id, obj_name, "s->g (drop)")

    def _do_random_correction(self, env_id: int):
        ct = self.command_term

        target_state_min = ct.num_sources + 1
        target_state_max = ct.num_sources + ct.num_targets

        states = ct.object_states[env_id]
        active = ct.is_active_mask[env_id]
        in_target_mask = active & (states >= target_state_min) & (states <= target_state_max)
        in_target_indices = torch.where(in_target_mask)[0]

        if len(in_target_indices) == 0:
            return

        correction_types = ["t_to_s"]
        if ct.num_targets >= 2:
            correction_types.append("t_to_t")

        correction_type = random.choice(correction_types)
        pick_idx = in_target_indices[random.randint(0, len(in_target_indices) - 1)].item()
        obj_name = ct.object_names[pick_idx]
        current_state = states[pick_idx].item()
        current_target_box = current_state - ct.num_sources - 1

        if correction_type == "t_to_t" and ct.num_targets >= 2:
            other_targets = [t for t in range(ct.num_targets) if t != current_target_box]
            new_target = random.choice(other_targets)
            self._teleport_to_target(env_id, pick_idx, new_target)
            self._log_action(env_id, obj_name,
                             f"t{current_target_box}->t{new_target}")
        else:
            source_idx = random.randint(0, ct.num_sources - 1)
            self._teleport_to_source(env_id, pick_idx, source_idx)
            self._log_action(env_id, obj_name,
                             f"t{current_target_box}->s{source_idx}")

    # ================================================================
    #  Teleport Primitives
    # ================================================================

    def _teleport_to_target(self, env_id: int, obj_idx: int, target_box_idx: int):
        env_ids = torch.tensor([env_id], device=self.device)
        obj_asset = self.command_term.object_assets[obj_idx]
        box_asset = self.command_term.target_box_assets[target_box_idx]

        rand_offset_x = (torch.rand(1).item() - 0.5) * 0.16
        rand_offset_y = (torch.rand(1).item() - 0.5) * 0.16
        drop_height = 0.25

        relative_pos = torch.tensor(
            [[rand_offset_x, rand_offset_y, drop_height]], device=self.device,
        )
        relative_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device,
        )

        set_asset_relative_position(
            env=self.env,
            env_ids=env_ids,
            target_asset=obj_asset,
            reference_asset=box_asset,
            relative_pos=relative_pos,
            relative_quat=relative_quat,
        )

    def _teleport_to_ground(self, env_id: int, obj_idx: int):
        env_ids = torch.tensor([env_id], device=self.device)
        obj_asset = self.command_term.object_assets[obj_idx]

        source_box_idx = self.command_term.obj_to_source_id[env_id, obj_idx].item()
        if source_box_idx < 0:
            source_box_idx = 0
        source_box_pos = (
            self.command_term.source_box_assets[source_box_idx].data.root_pos_w[env_id]
        )

        drop_pos_world = source_box_pos.clone()
        drop_pos_world[0] += 0.5
        drop_pos_world[2] = 0.1

        env_origin = self.env.scene.env_origins[env_id]
        drop_pos_local = drop_pos_world - env_origin

        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        set_asset_position(
            env=self.env,
            env_ids=env_ids,
            asset=obj_asset,
            position=drop_pos_local.unsqueeze(0),
            quat=quat,
        )

    def _teleport_to_source(self, env_id: int, obj_idx: int, source_box_idx: int):
        env_ids = torch.tensor([env_id], device=self.device)
        obj_asset = self.command_term.object_assets[obj_idx]
        box_asset = self.command_term.source_box_assets[source_box_idx]

        rand_offset_x = (torch.rand(1).item() - 0.5) * 0.10
        rand_offset_y = (torch.rand(1).item() - 0.5) * 0.10
        drop_height = 0.20

        relative_pos = torch.tensor(
            [[rand_offset_x, rand_offset_y, drop_height]], device=self.device,
        )
        relative_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=self.device,
        )

        set_asset_relative_position(
            env=self.env,
            env_ids=env_ids,
            target_asset=obj_asset,
            reference_asset=box_asset,
            relative_pos=relative_pos,
            relative_quat=relative_quat,
        )

    # ================================================================
    #  Logging
    # ================================================================

    def _log_action(self, env_id: int, obj_name: str, action_desc: str):
        cfg = self.test_suite[self.env_config_idx[env_id].item()]
        print(f"[Tester] Env {env_id} | {cfg.name} | {obj_name} | {action_desc}")
        if env_id not in self._action_log:
            self._action_log[env_id] = []
        self._action_log[env_id].append({"obj": obj_name, "action": action_desc})

    # ================================================================
    #  Metrics Snapshot
    # ================================================================

    def snapshot_metrics(self, env_ids: torch.Tensor):
        """
        从 command_term.eval_metrics 快照指标。
        必须在环境 reset 之后调用（reset 内部触发 _compute_total_metrics）。
        """
        if env_ids is None or len(env_ids) == 0:
            return

        check_ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids
        em = self.command_term.eval_metrics

        if "overall" not in em:
            return

        ct = self.command_term

        for eid in check_ids:
            config_idx = self.env_config_idx[eid].item()
            config_name = self.test_suite[config_idx].name

            snapshot = {
                "env_id": eid,
                "correct_ratio": em["overall"]["correct_ratio"][eid].item(),
                "missing_ratio": em["overall"]["missing_ratio"][eid].item(),
                "extra_ratio": em["overall"]["extra_ratio"][eid].item(),
                "completed_orders": em["overall"]["completed_orders"][eid].item(),
                "total_orders": em["overall"]["total_orders"][eid].item(),
                "total_needed_items": em["overall"]["total_needed_items"][eid].item(),
                "correct_items": em["overall"]["correct_items"][eid].item(),
                "missing_items": em["overall"]["missing_items"][eid].item(),
                "extra_items": em["overall"]["extra_items"][eid].item(),
            }

            if "episode_physics_steps" in em:
                snapshot["steps"] = em["episode_physics_steps"][eid].item()

            if "orders" in em:
                snapshot["orders"] = {}
                for t in range(ct.num_targets):
                    order_key = f"order_{t}"
                    if order_key not in em["orders"]:
                        continue
                    od = em["orders"][order_key]
                    order_snap = {
                        "total_need": od["total_need"][eid].item(),
                        "fulfilled_need": od["fulfilled_need"][eid].item(),
                        "missing": od["missing"][eid].item(),
                        "extra": od["extra"][eid].item(),
                        "skus": {},
                    }
                    if "skus" in od:
                        for sku_name, sku_d in od["skus"].items():
                            order_snap["skus"][sku_name] = {
                                "need": sku_d["need"][eid].item(),
                                "contain": sku_d["contain"][eid].item(),
                            }
                    snapshot["orders"][order_key] = order_snap

            snapshot["actions"] = self._action_log.get(eid, [])

            self.stats[config_name]["episodes"] += 1
            self.stats[config_name]["snapshots"].append(snapshot)

            print(
                f"[Tester] Episode 结束 | Env {eid} | Config: {config_name} "
                f"| correct={snapshot['correct_ratio']:.1%} "
                f"| missing={snapshot['missing_ratio']:.1%} "
                f"| extra={snapshot['extra_ratio']:.1%} "
                f"| orders={snapshot['completed_orders']}/{snapshot['total_orders']}"
            )

    # ================================================================
    #  Summary Output
    # ================================================================

    def print_summary(self):
        print("\n" + "=" * 80)
        print("                         测试报告总结")
        print("=" * 80)

        total_episodes = 0
        ct = self.command_term

        for cfg_name, stat in self.stats.items():
            snapshots = stat["snapshots"]
            n = stat["episodes"]
            if n == 0:
                continue
            total_episodes += n

            avg_correct = sum(s["correct_ratio"] for s in snapshots) / n
            avg_missing = sum(s["missing_ratio"] for s in snapshots) / n
            avg_extra = sum(s["extra_ratio"] for s in snapshots) / n
            avg_correct_items = sum(s["correct_items"] for s in snapshots) / n
            avg_needed = sum(s["total_needed_items"] for s in snapshots) / n
            avg_missing_items = sum(s["missing_items"] for s in snapshots) / n
            avg_extra_items = sum(s["extra_items"] for s in snapshots) / n
            total_completed = sum(s["completed_orders"] for s in snapshots)
            total_orders = sum(s["total_orders"] for s in snapshots)
            order_completion = total_completed / max(total_orders, 1)

            print(f"\n┌─ 配置: {cfg_name} ({n} episodes)")
            print(f"│")
            print(f"│  订单完成率:     {order_completion:>7.1%}  "
                  f"({total_completed}/{total_orders} 订单完美完成)")
            print(f"│  物品正确率:     {avg_correct:>7.1%}  "
                  f"(平均 {avg_correct_items:.1f}/{avg_needed:.1f} 个正确)")
            print(f"│  物品缺失率:     {avg_missing:>7.1%}  "
                  f"(平均 {avg_missing_items:.1f} 个缺失)")
            print(f"│  多余/错放率:    {avg_extra:>7.1%}  "
                  f"(平均 {avg_extra_items:.1f} 个多余)")
            print(f"│")

            # 分订单/分 SKU 明细
            if snapshots and "orders" in snapshots[0]:
                print(f"│  ┌─ 分订单明细 (平均) " + "─" * 30 + "┐")
                for t in range(ct.num_targets):
                    order_key = f"order_{t}"
                    order_snaps = [
                        s["orders"][order_key]
                        for s in snapshots
                        if order_key in s.get("orders", {})
                    ]
                    if not order_snaps:
                        continue

                    avg_need = sum(o["total_need"] for o in order_snaps) / len(order_snaps)
                    avg_ful = sum(o["fulfilled_need"] for o in order_snaps) / len(order_snaps)
                    avg_mis = sum(o["missing"] for o in order_snaps) / len(order_snaps)
                    avg_ext = sum(o["extra"] for o in order_snaps) / len(order_snaps)

                    if avg_need == 0:
                        continue

                    print(f"│  │ Order {t}: "
                          f"需要 {avg_need:.1f}, 正确 {avg_ful:.1f}, "
                          f"缺失 {avg_mis:.1f}, 多余 {avg_ext:.1f}")

                    for sku_name in ct.sku_names:
                        sku_needs = [
                            o["skus"][sku_name]["need"]
                            for o in order_snaps
                            if sku_name in o.get("skus", {})
                        ]
                        sku_contains = [
                            o["skus"][sku_name]["contain"]
                            for o in order_snaps
                            if sku_name in o.get("skus", {})
                        ]
                        if sku_needs and sum(sku_needs) > 0:
                            avg_sn = sum(sku_needs) / len(sku_needs)
                            avg_sc = sum(sku_contains) / len(sku_contains)
                            print(f"│  │   {sku_name}: "
                                  f"需要 {avg_sn:.1f}, 实际 {avg_sc:.1f}")

                print(f"│  └" + "─" * 50 + "┘")

            print(f"└" + "─" * 60)

        print(f"\n总计: {total_episodes} 个 episodes")
        print("=" * 80)

        # 汇总表
        print(f"\n┌" + "─" * 84 + "┐")
        print(f"│ {'配置名称':<16} │ {'Episodes':>8} │ "
              f"{'正确率':>8} │ {'缺失率':>8} │ "
              f"{'多余率':>8} │ {'订单完成':>8} │")
        print(f"├" + "─" * 84 + "┤")

        for cfg_name, stat in self.stats.items():
            n = stat["episodes"]
            if n == 0:
                continue
            snapshots = stat["snapshots"]
            avg_c = sum(s["correct_ratio"] for s in snapshots) / n
            avg_m = sum(s["missing_ratio"] for s in snapshots) / n
            avg_e = sum(s["extra_ratio"] for s in snapshots) / n
            tc = sum(s["completed_orders"] for s in snapshots)
            to = sum(s["total_orders"] for s in snapshots)
            oc = tc / max(to, 1)
            print(f"│ {cfg_name:<16} │ {n:>8} │ "
                  f"{avg_c:>7.1%} │ {avg_m:>7.1%} │ "
                  f"{avg_e:>7.1%} │ {oc:>7.1%} │")

        print(f"└" + "─" * 84 + "┘")
