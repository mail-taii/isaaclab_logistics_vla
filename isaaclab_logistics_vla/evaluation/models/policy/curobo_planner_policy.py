from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from isaaclab_logistics_vla.configs.pinned_eval_curobo import DEFAULT_CUROBO_KINEMATICS_YAML
from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner, RealmanRobotState


def _round_vec(v: np.ndarray, decimals: int = 4) -> List[float]:
    return np.asarray(v, dtype=np.float64).reshape(-1).round(decimals).tolist()


def _angle_between_quats_wxyz_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    a = np.asarray(q1, dtype=np.float64).reshape(4)
    b = np.asarray(q2, dtype=np.float64).reshape(4)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(abs(np.dot(a, b)))
    d = min(1.0, max(0.0, d))
    return float(2.0 * math.acos(d))


@dataclass
class _PlanCache:
    traj: Optional[np.ndarray] = None  # (T, 14) in left7+right7
    step: int = 0


def _interleave_left_right(q_lr: np.ndarray) -> np.ndarray:
    """(14,) left7+right7 -> (14,) interleaved l1,r1,l2,r2,... for env action layout."""
    q_lr = np.asarray(q_lr, dtype=np.float32).reshape(14)
    left = q_lr[:7]
    right = q_lr[7:14]
    out = np.zeros((14,), dtype=np.float32)
    out[0::2] = left
    out[1::2] = right
    return out


class CuroboPlannerPolicy:
    """把 cuRobo 规划当作 policy（Realman 双臂）。

    约定：
    - 从 env 读取当前左右末端在 World 下位姿，作为起点；
    - 目标为“离起点很近”的一个偏移（默认沿 world +x 方向小位移），姿态保持不变；
    - 内部使用 `CuroboPlanner.plan_dual_world_to_world_auto`（自动读取 root/platform 状态）。
    """

    def __init__(
        self,
        env: Any,
        *,
        device: str = "cuda:0",
        cache_path: str = DEFAULT_CUROBO_KINEMATICS_YAML,
        apply_robot_to_curobo_frame_transform: bool = False,
        goal_delta_w: np.ndarray | None = None,
        platform_action: float = 0.5,
        debug_print: bool = True,
        debug_print_every: int = 50,
        debug_coordinates: bool = True,
    ) -> None:
        self.env = env
        self.device = getattr(env, "device", torch.device("cuda:0"))
        self.num_envs = int(getattr(env, "num_envs", 1))

        self.platform_action = float(platform_action)
        self.debug_print = bool(debug_print)
        self.debug_print_every = int(debug_print_every)
        self.debug_coordinates = bool(debug_coordinates)
        self._global_step = 0
        self.goal_delta_w = (
            np.asarray(goal_delta_w, dtype=np.float64).reshape(3)
            if goal_delta_w is not None
            else np.array([0.05, 0.0, 0.0], dtype=np.float64)
        )

        self._cache = [_PlanCache() for _ in range(self.num_envs)]

        def _reader() -> RealmanRobotState:
            isaac_env = self.env.unwrapped
            robot = isaac_env.scene.articulations["robot"]
            root = robot.data.root_state_w  # (N, 13) or (N, 7+)
            root_pos = root[:, 0:3].detach().cpu().numpy()
            root_quat = root[:, 3:7].detach().cpu().numpy()
            # platform_joint 可选：找不到就当 0
            platform_val = 0.0
            try:
                jnames = list(robot.data.joint_names)
                if "platform_joint" in jnames:
                    jidx = jnames.index("platform_joint")
                    platform_val = float(robot.data.joint_pos[0, jidx].detach().cpu().item())
            except Exception:
                platform_val = 0.0
            return RealmanRobotState(
                root_pos_w=root_pos[0],
                root_quat_wxyz=root_quat[0],
                platform_joint_value=platform_val,
            )

        print(
            f"[CuroboPlannerPolicy] 正在构造 CuroboPlanner: "
            f"cache_path={cache_path!r} device={device!r} goal_delta_w={self.goal_delta_w.tolist()}",
            flush=True,
        )
        self.planner = CuroboPlanner(
            device=device,
            cache_path=cache_path,
            apply_robot_to_curobo_frame_transform=apply_robot_to_curobo_frame_transform,
            realman_state_reader=_reader,
        )
        print(
            "[CuroboPlannerPolicy] 已就绪；评测/测试脚本中通过 plan_dual_world_to_world_auto 规划。",
            flush=True,
        )

    def _find_body_index(self, body_names: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
        for i, bn in enumerate(body_names):
            if bn in candidates:
                return i
        for i, bn in enumerate(body_names):
            for c in candidates:
                if c in bn:
                    return i
        return None

    def _get_body_pose_w(self, env_id: int, *, body_name_candidates: Tuple[str, ...]) -> Optional[Dict[str, np.ndarray]]:
        isaac_env = self.env.unwrapped
        robot = isaac_env.scene.articulations["robot"]
        body_names = list(getattr(robot.data, "body_names", []))
        idx = self._find_body_index(body_names, body_name_candidates)
        if idx is None:
            return None
        st = robot.data.body_state_w
        pose = st[env_id, idx]
        pos = pose[0:3].detach().cpu().numpy()
        quat = pose[3:7].detach().cpu().numpy()
        return {"position": np.asarray(pos, dtype=np.float64), "quaternion": np.asarray(quat, dtype=np.float64)}

    def _get_ee_pose_w(self, env_id: int, *, link_name: str) -> Dict[str, np.ndarray]:
        isaac_env = self.env.unwrapped
        robot = isaac_env.scene.articulations["robot"]
        body_names = list(getattr(robot.data, "body_names", []))
        if link_name not in body_names:
            raise KeyError(f"找不到 link_name={link_name!r}，可用 body_names[:10]={body_names[:10]}")
        idx = body_names.index(link_name)
        st = robot.data.body_state_w  # (N, n_body, 13) 常见
        pose = st[env_id, idx]
        pos = pose[0:3].detach().cpu().numpy()
        quat = pose[3:7].detach().cpu().numpy()
        return {"position": np.asarray(pos, dtype=np.float64), "quaternion": np.asarray(quat, dtype=np.float64)}

    def _make_goal_near_start(self, start_pose: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "position": np.asarray(start_pose["position"], dtype=np.float64) + self.goal_delta_w,
            "quaternion": np.asarray(start_pose["quaternion"], dtype=np.float64),
        }

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for i in env_ids.detach().cpu().tolist():
            self._cache[i] = _PlanCache(traj=None, step=0)

    def act(self, obs: Any = None) -> torch.Tensor:
        actions = torch.zeros((self.num_envs, 17), device=self.device, dtype=torch.float32)
        actions[:, 14] = 0.0
        actions[:, 15] = 0.0
        actions[:, 16] = self.platform_action

        for env_id in range(self.num_envs):
            cache = self._cache[env_id]
            if cache.traj is None or cache.step >= cache.traj.shape[0]:
                # 从 env 读取起点（左右末端），并生成近邻目标
                start_left = self._get_ee_pose_w(env_id, link_name="panda_left_hand")
                start_right = self._get_ee_pose_w(env_id, link_name="panda_right_hand")
                goal_left = self._make_goal_near_start(start_left)
                goal_right = self._make_goal_near_start(start_right)

                want_coord_debug = (
                    self.debug_print
                    and self.debug_coordinates
                    and (self._global_step % max(1, self.debug_print_every) == 0)
                )
                out = self.planner.plan_dual_world_to_world_auto(
                    start_poses_world={"left": start_left, "right": start_right},
                    goal_poses_world={"left": goal_left, "right": goal_right},
                    # 先关掉默认障碍（桌子+箱子），用来验证 IK_FAIL 是否由碰撞/障碍约束导致。
                    set_default_world=False,
                    max_attempts=3,
                    timeout=1.0,
                    # cuRobo 要求 graph_search 与 opt 至少启用一个；这里启用 opt，关闭 graph_search，避免 batch graph 限制。
                    enable_graph=False,
                    enable_opt=True,
                    debug_coordinate_check=want_coord_debug,
                )
                if self.debug_print and (self._global_step % max(1, self.debug_print_every) == 0):
                    ik = out.get("ik") or {}
                    traj = out.get("position")
                    solve_ms = getattr(self.planner, "solve_time", None)
                    print(
                        "[curobo_planner_policy]"
                        f" env={env_id}"
                        f" status={out.get('status')}"
                        f" detail={out.get('detail')}"
                        f" ik_success={ik.get('success')}"
                        f" ik_detail={ik.get('detail', '')!r}"
                        f" dof={getattr(self.planner, 'dof', '?')}"
                        f" traj={'None' if traj is None else list(traj.shape)}"
                        f" solve_ms={None if solve_ms is None else round(float(solve_ms), 1)}"
                    )
                    if want_coord_debug and isinstance(out.get("_debug"), dict):
                        dbg = out["_debug"]
                        print("[curobo_planner_policy][coord] apply_frame_transform=", dbg.get("apply_robot_to_curobo_frame_transform"))
                        print("[curobo_planner_policy][coord] kinematics_link_names=", dbg.get("kinematics_link_names"))
                        print("[curobo_planner_policy][coord] motion_gen_arm_order=", dbg.get("motion_gen_arm_order"))
                        print(
                            "[curobo_planner_policy][coord] num_graph_seeds=",
                            dbg.get("motion_gen_num_graph_seeds"),
                            " use_cuda_graph=",
                            dbg.get("motion_gen_use_cuda_graph"),
                            " motion_gen_result_status=",
                            dbg.get("motion_gen_result_status"),
                        )
                        print("[curobo_planner_policy][coord] delta_pos_norm_base_link_m=", dbg.get("delta_pos_norm_base_link_m"))
                        print("[curobo_planner_policy][coord] delta_pos_vec_base_link_m=", dbg.get("delta_pos_vec_base_link_m"))
                        print(
                            "[curobo_planner_policy][coord] ik_goal_success=",
                            dbg.get("ik_goal_success"),
                            " ik_goal_detail=",
                            repr(dbg.get("ik_goal_detail")),
                        )
                        print(
                            "[curobo_planner_policy][coord] ik_goal_left_only_success=",
                            dbg.get("ik_goal_left_only_success"),
                            " detail=",
                            repr(dbg.get("ik_goal_left_only_detail")),
                        )
                        print(
                            "[curobo_planner_policy][coord] ik_goal_left_only_no_collision_success=",
                            dbg.get("ik_goal_left_only_no_collision_success"),
                            " detail=",
                            repr(dbg.get("ik_goal_left_only_no_collision_detail")),
                        )
                        print(
                            "[curobo_planner_policy][coord] ik_goal_right_only_success=",
                            dbg.get("ik_goal_right_only_success"),
                            " detail=",
                            repr(dbg.get("ik_goal_right_only_detail")),
                        )
                        sweep = dbg.get("ik_left_delta_sweep_no_collision")
                        if sweep is not None:
                            print("[curobo_planner_policy][coord] ik_left_delta_sweep_no_collision=", sweep)
                        print("[curobo_planner_policy][coord] base_est_pos_w=", dbg.get("base_link_estimated_pos_w"))
                        sim_base = self._get_body_pose_w(
                            env_id,
                            body_name_candidates=(
                                "dual_rm_75b_description_platform_base_link",
                                "platform_base_link",
                            ),
                        )
                        if sim_base is not None:
                            est = np.asarray(dbg.get("base_link_estimated_pos_w"), dtype=np.float64)
                            sp = sim_base["position"]
                            dpos = float(np.linalg.norm(sp - est))
                            ang_deg = math.degrees(
                                _angle_between_quats_wxyz_rad(sim_base["quaternion"], dbg.get("base_link_estimated_quat_wxyz"))
                            )
                            print(
                                "[curobo_planner_policy][coord] sim_platform_base_pos_w=",
                                _round_vec(sp),
                                "| pos_err_m=",
                                round(dpos, 5),
                                "| quat_angle_deg=",
                                round(ang_deg, 3),
                            )
                        else:
                            isaac_env = self.env.unwrapped
                            bn = list(
                                getattr(isaac_env.scene.articulations["robot"].data, "body_names", [])[:20]
                            )
                            print(
                                "[curobo_planner_policy][coord] 未找到 platform_base_link 刚体名；"
                                f" body_names(前20)={bn}"
                            )
                        print("[curobo_planner_policy][coord] URDF/scene 说明:")
                        print("  ", dbg.get("urdf_reference"))
                        print("  ", dbg.get("scene_reference"))
                if out.get("status") == "Success" and out.get("position") is not None:
                    cache.traj = out["position"]
                    cache.step = 0
                else:
                    cache.traj = None
                    cache.step = 0
                    continue

            q_lr = cache.traj[cache.step]  # (14,) left+right
            q_int = _interleave_left_right(q_lr)
            actions[env_id, 0:14] = torch.from_numpy(q_int).to(device=self.device)
            cache.step += 1

        self._global_step += 1
        return actions

