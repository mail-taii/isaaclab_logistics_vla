# Copyright (c) 2025, Logistics VLA extension contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""
在真实 Isaac Lab 任务场景中使用 ``utils.curobo_planner.CuroboPlanner`` 的示例策略。

典型启动（仓库根目录，需已配置 ``ASSET_ROOT_PATH`` 等）::

    ./isaaclab.sh -p /path/to/isaaclab_logistics_vla/scripts/evaluate_vla.py \\
        --policy curobo_plan --task_scene_name Spawn_ms_st_dense_EnvCfg --num_envs 1

说明:
    - 默认仅对 **env 0** 做规划，并将同一关节目标广播到所有并行环境（便于多 env 可视化）。
    - **默认 ``goal_mode='reach_box'``**：世界系 ``target_box_world_pos``，臂基世界位置 **纯减法**
      得到臂基系目标点；朝向用 ``grasp_pose_candidates_deg`` 多组 (r,p,y) 度 + ``euler_to_quat_isaac``
      依次尝试；``CuroboPlanner.plan_single_arm`` 只动 ``reach_arm``，对侧保持 **当前末端**（臂基系）。
      ``goal_mode='hand_delta'`` 时为「当前手位姿 + 增量」双臂 ``plan_dual``。
    - **臂基世界位姿**：优先 ``robot_registry`` 的 ``arm_base_offset_in_root`` + ``platform_joint`` 与
      ``combine_frame_transforms``；否则 ``platform_base_link`` / root。
    - **关节动作顺序**：cuRobo 左 7 + 右 7 → Isaac 交错 l1,r1,… 自动转换。
    - 可选传入 ``RobotSpec`` / ``WorldSpec``（或 ``WorldSpec.from_yaml``）显式配置机器人与世界障碍；
      未传 ``WorldSpec`` 时规划前为无障碍（空世界）。
    - **双规划器**：同时传入 ``left_robot_spec`` + ``right_robot_spec``（与 ``robot_spec`` 互斥）。
      每个 ``CuroboPlanner`` 以 ``ee_link=spec.left_ee_link`` / ``right_ee_link`` 建 **单链** 模型，需与 URDF 一致（通常各 7 关节）。
      可选 ``dual_left_base_body_contains`` / ``dual_right_base_body_contains`` 指定左右臂基 link 名子串；省略则两侧目标仍用同一套 ``_resolve_arm_base_world``（与单规划器一致）。

环境变量:
    - ``CUROBO_PLAN_ROBOT_ID``：注册表 ``robot_id``，默认 ``realman_franka_dual``；设为空或 ``none`` / ``off`` 则不走注册表，仅用 body/root 回退。
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from isaaclab_logistics_vla import ISAACLAB_LOGISTICS_VLA_EXT_DIR
from isaaclab_logistics_vla.evaluation.robot_registry import RobotEvalConfig, get_robot_eval_config
from isaaclab_logistics_vla.utils.curobo_planner.robot_spec import RobotSpec
from isaaclab_logistics_vla.utils.curobo_planner.world_spec import WorldSpec
from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac

# 送入 cuRobo 的起始关节：左 7 + 右 7（与 verify / URDF 臂链顺序一致）
_Q_CUROBO_NAMES: List[str] = [f"l_joint{i}" for i in range(1, 8)] + [f"r_joint{i}" for i in range(1, 8)]

_IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

# 默认箱子抓取：世界系目标点与抓取姿态候选（角度制 r,p,y）
_DEFAULT_TARGET_BOX_WORLD_POS = np.array([1.025, 1.45, 0.9], dtype=np.float64)
_DEFAULT_GRASP_POSE_CANDIDATES_DEG: List[Tuple[float, float, float]] = [
    (0, 180, 0),
    (0, 0, 0),
    (0, 90, 0),
    (0, 90, 180),
    (0, 90, 90),
    (0, 90, -90),
]


def _world_pos_to_arm_base_subtract(
    target_world: np.ndarray,
    arm_base_pos_w: np.ndarray,
) -> np.ndarray:
    """世界系目标点 → 臂基系位置：仅用臂基世界位置做向量差（假定与 Isaac 轴对齐）。"""
    p = np.asarray(target_world, dtype=np.float64).reshape(3)
    b = np.asarray(arm_base_pos_w, dtype=np.float64).reshape(3)
    return (p - b).astype(np.float32)


def _euler_deg_to_quat_wxyz_numpy(r: float, p: float, y: float) -> np.ndarray:
    qt = euler_to_quat_isaac(r, p, y)
    if isinstance(qt, torch.Tensor):
        return qt.detach().float().cpu().numpy().reshape(4)
    return np.asarray(qt, dtype=np.float64).reshape(4)


def _find_body_index(body_names: List[str], substr: str) -> Optional[int]:
    s = substr.lower()
    for i, n in enumerate(body_names):
        if s in n.lower():
            return i
    return None


def _resolve_arm_base_world(
    robot: Any,
    env_idx: int,
    cfg: Optional[RobotEvalConfig],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """臂基世界 (pos, quat)：registry 的 root + R@offset + platform；否则 platform_base_link；再否则 root。"""
    e = env_idx
    root_pos = robot.data.root_pos_w[e : e + 1, :3]
    root_quat = robot.data.root_quat_w[e : e + 1, :4]
    dev, dt_pos, dt_quat = root_pos.device, root_pos.dtype, root_quat.dtype

    if cfg is not None and cfg.arm_base_offset_in_root is not None:
        ox, oy, oz = cfg.arm_base_offset_in_root
        z_extra = 0.0
        if cfg.platform_joint_name:
            jnames = list(robot.data.joint_names)
            if cfg.platform_joint_name in jnames:
                ji = jnames.index(cfg.platform_joint_name)
                z_extra = float(robot.data.joint_pos[e, ji].item())
        t12 = torch.tensor([[ox, oy, oz + z_extra]], device=dev, dtype=dt_pos)
        q12 = torch.tensor([list(_IDENTITY_QUAT_WXYZ)], device=dev, dtype=dt_quat)
        return combine_frame_transforms(root_pos, root_quat, t12, q12)

    names = list(robot.data.body_names)
    i_base = _find_body_index(names, "platform_base_link")
    if i_base is not None:
        bs = robot.data.body_state_w[e]
        p = bs[i_base, :3].unsqueeze(0)
        q = bs[i_base, 3:7].unsqueeze(0)
        return p, q

    return root_pos, root_quat


def _resolve_body_world_by_substr(
    robot: Any,
    env_idx: int,
    substr: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 body 名子串取该 link 的世界位姿 (pos, quat)，形状与 ``root_pos_w`` 一致。"""
    names = list(robot.data.body_names)
    i = _find_body_index(names, substr)
    if i is None:
        raise RuntimeError(
            f"未在 body_names 中找到含 {substr!r} 的 link；请检查 dual_*_base_body_contains 或场景。"
        )
    e = env_idx
    bs = robot.data.body_state_w[e]
    p = bs[i, :3].unsqueeze(0)
    q = bs[i, 3:7].unsqueeze(0)
    return p, q


def _resolve_dual_arm_bases(
    robot: Any,
    env_idx: int,
    cfg: Optional[RobotEvalConfig],
    left_sub: Optional[str],
    right_sub: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """左/右臂基世界 (pos, quat)。若未提供子串则两侧共用 ``_resolve_arm_base_world``。"""
    if left_sub is not None and right_sub is not None:
        blp, blq = _resolve_body_world_by_substr(robot, env_idx, left_sub)
        brp, brq = _resolve_body_world_by_substr(robot, env_idx, right_sub)
        return blp, blq, brp, brq
    shared_p, shared_q = _resolve_arm_base_world(robot, env_idx, cfg)
    return shared_p, shared_q, shared_p, shared_q


def _resolve_hands_world(
    robot: Any,
    isaac_env: Any,
    env_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """左右手世界位姿；优先 body，否则 ee_frame TCP。"""
    e = env_idx
    names = list(robot.data.body_names)
    i_l = _find_body_index(names, "panda_left_hand")
    i_r = _find_body_index(names, "panda_right_hand")
    if i_l is not None and i_r is not None:
        bs = robot.data.body_state_w[e]
        pl = bs[i_l, :3].unsqueeze(0)
        ql = bs[i_l, 3:7].unsqueeze(0)
        pr = bs[i_r, :3].unsqueeze(0)
        qr = bs[i_r, 3:7].unsqueeze(0)
        return pl, ql, pr, qr

    ee_frame = isaac_env.scene["ee_frame"]
    try:
        frame_names = list(ee_frame.data.target_frame_names)
        i_left = frame_names.index("left_ee_tcp")
        i_right = frame_names.index("right_ee_tcp")
    except (AttributeError, ValueError) as err:
        raise RuntimeError(
            "未找到 panda_*_hand body，且无法从 ee_frame 读取 left_ee_tcp / right_ee_tcp；请检查场景配置。"
        ) from err
    ee_pos_w = ee_frame.data.target_pos_w[e : e + 1, :, :3]
    ee_quat_w = ee_frame.data.target_quat_w[e : e + 1, :, :4]
    return (
        ee_pos_w[:, i_left, :],
        ee_quat_w[:, i_left, :],
        ee_pos_w[:, i_right, :],
        ee_quat_w[:, i_right, :],
    )


def _q_block_lr_to_action_interleaved(q_block: np.ndarray) -> np.ndarray:
    """(14,) 左7+右7 → Isaac 动作前 14 维：l1,r1,l2,r2,…,l7,r7。"""
    q = np.asarray(q_block, dtype=np.float32).reshape(14)
    out = np.empty(14, dtype=np.float32)
    left, right = q[:7], q[7:]
    for i in range(7):
        out[2 * i] = left[i]
        out[2 * i + 1] = right[i]
    return out


def _split_q_lr(q14: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q14, dtype=np.float32).reshape(14)
    return q[:7].copy(), q[7:].copy()


def _merge_lr_trajs(traj_l: np.ndarray, traj_r: np.ndarray) -> np.ndarray:
    """(T1,d)(T2,d) → (max(T), 2d)，较短轨迹在末端保持最后一帧。"""
    traj_l = np.asarray(traj_l, dtype=np.float32)
    traj_r = np.asarray(traj_r, dtype=np.float32)
    if traj_l.ndim != 2 or traj_r.ndim != 2:
        raise ValueError(f"轨迹须为 2 维，得到 {traj_l.shape=} {traj_r.shape=}")
    dl, dr = traj_l.shape[1], traj_r.shape[1]
    n_l, n_r = traj_l.shape[0], traj_r.shape[0]
    T = max(n_l, n_r)
    out_l = np.zeros((T, dl), dtype=np.float32)
    out_r = np.zeros((T, dr), dtype=np.float32)
    for t in range(T):
        out_l[t] = traj_l[min(t, n_l - 1)]
        out_r[t] = traj_r[min(t, n_r - 1)]
    return np.concatenate([out_l, out_r], axis=1)


def _default_urdf_path() -> str:
    p = os.path.join(
        ISAACLAB_LOGISTICS_VLA_EXT_DIR,
        "isaaclab_logistics_vla",
        "assets",
        "robots",
        "realman",
        "realman_franka_ee.urdf",
    )
    if os.path.isfile(p):
        return p
    alt = os.environ.get("CUROBO_PLAN_URDF", "")
    if alt and os.path.isfile(alt):
        return alt
    raise FileNotFoundError(
        f"未找到 Realman URDF: {p} 。请设置环境变量 CUROBO_PLAN_URDF 指向 realman_franka_ee.urdf"
    )


def _parse_robot_eval_cfg(robot_id: Optional[str]) -> Optional[RobotEvalConfig]:
    if robot_id is not None:
        rid = robot_id.strip()
    else:
        rid = os.environ.get("CUROBO_PLAN_ROBOT_ID", "realman_franka_dual").strip()
    if not rid or rid.lower() in ("none", "off"):
        return None
    return get_robot_eval_config(rid)


def _normalize_goal_mode(goal_mode: Optional[str]) -> str:
    """``cjx_reach_box`` 为历史别名，映射为 ``reach_box``。"""
    g = (goal_mode or "reach_box").strip().lower()
    if g == "cjx_reach_box":
        return "reach_box"
    return g


class CuRoboPlanPolicy:
    """使用封装后的 CuroboPlanner，在仿真中执行一条双臂关节空间轨迹。

    可同时传入 ``left_robot_spec`` + ``right_robot_spec`` 使用两个 ``CuroboPlanner``（分离基座 / 单臂 URDF）；与 ``robot_spec`` 互斥。
    """

    def __init__(
        self,
        device: str = "cuda:0",
        urdf_path: Optional[str] = None,
        robot_spec: Optional[RobotSpec] = None,
        left_robot_spec: Optional[RobotSpec] = None,
        right_robot_spec: Optional[RobotSpec] = None,
        world_spec: Optional[WorldSpec] = None,
        interpolation_dt: float = 0.05,
        apply_robot_to_curobo_frame_transform: bool = False,
        robot_id: Optional[str] = None,
        goal_mode: str = "reach_box",
        target_box_world_pos: Optional[Tuple[float, float, float]] = None,
        grasp_pose_candidates_deg: Optional[List[Tuple[float, float, float]]] = None,
        reach_arm: str = "left",
        reach_box_max_attempts: int = 10,
        reach_box_timeout: float = 2.0,
        reach_box_enable_opt: bool = False,
        left_goal_delta_base: tuple[float, float, float] = (0.07, 0.04, 0.06),
        right_goal_delta_base: tuple[float, float, float] = (0.07, -0.04, 0.06),
        gripper_open: float = 0.04,
        warmup_env_steps: int = 5,
        max_plan_attempts: int = 24,
        plan_timeout: float = 12.0,
        enable_graph_plan: bool = False,
        dual_left_base_body_contains: Optional[str] = None,
        dual_right_base_body_contains: Optional[str] = None,
    ):
        self.device = device
        self._robot_spec = robot_spec
        self._left_robot_spec = left_robot_spec
        self._right_robot_spec = right_robot_spec
        if robot_spec is not None and (
            left_robot_spec is not None or right_robot_spec is not None
        ):
            raise ValueError("不要同时传入 robot_spec 与 left_robot_spec / right_robot_spec")
        if (left_robot_spec is None) ^ (right_robot_spec is None):
            raise ValueError("left_robot_spec 与 right_robot_spec 须成对同时传入，或二者均省略")
        dsl = (dual_left_base_body_contains or "").strip() or None
        dsr = (dual_right_base_body_contains or "").strip() or None
        if (dsl is None) ^ (dsr is None):
            raise ValueError(
                "dual_left_base_body_contains 与 dual_right_base_body_contains 须同时设置或同时省略"
            )
        self._dual_left_base_substr = dsl
        self._dual_right_base_substr = dsr
        self.world_spec = world_spec if world_spec is not None else WorldSpec.empty()
        if robot_spec is not None:
            self.urdf_path = robot_spec.urdf_path
        elif left_robot_spec is not None and right_robot_spec is not None:
            self.urdf_path = urdf_path or left_robot_spec.urdf_path
        else:
            self.urdf_path = urdf_path or _default_urdf_path()
        self.interpolation_dt = interpolation_dt
        self.apply_frame_transform = apply_robot_to_curobo_frame_transform
        self._robot_eval_cfg = _parse_robot_eval_cfg(robot_id)
        gm = _normalize_goal_mode(goal_mode)
        if gm not in ("reach_box", "hand_delta"):
            raise ValueError(
                f"goal_mode 须为 'reach_box' 或 'hand_delta'（'cjx_reach_box' 为 reach_box 别名），得到 {goal_mode!r}"
            )
        self.goal_mode = gm
        _tb = (
            target_box_world_pos
            if target_box_world_pos is not None
            else tuple(float(x) for x in _DEFAULT_TARGET_BOX_WORLD_POS)
        )
        self.target_box_world_pos = np.array(_tb, dtype=np.float64).reshape(3)
        self._grasp_candidates = list(
            grasp_pose_candidates_deg if grasp_pose_candidates_deg is not None else _DEFAULT_GRASP_POSE_CANDIDATES_DEG
        )
        ra = (reach_arm or "left").strip().lower()
        if ra not in ("left", "right"):
            raise ValueError(f"reach_arm 须为 'left' 或 'right'，得到 {reach_arm!r}")
        self.reach_arm = ra
        self.reach_box_max_attempts = int(reach_box_max_attempts)
        self.reach_box_timeout = float(reach_box_timeout)
        self.reach_box_enable_opt = bool(reach_box_enable_opt)
        self.left_goal_delta_base = np.array(left_goal_delta_base, dtype=np.float64)
        self.right_goal_delta_base = np.array(right_goal_delta_base, dtype=np.float64)
        self.gripper_open = gripper_open
        self.warmup_env_steps = warmup_env_steps
        self.max_plan_attempts = max_plan_attempts
        self.plan_timeout = plan_timeout
        self.enable_graph_plan = enable_graph_plan

        self._planner: Optional[Any] = None
        self._planner_left: Optional[Any] = None
        self._planner_right: Optional[Any] = None
        self._planned = False
        self._traj: Optional[np.ndarray] = None
        self._traj_idx = 0
        self._env_step_counter = 0
        self._sim_step_s = 0.02
        self._fail_printed = False
        self._goal_frame_logged = False
        self._robot_spec_logged = False
        self._dual_spec_logged = False

    def reset(self) -> None:
        self._planned = False
        self._traj = None
        self._traj_idx = 0
        self._env_step_counter = 0
        self._fail_printed = False
        self._goal_frame_logged = False
        self._robot_spec_logged = False
        self._dual_spec_logged = False
        if self._planner is not None:
            self._planner.reset(reset_seed=True)
        if self._planner_left is not None:
            self._planner_left.reset(reset_seed=True)
        if self._planner_right is not None:
            self._planner_right.reset(reset_seed=True)

    def _lazy_planner(self):
        if self._planner is None:
            from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner

            if self._robot_spec is not None:
                print(
                    f"[CuRoboPlanPolicy] 正在初始化 CuroboPlanner（RobotSpec, 首次较慢）… "
                    f"urdf={self._robot_spec.urdf_path!r} device={self.device}"
                )
                self._planner = CuroboPlanner(
                    robot_spec=self._robot_spec,
                    device=self.device,
                    interpolation_dt=self.interpolation_dt,
                    apply_robot_to_curobo_frame_transform=self.apply_frame_transform,
                    use_cuda_graph=False,
                )
            else:
                print(
                    f"[CuRoboPlanPolicy] 正在初始化 CuroboPlanner（首次较慢）… URDF={self.urdf_path} device={self.device}"
                )
                self._planner = CuroboPlanner(
                    urdf_path=self.urdf_path,
                    device=self.device,
                    interpolation_dt=self.interpolation_dt,
                    apply_robot_to_curobo_frame_transform=self.apply_frame_transform,
                    use_cuda_graph=False,
                )
        return self._planner

    def _uses_dual_planners(self) -> bool:
        return self._left_robot_spec is not None and self._right_robot_spec is not None

    def _lazy_dual_planners(self) -> Tuple[Any, Any]:
        if self._planner_left is None or self._planner_right is None:
            from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner

            assert self._left_robot_spec is not None and self._right_robot_spec is not None
            lee = self._left_robot_spec.left_ee_link
            ree = self._right_robot_spec.right_ee_link
            print(
                f"[CuRoboPlanPolicy] 正在初始化双 CuroboPlanner（首次较慢）… "
                f"左 urdf={self._left_robot_spec.urdf_path!r} ee_link={lee!r}；"
                f"右 urdf={self._right_robot_spec.urdf_path!r} ee_link={ree!r} device={self.device}"
            )
            self._planner_left = CuroboPlanner(
                robot_spec=self._left_robot_spec,
                ee_link=lee,
                device=self.device,
                interpolation_dt=self.interpolation_dt,
                apply_robot_to_curobo_frame_transform=self.apply_frame_transform,
                use_cuda_graph=False,
            )
            self._planner_right = CuroboPlanner(
                robot_spec=self._right_robot_spec,
                ee_link=ree,
                device=self.device,
                interpolation_dt=self.interpolation_dt,
                apply_robot_to_curobo_frame_transform=self.apply_frame_transform,
                use_cuda_graph=False,
            )
        return self._planner_left, self._planner_right

    def _dual_planner_plan(
        self,
        q_current: np.ndarray,
        base_l_pos: torch.Tensor,
        base_l_quat: torch.Tensor,
        base_r_pos: torch.Tensor,
        base_r_quat: torch.Tensor,
        pos_l_w: torch.Tensor,
        quat_l_w: torch.Tensor,
        pos_r_w: torch.Tensor,
        quat_r_w: torch.Tensor,
    ) -> Optional[dict[str, Any]]:
        """双单链规划器：返回与单规划器相同的 dict（``position`` 为 (T,14) 左块+右块）。"""
        pl, pr = self._lazy_dual_planners()
        q_l, q_r = _split_q_lr(q_current)
        out: Optional[dict[str, Any]] = None

        if self.goal_mode == "reach_box":
            bl_np = base_l_pos[0].detach().cpu().numpy().reshape(3)
            br_np = base_r_pos[0].detach().cpu().numpy().reshape(3)
            if self.reach_arm == "left":
                target_local = _world_pos_to_arm_base_subtract(self.target_box_world_pos, bl_np)
            else:
                target_local = _world_pos_to_arm_base_subtract(self.target_box_world_pos, br_np)

            if not self._goal_frame_logged:
                print(
                    f"[CuRoboPlanPolicy] [双规划器] goal_mode=reach_box：动臂={self.reach_arm!r}；"
                    f"世界目标 {self.target_box_world_pos.tolist()} m；抓取候选 {len(self._grasp_candidates)} 组 (r,p,y)°"
                )
                self._goal_frame_logged = True

            for ri, (r_deg, p_deg, y_deg) in enumerate(self._grasp_candidates):
                quat_m = _euler_deg_to_quat_wxyz_numpy(r_deg, p_deg, y_deg)
                goal_pose = {
                    "position": target_local.astype(np.float64),
                    "quaternion": quat_m.astype(np.float64),
                }
                if self.reach_arm == "left":
                    one = pl.plan_one_ee(
                        q_l,
                        goal_pose,
                        max_attempts=self.reach_box_max_attempts,
                        timeout=self.reach_box_timeout,
                        enable_graph=self.enable_graph_plan,
                        enable_opt=self.reach_box_enable_opt,
                    )
                else:
                    one = pr.plan_one_ee(
                        q_r,
                        goal_pose,
                        max_attempts=self.reach_box_max_attempts,
                        timeout=self.reach_box_timeout,
                        enable_graph=self.enable_graph_plan,
                        enable_opt=self.reach_box_enable_opt,
                    )
                if one.get("status") == "Success" and one.get("position") is not None:
                    traj_m = np.asarray(one["position"], dtype=np.float32)
                    if traj_m.shape[1] != 7:
                        raise RuntimeError(
                            f"双规划器 reach_box 期望动臂轨迹每步 7 关节，得到 {traj_m.shape[1]=}"
                        )
                    n_t = traj_m.shape[0]
                    if self.reach_arm == "left":
                        traj_other = np.tile(q_r.astype(np.float32), (n_t, 1))
                        merged = np.concatenate([traj_m, traj_other], axis=1)
                    else:
                        traj_other = np.tile(q_l.astype(np.float32), (n_t, 1))
                        merged = np.concatenate([traj_other, traj_m], axis=1)
                    out = {"status": "Success", "position": merged, "velocity": one.get("velocity")}
                    print(
                        f"[CuRoboPlanPolicy] [双规划器] 抓取姿态 (r,p,y)=({r_deg},{p_deg},{y_deg})° 规划成功"
                    )
                    break
        else:
            if not self._goal_frame_logged:
                print(
                    "[CuRoboPlanPolicy] [双规划器] goal_mode=hand_delta：左右臂各自 plan_one_ee 后合并轨迹"
                )
                self._goal_frame_logged = True

            pos_l_arm, quat_l_arm = subtract_frame_transforms(
                base_l_pos, base_l_quat, pos_l_w, quat_l_w
            )
            pos_r_arm, quat_r_arm = subtract_frame_transforms(
                base_r_pos, base_r_quat, pos_r_w, quat_r_w
            )
            goal_l = pos_l_arm[0].detach().cpu().numpy() + self.left_goal_delta_base
            goal_r = pos_r_arm[0].detach().cpu().numpy() + self.right_goal_delta_base
            quat_l = quat_l_arm[0].detach().cpu().numpy()
            quat_r = quat_r_arm[0].detach().cpu().numpy()
            ol = pl.plan_one_ee(
                q_l,
                {"position": goal_l, "quaternion": quat_l},
                max_attempts=self.max_plan_attempts,
                timeout=self.plan_timeout,
                enable_graph=self.enable_graph_plan,
                enable_opt=True,
            )
            or_ = pr.plan_one_ee(
                q_r,
                {"position": goal_r, "quaternion": quat_r},
                max_attempts=self.max_plan_attempts,
                timeout=self.plan_timeout,
                enable_graph=self.enable_graph_plan,
                enable_opt=True,
            )
            if (
                ol.get("status") == "Success"
                and or_.get("status") == "Success"
                and ol.get("position") is not None
                and or_.get("position") is not None
            ):
                merged = _merge_lr_trajs(
                    np.asarray(ol["position"], dtype=np.float32),
                    np.asarray(or_["position"], dtype=np.float32),
                )
                if merged.shape[1] != 14:
                    raise RuntimeError(
                        f"双规划器合并后关节维应为 14（左+右各 7），得到 {merged.shape[1]=}"
                    )
                out = {"status": "Success", "position": merged}
        return out

    def _gather_arm_q(self, robot, env_ids: int | torch.Tensor = 0) -> np.ndarray:
        jnames = list(robot.data.joint_names)
        q = robot.data.joint_pos
        if isinstance(env_ids, int):
            row = q[env_ids]
        else:
            row = q[env_ids[0]]
        out = np.zeros(14, dtype=np.float32)
        for i, name in enumerate(_Q_CUROBO_NAMES):
            out[i] = float(row[jnames.index(name)].item())
        return out

    def __call__(self, env: Any) -> torch.Tensor:
        isaac_env = env.unwrapped
        device = isaac_env.device
        num_envs = isaac_env.scene.num_envs
        action_dim = isaac_env.action_manager.total_action_dim
        actions = torch.zeros((num_envs, action_dim), device=device)

        robot = isaac_env.scene["robot"]

        dec = getattr(isaac_env.cfg, "decimation", 1)
        sim_dt = float(isaac_env.cfg.sim.dt)
        self._sim_step_s = dec * sim_dt

        actions[:, 14] = self.gripper_open
        actions[:, 15] = self.gripper_open
        pj = list(robot.data.joint_names).index("platform_joint")
        actions[:, 16] = robot.data.joint_pos[:, pj]

        e = 0
        q_current = self._gather_arm_q(robot, e)
        q_act = _q_block_lr_to_action_interleaved(q_current)
        q_t = torch.as_tensor(q_act, device=device, dtype=torch.float32)
        actions[:, :14] = q_t.unsqueeze(0).expand(num_envs, -1)

        self._env_step_counter += 1
        if self._env_step_counter <= self.warmup_env_steps:
            return actions

        if self._traj is not None:
            t = self._traj
            q_cmd = _q_block_lr_to_action_interleaved(t[self._traj_idx])
            actions[:, :14] = torch.as_tensor(q_cmd, device=device, dtype=torch.float32).unsqueeze(0).expand(
                num_envs, -1
            )
            if self._traj_idx < t.shape[0] - 1:
                step_adv = max(1, int(round(self._sim_step_s / self.interpolation_dt)))
                self._traj_idx = min(self._traj_idx + step_adv, t.shape[0] - 1)
            return actions

        if self._planned:
            return actions

        self._planned = True

        base_l_pos, base_l_quat, base_r_pos, base_r_quat = _resolve_dual_arm_bases(
            robot,
            e,
            self._robot_eval_cfg,
            self._dual_left_base_substr,
            self._dual_right_base_substr,
        )
        base_pos, base_quat = base_l_pos, base_l_quat
        pos_l_w, quat_l_w, pos_r_w, quat_r_w = _resolve_hands_world(robot, isaac_env, e)

        pos_l_b, quat_l_b = subtract_frame_transforms(base_pos, base_quat, pos_l_w, quat_l_w)
        pos_r_b, quat_r_b = subtract_frame_transforms(base_pos, base_quat, pos_r_w, quat_r_w)

        out: Optional[dict[str, Any]] = None

        if self._uses_dual_planners():
            pl, pr = self._lazy_dual_planners()
            if not self._dual_spec_logged:
                extra = ""
                if self._dual_left_base_substr and self._dual_right_base_substr:
                    extra = (
                        f" 臂基 body 子串: 左={self._dual_left_base_substr!r} 右={self._dual_right_base_substr!r}"
                    )
                print(f"[CuRoboPlanPolicy] 双规划器模式（left_robot_spec + right_robot_spec）。{extra}")
                self._dual_spec_logged = True
            pl.apply_world(self.world_spec)
            pr.apply_world(self.world_spec)
            out = self._dual_planner_plan(
                q_current,
                base_l_pos,
                base_l_quat,
                base_r_pos,
                base_r_quat,
                pos_l_w,
                quat_l_w,
                pos_r_w,
                quat_r_w,
            )
        else:
            planner = self._lazy_planner()
            if self._robot_spec is not None and not self._robot_spec_logged:
                print(
                    f"[CuRoboPlanPolicy] 共享基座 RobotSpec：base={self._robot_spec.base_link} "
                    f"left_ee={self._robot_spec.left_ee_link} right_ee={self._robot_spec.right_ee_link}"
                )
                self._robot_spec_logged = True
            planner.apply_world(self.world_spec)

            if self.goal_mode == "reach_box":
                arm_base_pos_np = base_pos[0].detach().cpu().numpy().reshape(3)
                target_local = _world_pos_to_arm_base_subtract(self.target_box_world_pos, arm_base_pos_np)
                if self.reach_arm == "left":
                    fixed_pos = pos_r_b[0].detach().cpu().numpy()
                    fixed_quat = quat_r_b[0].detach().cpu().numpy()
                else:
                    fixed_pos = pos_l_b[0].detach().cpu().numpy()
                    fixed_quat = quat_l_b[0].detach().cpu().numpy()
                fixed_pose = {
                    "position": fixed_pos.astype(np.float64),
                    "quaternion": fixed_quat.astype(np.float64),
                }

                if not self._goal_frame_logged:
                    print(
                        f"[CuRoboPlanPolicy] goal_mode=reach_box：世界目标 {self.target_box_world_pos.tolist()} m → "
                        f"臂基系位置 {target_local.tolist()}（臂基世界位置纯减）；"
                        f"动臂={self.reach_arm!r}，对侧保持当前末端；抓取候选 {len(self._grasp_candidates)} 组 (r,p,y)°"
                    )
                    self._goal_frame_logged = True

                for ri, (r_deg, p_deg, y_deg) in enumerate(self._grasp_candidates):
                    quat_m = _euler_deg_to_quat_wxyz_numpy(r_deg, p_deg, y_deg)
                    goal_pose = {
                        "position": target_local.astype(np.float64),
                        "quaternion": quat_m.astype(np.float64),
                    }
                    if ri == 0:
                        print(
                            f"[CuRoboPlanPolicy] 传给 CuroboPlanner 的首个 goal（臂基系）: "
                            f"pos={goal_pose['position'].tolist()}, quat(wxyz)={goal_pose['quaternion'].tolist()}"
                        )
                    out = planner.plan_single_arm(
                        q_current,
                        goal_pose,
                        self.reach_arm,
                        fixed_pose,
                        max_attempts=self.reach_box_max_attempts,
                        timeout=self.reach_box_timeout,
                        enable_graph=self.enable_graph_plan,
                        enable_opt=self.reach_box_enable_opt,
                    )
                    if out.get("status") == "Success":
                        print(
                            f"[CuRoboPlanPolicy] 抓取姿态 (r,p,y)=({r_deg},{p_deg},{y_deg})° 规划成功"
                        )
                        break
            else:
                if not self._goal_frame_logged:
                    if self._robot_eval_cfg is not None and self._robot_eval_cfg.arm_base_offset_in_root is not None:
                        print(
                            f"[CuRoboPlanPolicy] goal_mode=hand_delta：臂基 robot_id={self._robot_eval_cfg.robot_id!r} + "
                            "当前手位姿 + 臂基系增量 → plan_dual"
                        )
                    else:
                        print(
                            "[CuRoboPlanPolicy] goal_mode=hand_delta：回退臂基 + 手/TCP + 增量 → plan_dual"
                        )
                    self._goal_frame_logged = True

                goal_l = pos_l_b[0].detach().cpu().numpy() + self.left_goal_delta_base
                goal_r = pos_r_b[0].detach().cpu().numpy() + self.right_goal_delta_base
                quat_l = quat_l_b[0].detach().cpu().numpy()
                quat_r = quat_r_b[0].detach().cpu().numpy()
                goal_poses = {
                    "left": {"position": goal_l, "quaternion": quat_l},
                    "right": {"position": goal_r, "quaternion": quat_r},
                }
                out = planner.plan_dual(
                    q_current,
                    goal_poses,
                    max_attempts=self.max_plan_attempts,
                    timeout=self.plan_timeout,
                    enable_graph=self.enable_graph_plan,
                    enable_opt=True,
                )

        if out is not None and out.get("status") == "Success" and out.get("position") is not None:
            self._traj = np.asarray(out["position"], dtype=np.float32)
            self._traj_idx = 0
            print(
                f"[CuRoboPlanPolicy] 规划成功: {self._traj.shape[0]} 个插值点, "
                f"interpolation_dt≈{self.interpolation_dt}s"
            )
            q0 = _q_block_lr_to_action_interleaved(self._traj[0])
            actions[:, :14] = torch.as_tensor(q0, device=device, dtype=torch.float32).unsqueeze(0).expand(
                num_envs, -1
            )
        else:
            if not self._fail_printed:
                st = out.get("status") if out is not None else None
                detail = out.get("detail") if out is not None else None
                print(
                    f"[CuRoboPlanPolicy] 规划未成功 (status={st!r}, detail={detail!r})，"
                    "将保持当前臂关节。可试: goal_mode=hand_delta、reach_arm、target_box_world_pos、"
                    "apply_robot_to_curobo_frame_transform、enable_graph_plan。"
                )
                self._fail_printed = True

        return actions
