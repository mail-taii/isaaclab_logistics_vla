"""
cuRobo 运动规划封装（参照 RoboTwin envs/robot/planner.py 中 CuroboPlanner 的用法）：

- 构造期：从 **已存在的 kinematics YAML**（``RobotConfig.from_dict``）或调用方传入的 ``RobotConfig`` 加载模型，再构建 MotionGen、warmup。
  URDF → YAML 的生成在仓库 ``scripts/generate_curobo_robot_kinematics_yaml.py``，**不在**本包内执行。
- 世界：用 ``WorldSpec.from_yaml`` / ``from_cuboids`` 描述长方体障碍，经 ``apply_world`` 更新（空 ``WorldSpec`` 即无障碍）。
- 规划期：双臂 ``plan_dual`` / ``plan_single_arm`` 返回 **dict + CPU numpy**，不向上层暴露 CuRobo Tensor。
- 夹爪：``plan_grippers`` 为线性插值，不经 CuRobo。

坐标：默认将「机器人系 (x 右, y 前, z 上)」下的位姿经绕 z 轴 -90° 对齐到 cuRobo 常用前向 x；若资产已与 cuRobo 一致，设 ``apply_robot_to_curobo_frame_transform=False``。
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from isaaclab_logistics_vla.configs.pinned_eval_curobo import DEFAULT_CUROBO_KINEMATICS_YAML

from .result_utils import motion_gen_batch_result_to_plan_dict, plan_grippers_linear
from .realman_frames import compute_realman_base_link_pose_w, world_pose_to_base_pose
from .robot_spec import RobotSpec
from .world_spec import WorldSpec


@dataclass(frozen=True)
class RealmanRobotState:
    """规划所需的 Realman “实时状态”（全部为 World 系）。"""

    root_pos_w: np.ndarray  # (3,)
    root_quat_wxyz: np.ndarray  # (4,)
    platform_joint_value: float = 0.0


RealmanStateReader = Callable[[], RealmanRobotState]


def _round_list(arr: np.ndarray, decimals: int = 4) -> List[float]:
    return np.asarray(arr, dtype=np.float64).reshape(-1).round(decimals).tolist()


def _load_robot_config_yaml(path: str, tensor_args: TensorDeviceType) -> RobotConfig:
    """从 YAML 加载 ``RobotConfig``；文件须已存在（由仓库脚本或等价流程生成）。"""
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        raise FileNotFoundError(
            f"未找到 kinematics YAML: {expanded}\n"
            "请运行仓库 scripts/generate_curobo_robot_kinematics_yaml.py 从 URDF 生成，"
            "或在构造 CuroboPlanner 时传入 robot_config=...（curobo.types.robot.RobotConfig）。"
        )
    with open(expanded, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return RobotConfig.from_dict(config_dict, tensor_args)


def _infer_robot_dof(kinematics: Any) -> int:
    """从 cuRobo kinematics 推断关节数；兼容 ``joint_limits`` 与 ``get_joint_limits()`` 等 API 差异。"""
    jl = getattr(kinematics, "joint_limits", None)
    if jl is not None:
        shp = getattr(jl, "shape", None)
        if shp is not None and len(shp) > 0:
            return int(shp[0])
        try:
            return int(len(jl))  # type: ignore[arg-type]
        except TypeError:
            pass
    get_lim = getattr(kinematics, "get_joint_limits", None)
    if callable(get_lim):
        try:
            lim = get_lim()
        except Exception:
            lim = None
        if lim is not None:
            shp = getattr(lim, "shape", None)
            if shp is not None and len(shp) > 0:
                return int(shp[0])
            try:
                return int(len(lim))  # type: ignore[arg-type]
            except TypeError:
                pass
    return 14


@contextmanager
def _curobo_autograd_context() -> Iterator[None]:
    """在 Isaac 等 ``torch.inference_mode()`` 嵌套环境中安全运行 cuRobo。

    - 优化需要 ``enable_grad`` 才能 ``backward``。
    - ``MotionGen`` 若在 inference 下构造会得到 inference tensor，随后在非 inference 下
      ``copy_`` 会报 *Inplace update to inference tensor outside InferenceMode*。
    因此相关 **配置加载、MotionGen 构造、warmup、plan_batch** 均应在该上下文中执行。
    """
    with torch.inference_mode(False):
        with torch.enable_grad():
            yield


class CuroboPlanner:
    """
    RoboTwin 风格封装：上层只使用 numpy 与标准 dict。

    主要 API：
        - ``plan_dual`` / ``plan``：双臂末端目标 → ``{status, position, velocity, ...}``
        - ``plan_one_ee``：单链单末端（kinematics YAML 为单 ``ee`` 模型，如 7-DOF）→ 同上 dict，``position`` 为 ``(T, n_dof)``
        - ``plan_single_arm``：单臂移动，另一臂目标位姿由调用方给出（通常取当前末端位姿）
        - ``apply_world`` / ``set_world`` / ``clear_world``：障碍更新（推荐 ``WorldSpec`` + ``apply_world``）
        - ``plan_grippers``：夹爪插值 dict
        - ``reset``：重置 MotionGen 内部状态
    """

    @property
    def dof(self) -> int:
        """当前规划器配置下的自由度（关节数）。"""
        return self._dof

    def __init__(
        self,
        device: str = "cuda:0",
        use_curobo_cache: bool = True,
        cache_path: Optional[str] = None,
        interpolation_dt: float = 0.05,
        apply_robot_to_curobo_frame_transform: bool = True,
        use_cuda_graph: bool = False,
        robot_spec: Optional[RobotSpec] = None,
        robot_config: Optional[RobotConfig] = None,
        *,
        realman_state_reader: Optional[RealmanStateReader] = None,
    ):
        self.device = device
        self.tensor_args = TensorDeviceType(device=device)
        self.apply_frame_transform = apply_robot_to_curobo_frame_transform
        self.interpolation_dt = interpolation_dt

        eff_cache: Optional[str] = None
        if robot_config is None:
            if robot_spec is not None:
                eff_cache = cache_path if cache_path is not None else robot_spec.cache_path
            else:
                eff_cache = cache_path
            if eff_cache is None and use_curobo_cache:
                # 团队固定：包内 ``assets/curobo/realman_kinematics.yaml``（见 configs/pinned_eval_curobo.py）
                eff_cache = DEFAULT_CUROBO_KINEMATICS_YAML
            if eff_cache is None:
                raise ValueError(
                    "须提供 robot_config、cache_path / RobotSpec.cache_path，"
                    "或将 use_curobo_cache=True 以使用包内默认 kinematics YAML（须已预生成，见 pinned_eval_curobo）。"
                )

        if robot_config is not None:
            print(f"[CuroboPlanner] 加载运动学: 内存 RobotConfig, device={device!r}", flush=True)
        else:
            assert eff_cache is not None
            print(
                f"[CuroboPlanner] 加载运动学 YAML: {os.path.abspath(eff_cache)} | "
                f"device={device!r} | apply_frame_transform={apply_robot_to_curobo_frame_transform}",
                flush=True,
            )

        # 整块放入非 inference 上下文，避免内部缓冲区成为 inference tensor 且与 warmup/plan 冲突
        with _curobo_autograd_context():
            if robot_config is not None:
                self.robot_config = robot_config
            else:
                assert eff_cache is not None
                self.robot_config = _load_robot_config_yaml(eff_cache, self.tensor_args)
            # world_model 不可为 None：否则 cuRobo 不会创建 world_coll_checker，后续 update_world 会崩。
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
                WorldConfig(),
                tensor_args=self.tensor_args,
                interpolation_dt=interpolation_dt,
                use_cuda_graph=use_cuda_graph,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
            )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        
        # 计算自由度：不同 cuRobo 版本 kinematics 可能是 joint_limits 属性或 get_joint_limits()
        self._dof = _infer_robot_dof(self.robot_config.kinematics)

        self.world_config = WorldConfig()
        self._update_world()
        
        print(
            f"[CuroboPlanner] MotionGen 已 warmup，就绪: dof={self._dof}，"
            f"realman_state_reader={'已注入' if realman_state_reader is not None else '未注入'}",
            flush=True,
        )

        self.last_result: Any = None
        self.last_plan_dict: Optional[dict[str, Any]] = None
        self._ik_solver: Any = None
        self._ik_solver_no_collision: Any = None
        self._realman_state_reader: Optional[RealmanStateReader] = realman_state_reader

        self.rotation_transform = self._get_rotation_transform()
    
    def set_realman_state_reader(self, reader: Optional[RealmanStateReader]) -> None:
        """注入/更新 Realman 状态读取器；用于 *_auto 一键接口。"""
        self._realman_state_reader = reader

    def _read_realman_state(self) -> RealmanRobotState:
        if self._realman_state_reader is None:
            raise RuntimeError(
                "未设置 realman_state_reader。请在构造 CuroboPlanner 时传入 realman_state_reader=...，"
                "或先调用 planner.set_realman_state_reader(...)。"
            )
        st = self._realman_state_reader()
        # 允许 reader 返回 numpy/列表等，只要能转成正确形状
        return RealmanRobotState(
            root_pos_w=np.asarray(st.root_pos_w, dtype=np.float64).reshape(3),
            root_quat_wxyz=np.asarray(st.root_quat_wxyz, dtype=np.float64).reshape(4),
            platform_joint_value=float(st.platform_joint_value),
        )

    def _get_rotation_transform(self) -> np.ndarray:
        """绕 z 轴 -90°：机器人 y 朝前 → cuRobo x 朝前（位置用）。"""
        theta = -math.pi / 2
        return np.array(
            [
            [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def _quat_rotate_z(self, quaternion_wxyz: np.ndarray, angle_rad: float) -> np.ndarray:
        """绕世界 z 轴旋转四元数（wxyz）。"""
        half = angle_rad * 0.5
        q_rot = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
        w1, x1, y1, z1 = q_rot
        w2, x2, y2, z2 = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z], dtype=np.float64)

    def _transform_pose(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.apply_frame_transform:
            return np.asarray(position, dtype=np.float64), np.asarray(quaternion, dtype=np.float64)
        position_curobo = self.rotation_transform @ np.asarray(position, dtype=np.float64).reshape(3)
        quaternion_curobo = self._quat_rotate_z(np.asarray(quaternion).reshape(4), -math.pi / 2)
        return position_curobo, quaternion_curobo
    
    def _dual_arm_link_names_and_order(self) -> Tuple[List[str], List[str]]:
        """与 ``plan_dual`` 一致：按 kinematics.link_names 推断 MotionGen 末端顺序（left/right）。"""
        kin = self.robot_config.kinematics
        link_names = getattr(kin, "link_names", None)
        if link_names is None:
            link_names = getattr(kin, "link_name", None)
        try:
            link_names_list = list(link_names) if link_names is not None else []
        except Exception:
            link_names_list = []

        def _arm_for_idx(i: int, link_name: Optional[str]) -> str:
            if link_name is not None:
                n = str(link_name).lower()
                if "left" in n:
                    return "left"
                if "right" in n:
                    return "right"
            return "left" if i == 0 else "right"

        if len(link_names_list) >= 2:
            order = [_arm_for_idx(0, link_names_list[0]), _arm_for_idx(1, link_names_list[1])]
        else:
            order = ["left", "right"]
        return link_names_list, order
    
    def _update_world(self) -> None:
        self.motion_gen.update_world(self.world_config)
    
    def set_world(self, obstacles: List[Dict[str, np.ndarray]]) -> None:
        """
        设置世界障碍物（长方体列表）。

        每个元素字典字段：
            - ``position``: (3,) 机器人约定坐标系下的位置
            - ``size`` / ``dims``: (3,) 长方体尺寸
            - ``quaternion``: 可选 (4,) wxyz，默认单位四元数
        """
        cuboids: List[Cuboid] = []
        for i, obs in enumerate(obstacles):
            pos = np.asarray(obs["position"], dtype=np.float64).reshape(3)
            size = obs.get("size", obs.get("dims"))
            if size is None:
                raise KeyError("obstacle 需要 'size' 或 'dims'")
            size = np.asarray(size, dtype=np.float64).reshape(3)
            quat = obs.get("quaternion", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
            pos_c, quat_c = self._transform_pose(pos, quat)
            pose = np.concatenate([pos_c, quat_c]).astype(np.float64).tolist()
            cuboids.append(
                Cuboid(
                    name=obs.get("name", f"obs_{i}"),
                    pose=pose,
                    dims=size.astype(float).tolist(),
                )
            )
        self.world_config = WorldConfig(cuboid=cuboids)
        self._update_world()

    def clear_world(self) -> None:
        self.world_config = WorldConfig()
        self._update_world()

    def apply_world(self, spec: WorldSpec) -> None:
        """使用 :class:`WorldSpec` 更新障碍（空则等价于 :meth:`clear_world`）。"""
        if not spec.obstacles:
            self.clear_world()
            return
        self.set_world(spec.to_planner_obstacles())

    # ---------------------------------------------------------------------
    # IK helpers (cuRobo IKSolver)
    # ---------------------------------------------------------------------

    def _get_ik_solver(self) -> Any:
        """惰性构造 cuRobo `IKSolver`（用于把 EE 位姿逆解为关节角）。"""
        if self._ik_solver is not None:
            return self._ik_solver
        try:
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        except Exception as e:
            raise RuntimeError(f"无法导入 cuRobo IKSolver: {e}") from e

        # 关键：IKSolver 内部会分配/缓存张量；若在外层 inference_mode 下构造，
        # 会产生 inference tensor，随后在非 inference 下 reset/solve 会触发
        # "Inplace update to inference tensor outside InferenceMode"。
        # 因此强制在非 inference 上下文中完成初始化。
        with _curobo_autograd_context():
            # 共享 MotionGen 的 world collision checker 更省显存；没有则退化为无碰撞 IK
            world_coll = getattr(self.motion_gen_config, "world_coll_checker", None)
            ik_cfg = IKSolverConfig.load_from_robot_config(
                self.robot_config,
                WorldConfig(),
                tensor_args=self.tensor_args,
                world_coll_checker=world_coll,
                # 评估侧常在 batch/循环中调用，关闭 cuda graph 避免固定 batch/seed 约束与相关 warning。
                use_cuda_graph=False,
            )
            self._ik_solver = IKSolver(ik_cfg)
        return self._ik_solver

    def _get_ik_solver_no_collision(self) -> Any:
        """用于调试：构造一个**不带碰撞检查**的 IKSolver，区分“不可达” vs “碰撞导致不可行”。

        注意：只用于诊断，不建议用于最终规划。
        """
        if self._ik_solver_no_collision is not None:
            return self._ik_solver_no_collision
        try:
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        except Exception as e:
            raise RuntimeError(f"无法导入 cuRobo IKSolver: {e}") from e

        with _curobo_autograd_context():
            ik_cfg = IKSolverConfig.load_from_robot_config(
                self.robot_config,
                WorldConfig(),
                tensor_args=self.tensor_args,
                world_coll_checker=None,  # 关键：禁用碰撞
                use_cuda_graph=False,
            )
            self._ik_solver_no_collision = IKSolver(ik_cfg)
        return self._ik_solver_no_collision

    def ik_dual_no_collision(
        self,
        goal_poses_b: Dict[str, Dict[str, np.ndarray]],
        *,
        seed_q: Optional[np.ndarray] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """调试用：双臂 IK（base_link 系输入），**禁用碰撞检查**。"""
        ik = self._get_ik_solver_no_collision()

        kin = self.robot_config.kinematics
        link_names = getattr(kin, "link_names", None)
        if link_names is None:
            link_names = getattr(kin, "link_name", None)
        if link_names is None:
            link_names = []
        try:
            link_names_list = list(link_names)
        except Exception:
            link_names_list = []

        def _infer_arm_from_link_name(name: str) -> Optional[str]:
            n = (name or "").lower()
            if "left" in n:
                return "left"
            if "right" in n:
                return "right"
            if n.endswith("_l") or n.endswith("-l"):
                return "left"
            if n.endswith("_r") or n.endswith("-r"):
                return "right"
            return None

        primary_link = str(link_names_list[0]) if len(link_names_list) >= 1 else "panda_left_hand"
        primary_arm = _infer_arm_from_link_name(primary_link) or "left"
        other_arm = "right" if primary_arm == "left" else "left"

        primary_pos = np.asarray(goal_poses_b[primary_arm]["position"], dtype=np.float32).reshape(1, 1, 3)
        primary_quat = np.asarray(goal_poses_b[primary_arm]["quaternion"], dtype=np.float32).reshape(1, 1, 4)
        goal = Pose(
            position=self.tensor_args.to_device(primary_pos),
            quaternion=self.tensor_args.to_device(primary_quat),
        )

        other_link = str(link_names_list[1]) if len(link_names_list) >= 2 else (
            "panda_right_hand" if other_arm == "right" else "panda_left_hand"
        )
        other_pos = np.asarray(goal_poses_b[other_arm]["position"], dtype=np.float32).reshape(1, 1, 3)
        other_quat = np.asarray(goal_poses_b[other_arm]["quaternion"], dtype=np.float32).reshape(1, 1, 4)
        link_poses = {
            other_link: Pose(
                position=self.tensor_args.to_device(other_pos),
                quaternion=self.tensor_args.to_device(other_quat),
            )
        }

        retract = None
        seeds = None
        if seed_q is not None:
            q = np.asarray(seed_q, dtype=np.float32).reshape(1, -1)
            if q.shape[1] != self._dof:
                raise ValueError(f"seed_q 期望 ({self._dof},)，得到 {q.shape}")
            retract = self.tensor_args.to_device(q)
            seeds = self.tensor_args.to_device(q.reshape(1, 1, -1))

        with _curobo_autograd_context():
            result = ik.solve_batch(
                goal,
                retract_config=retract,
                seed_config=seeds,
                return_seeds=return_seeds,
                num_seeds=num_seeds,
                link_poses=link_poses,
            )

        out: Dict[str, Any] = {"success": False, "q": None}
        try:
            succ = result.success
            # cuRobo 不同版本 success 可能是 (batch,) / (seeds,batch) 等，统一取第一个标量
            if hasattr(succ, "reshape"):
                ok = bool(succ.reshape(-1)[0].item())
            else:
                ok = bool(succ)
        except Exception:
            ok = False
        if not ok:
            out["detail"] = "IK_FAIL"
            return out
        q_sol = getattr(result, "solution", None)
        if q_sol is None:
            q_sol = getattr(result, "q", None)
        if q_sol is None:
            out["detail"] = "IK_NO_SOLUTION_TENSOR"
            return out
        try:
            import torch

            if isinstance(q_sol, torch.Tensor):
                q0 = q_sol[0, 0] if q_sol.dim() == 3 else q_sol[0]
                out["q"] = q0.detach().float().cpu().numpy()
            else:
                out["q"] = np.asarray(q_sol, dtype=np.float32).reshape(-1)[: self._dof]
        except Exception as e:
            out["detail"] = f"IK_EXTRACT_FAIL: {e}"
            return out
        out["success"] = True
        out["detail"] = f"no_collision primary_link={primary_link} other_link={other_link}"
        return out

    def ik_one_ee(
        self,
        goal_pose_b: Dict[str, np.ndarray],
        *,
        seed_q: Optional[np.ndarray] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """对单末端目标做 IK（base_link 系输入），返回 numpy 关节向量。

        说明：
        - 用于“用户给 EE 起始位姿 → 逆解关节初值 → 再调用 MotionGen 规划”。
        - `seed_q`（若给）会同时作为 `retract_config` 与 `seed_config`，倾向得到离 seed 更近的解。
        """
        ik = self._get_ik_solver()
        pos = np.asarray(goal_pose_b["position"], dtype=np.float32).reshape(1, 1, 3)
        quat = np.asarray(goal_pose_b["quaternion"], dtype=np.float32).reshape(1, 1, 4)
        goal = Pose(
            position=self.tensor_args.to_device(pos),
            quaternion=self.tensor_args.to_device(quat),
        )

        retract = None
        seeds = None
        if seed_q is not None:
            q = np.asarray(seed_q, dtype=np.float32).reshape(1, -1)
            if q.shape[1] != self._dof:
                raise ValueError(f"seed_q 期望 ({self._dof},)，得到 {q.shape}")
            retract = self.tensor_args.to_device(q)
            seeds = self.tensor_args.to_device(q.reshape(1, 1, -1))  # (n=1, batch=1, dof)

        with _curobo_autograd_context():
            result = ik.solve_batch(
                goal,
                retract_config=retract,
                seed_config=seeds,
                return_seeds=return_seeds,
                num_seeds=num_seeds,
            )

        out: Dict[str, Any] = {"success": False, "q": None}
        try:
            succ = result.success
            ok = bool(succ[0].item()) if hasattr(succ, "shape") else bool(succ)
        except Exception:
            ok = False
        if not ok:
            out["detail"] = "IK_FAIL"
            return out

        q_sol = getattr(result, "solution", None)
        if q_sol is None:
            q_sol = getattr(result, "q", None)
        if q_sol is None:
            out["detail"] = "IK_NO_SOLUTION_TENSOR"
            return out

        try:
            import torch

            if isinstance(q_sol, torch.Tensor):
                # 常见 shape: (return_seeds, batch, dof) 或 (batch, dof)
                if q_sol.dim() == 3:
                    q0 = q_sol[0, 0]
                elif q_sol.dim() == 2:
                    q0 = q_sol[0]
                else:
                    q0 = q_sol.reshape(-1)[: self._dof]
                out["q"] = q0.detach().float().cpu().numpy()
            else:
                out["q"] = np.asarray(q_sol, dtype=np.float32).reshape(-1)[: self._dof]
        except Exception as e:
            out["detail"] = f"IK_EXTRACT_FAIL: {e}"
            return out

        out["success"] = True
        return out

    def ik_dual(
        self,
        goal_poses_b: Dict[str, Dict[str, np.ndarray]],
        *,
        seed_q: Optional[np.ndarray] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """双臂 IK：输入 base_link 系下左右末端位姿，逆解得到整条关节向量。

        依赖 cuRobo IKSolver 的 `link_poses` 功能：主 `goal_pose` 对应一个末端，其它末端通过字典传入。

        Returns:
            {"success": bool, "q": np.ndarray|None, "detail"?: str}
        """
        ik = self._get_ik_solver()

        # 尽量从 robot_config.kinematics 里取出多末端 link_names；取不到则默认 left/right 的顺序。
        kin = self.robot_config.kinematics
        link_names = getattr(kin, "link_names", None)
        if link_names is None:
            link_names = getattr(kin, "link_name", None)
        if link_names is None:
            link_names = []
        try:
            link_names_list = list(link_names)
        except Exception:
            link_names_list = []

        def _infer_arm_from_link_name(name: str) -> Optional[str]:
            n = (name or "").lower()
            if "left" in n:
                return "left"
            if "right" in n:
                return "right"
            if n.endswith("_l") or n.endswith("-l"):
                return "left"
            if n.endswith("_r") or n.endswith("-r"):
                return "right"
            return None

        # 关键：cuRobo IKSolver 的主 `goal` 对应的是其“主末端 link”（通常是 kinematics.link_names[0]）。
        # 因此必须按 link_names 的真实顺序把 left/right 目标映射过去，避免左右对调导致 IK/规划失败。
        primary_link = str(link_names_list[0]) if len(link_names_list) >= 1 else "panda_left_hand"
        primary_arm = _infer_arm_from_link_name(primary_link) or "left"
        other_arm = "right" if primary_arm == "left" else "left"

        primary_pos = np.asarray(goal_poses_b[primary_arm]["position"], dtype=np.float32).reshape(1, 1, 3)
        primary_quat = np.asarray(goal_poses_b[primary_arm]["quaternion"], dtype=np.float32).reshape(1, 1, 4)
        goal = Pose(
            position=self.tensor_args.to_device(primary_pos),
            quaternion=self.tensor_args.to_device(primary_quat),
        )

        # 其它末端目标：优先用 link_names 里剩下的那个；没有就用常见命名兜底
        other_link = None
        if len(link_names_list) >= 2:
            other_link = str(link_names_list[1])
        if not other_link:
            other_link = "panda_right_hand" if other_arm == "right" else "panda_left_hand"

        other_pos = np.asarray(goal_poses_b[other_arm]["position"], dtype=np.float32).reshape(1, 1, 3)
        other_quat = np.asarray(goal_poses_b[other_arm]["quaternion"], dtype=np.float32).reshape(1, 1, 4)
        link_poses = {
            other_link: Pose(
                position=self.tensor_args.to_device(other_pos),
                quaternion=self.tensor_args.to_device(other_quat),
            )
        }

        retract = None
        seeds = None
        if seed_q is not None:
            q = np.asarray(seed_q, dtype=np.float32).reshape(1, -1)
            if q.shape[1] != self._dof:
                raise ValueError(f"seed_q 期望 ({self._dof},)，得到 {q.shape}")
            retract = self.tensor_args.to_device(q)
            seeds = self.tensor_args.to_device(q.reshape(1, 1, -1))

        with _curobo_autograd_context():
            result = ik.solve_batch(
                goal,
                retract_config=retract,
                seed_config=seeds,
                return_seeds=return_seeds,
                num_seeds=num_seeds,
                link_poses=link_poses,
            )

        out: Dict[str, Any] = {"success": False, "q": None}
        try:
            succ = result.success
            ok = bool(succ[0].item()) if hasattr(succ, "shape") else bool(succ)
        except Exception:
            ok = False
        if not ok:
            out["detail"] = "IK_FAIL"
            return out

        q_sol = getattr(result, "solution", None)
        if q_sol is None:
            q_sol = getattr(result, "q", None)
        if q_sol is None:
            out["detail"] = "IK_NO_SOLUTION_TENSOR"
            return out

        try:
            import torch

            if isinstance(q_sol, torch.Tensor):
                if q_sol.dim() == 3:
                    q0 = q_sol[0, 0]
                elif q_sol.dim() == 2:
                    q0 = q_sol[0]
                else:
                    q0 = q_sol.reshape(-1)[: self._dof]
                out["q"] = q0.detach().float().cpu().numpy()
            else:
                out["q"] = np.asarray(q_sol, dtype=np.float32).reshape(-1)[: self._dof]
        except Exception as e:
            out["detail"] = f"IK_EXTRACT_FAIL: {e}"
            return out

        out["success"] = True
        out["detail"] = f"primary_link={primary_link} primary_arm={primary_arm} other_link={other_link} other_arm={other_arm}"
        return out

    # ---------------------------------------------------------------------
    # Realman one-click helpers (World → base_link → planner)
    # ---------------------------------------------------------------------

    def _realman_base_pose_w(
        self,
        *,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        platform_joint_value: float = 0.0,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Realman root 位姿 + 平台关节 → 估算 base_link 的 world 位姿。

        该逻辑参考 `cjx` 分支注册表中的默认 offset（来自 URDF platform_joint origin）。
        """
        return compute_realman_base_link_pose_w(
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            platform_joint_value=platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )

    def plan_dual_from_world(
        self,
        start_joint_positions: np.ndarray,
        goal_poses_world: Dict[str, Dict[str, np.ndarray]],
        *,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        platform_joint_value: float = 0.0,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Realman 一键：世界系目标 → base_link 系 → `plan_dual`。

        Args:
            goal_poses_world: 与 `plan_dual` 同结构，但 position/quaternion 在 **World** 系下。
            root_pos_w/root_quat_wxyz/platform_joint_value: 用于估算 base_link 在 world 下位姿。
        """
        base_pos_w, base_quat = self._realman_base_pose_w(
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            platform_joint_value=platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )
        goal_poses_b: Dict[str, Dict[str, np.ndarray]] = {}
        for arm in ("left", "right"):
            g = goal_poses_world[arm]
            p_b, q_b = world_pose_to_base_pose(
                pos_w=np.asarray(g["position"], dtype=np.float64),
                quat_wxyz=np.asarray(g["quaternion"], dtype=np.float64),
                base_pos_w=base_pos_w,
                base_quat_wxyz=base_quat,
            )
            goal_poses_b[arm] = {
                "position": np.asarray(p_b, dtype=np.float64),
                "quaternion": np.asarray(q_b, dtype=np.float64),
            }
        return self.plan_dual(start_joint_positions, goal_poses_b, **kwargs)

    # （已回退）不提供 “读当前关节角当 q_start” 的 auto 版本；当前 policy 使用起始末端位姿→IK→规划链路。

    def plan_one_ee_from_world(
        self,
        start_joint_positions: np.ndarray,
        goal_pose_world: Dict[str, np.ndarray],
        *,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        platform_joint_value: float = 0.0,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Realman 一键：世界系单末端目标 → base_link 系 → `plan_one_ee`。"""
        base_pos_w, base_quat = self._realman_base_pose_w(
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            platform_joint_value=platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )
        p_b, q_b = world_pose_to_base_pose(
            pos_w=np.asarray(goal_pose_world["position"], dtype=np.float64),
            quat_wxyz=np.asarray(goal_pose_world["quaternion"], dtype=np.float64),
            base_pos_w=base_pos_w,
            base_quat_wxyz=base_quat,
        )
        return self.plan_one_ee(
            start_joint_positions,
            {"position": np.asarray(p_b, dtype=np.float64), "quaternion": np.asarray(q_b, dtype=np.float64)},
            **kwargs,
        )

    def set_world_from_world(
        self,
        obstacles_world: List[Dict[str, np.ndarray]],
        *,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        platform_joint_value: float = 0.0,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
    ) -> None:
        """Realman 一键：世界系障碍 → base_link 系 → `set_world`。"""
        base_pos_w, base_quat = self._realman_base_pose_w(
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            platform_joint_value=platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )
        obs_b: List[Dict[str, np.ndarray]] = []
        for o in obstacles_world:
            pos = np.asarray(o["position"], dtype=np.float64)
            quat = np.asarray(
                o.get("quaternion", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)), dtype=np.float64
            )
            p_b, q_b = world_pose_to_base_pose(
                pos_w=pos,
                quat_wxyz=quat,
                base_pos_w=base_pos_w,
                base_quat_wxyz=base_quat,
            )
            out: Dict[str, np.ndarray] = {
                "name": o.get("name", "obs"),
                "position": np.asarray(p_b, dtype=np.float64),
                "quaternion": np.asarray(q_b, dtype=np.float64),
            }
            if "size" in o:
                out["size"] = np.asarray(o["size"], dtype=np.float64)
            if "dims" in o:
                out["dims"] = np.asarray(o["dims"], dtype=np.float64)
            obs_b.append(out)
        self.set_world(obs_b)

    def set_default_realman_scene_world(
        self,
        *,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        platform_joint_value: float = 0.0,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
    ) -> None:
        """设置本 benchmark 默认世界障碍（Realman：1 张桌子 + 6 个箱子，World 系输入）。

        - 桌子位置来自 `tasks/base_scene_cfg.py` 的 `e_table`（pos=(0.9, 3.5, 0), scale=0.8）。
        - 6 个箱子位置来自 `s_box_1..3` / `t_box_1..3` 的 init_state。
        - cuboid 尺寸采用历史 cuRobo 配置的近似：箱子 dims=(0.56,0.36,0.23)，桌子 dims=(1.2,2.0,0.75)。
        """
        # world positions (meters) from BaseOrderSceneCfg defaults:
        table_pos_w = np.array([0.9, 3.5, 0.0], dtype=np.float64)
        box_pos_w = {
            "s_box_1": (1.57989, 1.33474, 0.750),
            "s_box_2": (1.025, 1.33614, 0.725),
            "s_box_3": (0.51025, 1.33614, 0.750),
            "t_box_1": (1.57989, 3.4429, 0.82),
            "t_box_2": (1.025, 3.4429, 0.82),
            "t_box_3": (0.510, 3.4429, 0.82),
        }

        obstacles_world: List[Dict[str, np.ndarray]] = []
        obstacles_world.append(
            {
                "name": "table",
                "position": table_pos_w,
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "dims": np.array([1.2, 2.0, 0.75], dtype=np.float64),
            }
        )
        for name, p in box_pos_w.items():
            obstacles_world.append(
                {
                    "name": name,
                    "position": np.array(p, dtype=np.float64),
                    "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    "dims": np.array([0.56, 0.36, 0.23], dtype=np.float64),
                }
            )

        self.set_world_from_world(
            obstacles_world,
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            platform_joint_value=platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )

    def plan_one_ee_world_to_world(
        self,
        *,
        start_pose_world: Dict[str, np.ndarray],
        goal_pose_world: Dict[str, np.ndarray],
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        platform_joint_value: float = 0.0,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
        set_default_world: bool = True,
        ik_seed_q: Optional[np.ndarray] = None,
        ik_num_seeds: Optional[int] = None,
        **plan_kwargs: Any,
    ) -> dict[str, Any]:
        """顶层一键：World 系起始 EE 位姿 + World 系目标 EE 位姿 → IK 求 q_start → 规划。

        适用于“用户只给世界坐标系起点/终点（位姿）”的接口层。
        """
        if set_default_world:
            self.set_default_realman_scene_world(
                root_pos_w=root_pos_w,
                root_quat_wxyz=root_quat_wxyz,
                platform_joint_value=platform_joint_value,
                arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
            )

        # 1) World → base_link
        base_pos_w, base_quat = self._realman_base_pose_w(
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            platform_joint_value=platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )
        start_p_b, start_q_b = world_pose_to_base_pose(
            pos_w=np.asarray(start_pose_world["position"], dtype=np.float64),
            quat_wxyz=np.asarray(start_pose_world["quaternion"], dtype=np.float64),
            base_pos_w=base_pos_w,
            base_quat_wxyz=base_quat,
        )
        goal_p_b, goal_q_b = world_pose_to_base_pose(
            pos_w=np.asarray(goal_pose_world["position"], dtype=np.float64),
            quat_wxyz=np.asarray(goal_pose_world["quaternion"], dtype=np.float64),
            base_pos_w=base_pos_w,
            base_quat_wxyz=base_quat,
        )

        # 2) IK for start
        ik_out = self.ik_one_ee(
            {"position": start_p_b, "quaternion": start_q_b},
            seed_q=ik_seed_q,
            num_seeds=ik_num_seeds,
        )
        if not ik_out.get("success"):
            return {
                "status": "Fail",
                "position": None,
                "velocity": None,
                "detail": f"IK_START_FAIL:{ik_out.get('detail', '?')}",
                "ik": ik_out,
            }

        q_start = np.asarray(ik_out["q"], dtype=np.float32).reshape(-1)
        out = self.plan_one_ee(
            q_start,
            {"position": goal_p_b, "quaternion": goal_q_b},
            **plan_kwargs,
        )
        out["ik"] = ik_out
        return out

    def plan_dual_world_to_world(
        self,
        *,
        start_poses_world: Dict[str, Dict[str, np.ndarray]],
        goal_poses_world: Dict[str, Dict[str, np.ndarray]],
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        platform_joint_value: float = 0.0,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
        set_default_world: bool = True,
        ik_seed_q: Optional[np.ndarray] = None,
        ik_num_seeds: Optional[int] = None,
        debug_coordinate_check: bool = False,
        **plan_kwargs: Any,
    ) -> dict[str, Any]:
        """顶层一键（双臂）：World 系起始/目标末端位姿 → IK 求 q_start → `plan_dual`。

        关键点：
        - 先用 Realman root→base_link（含 platform_joint）解决 base_link 的 world 位姿；
        - 再把 world 下的左右末端位姿变换到 base_link 系；
        - 对起始位姿做双臂 IK 得到 `q_start`；
        - 设置（可选）默认障碍并规划到目标位姿。

        若 ``debug_coordinate_check=True``，返回 dict 会多一个键 ``_debug``，便于对照 URDF/仿真核对
        root→platform_base_link 与 World→base_link 末端变换（不参与规划逻辑）。
        """
        if set_default_world:
            self.set_default_realman_scene_world(
                root_pos_w=root_pos_w,
                root_quat_wxyz=root_quat_wxyz,
                platform_joint_value=platform_joint_value,
                arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
            )

        base_pos_w, base_quat = self._realman_base_pose_w(
            root_pos_w=root_pos_w,
            root_quat_wxyz=root_quat_wxyz,
            platform_joint_value=platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )

        def _to_b(pose_w: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            return world_pose_to_base_pose(
                pos_w=np.asarray(pose_w["position"], dtype=np.float64),
                quat_wxyz=np.asarray(pose_w["quaternion"], dtype=np.float64),
                base_pos_w=base_pos_w,
                base_quat_wxyz=base_quat,
            )

        start_b: Dict[str, Dict[str, np.ndarray]] = {}
        goal_b: Dict[str, Dict[str, np.ndarray]] = {}
        for arm in ("left", "right"):
            sp, sq = _to_b(start_poses_world[arm])
            gp, gq = _to_b(goal_poses_world[arm])
            start_b[arm] = {"position": sp, "quaternion": sq}
            goal_b[arm] = {"position": gp, "quaternion": gq}

        debug_payload: Optional[dict[str, Any]] = None
        if debug_coordinate_check:
            link_names_list, arm_order = self._dual_arm_link_names_and_order()
            goals_after_curobo_frame: List[dict[str, List[float]]] = []
            for arm in arm_order:
                pos = np.asarray(goal_b[arm]["position"], dtype=np.float64).reshape(3)
                quat = np.asarray(goal_b[arm]["quaternion"], dtype=np.float64).reshape(4)
                pc, qc = self._transform_pose(pos, quat)
                goals_after_curobo_frame.append(
                    {
                        "arm": arm,
                        "position": _round_list(pc),
                        "quaternion": _round_list(qc),
                    }
                )
            mgc = getattr(self, "motion_gen_config", None)
            # 计算 base 系下的 delta，验证“world 平移 5cm”在 base 系下是否仍是小位移
            delta_b: Dict[str, float] = {}
            delta_vec_b: Dict[str, List[float]] = {}
            for arm in ("left", "right"):
                dp = np.asarray(goal_b[arm]["position"], dtype=np.float64).reshape(3) - np.asarray(
                    start_b[arm]["position"], dtype=np.float64
                ).reshape(3)
                delta_b[arm] = float(np.linalg.norm(dp))
                delta_vec_b[arm] = _round_list(dp, decimals=5)
            debug_payload = {
                "urdf_reference": (
                    "Benchmark `realman_franka_ee.urdf`: joint platform_joint parent "
                    "`dual_rm_75b_description` (底盘/underpan 几何) → child "
                    "`dual_rm_75b_description_platform_base_link`, "
                    "origin xyz=(0,-0.11663,0.271) axis z; offset_z += platform_joint 与封装一致。"
                ),
                "scene_reference": (
                    "Isaac `realman_config.py`: 场景 root / FrameTransformer 使用 "
                    "`base_link_underpan`，对应 URDF 中 `dual_rm_75b_description` 连杆；"
                    "cuRobo kinematics base_link 为 `dual_rm_75b_description_platform_base_link`。"
                ),
                "apply_robot_to_curobo_frame_transform": bool(self.apply_frame_transform),
                "root_pos_w": _round_list(root_pos_w),
                "root_quat_wxyz": _round_list(root_quat_wxyz),
                "platform_joint_value": float(platform_joint_value),
                "arm_base_offset_in_root_xyz": [float(x) for x in arm_base_offset_in_root_xyz],
                "base_link_estimated_pos_w": _round_list(base_pos_w),
                "base_link_estimated_quat_wxyz": _round_list(base_quat),
                "kinematics_link_names": [str(x) for x in link_names_list],
                "motion_gen_arm_order": list(arm_order),
                "start_in_base_link_frame": {
                    k: {"position": _round_list(v["position"]), "quaternion": _round_list(v["quaternion"])}
                    for k, v in start_b.items()
                },
                "goal_in_base_link_frame": {
                    k: {"position": _round_list(v["position"]), "quaternion": _round_list(v["quaternion"])}
                    for k, v in goal_b.items()
                },
                "delta_pos_norm_base_link_m": delta_b,
                "delta_pos_vec_base_link_m": delta_vec_b,
                "goal_for_motion_gen_after_frame_transform": goals_after_curobo_frame,
                "motion_gen_num_graph_seeds": getattr(mgc, "num_graph_seeds", None),
                "motion_gen_use_cuda_graph": getattr(mgc, "use_cuda_graph", None),
            }

        ik_out = self.ik_dual(start_b, seed_q=ik_seed_q, num_seeds=ik_num_seeds)
        if not ik_out.get("success"):
            out_fail: dict[str, Any] = {
                "status": "Fail",
                "position": None,
                "velocity": None,
                "detail": f"IK_START_FAIL:{ik_out.get('detail', '?')}",
                "ik": ik_out,
            }
            if debug_payload is not None:
                debug_payload["ik_detail"] = ik_out.get("detail")
                out_fail["_debug"] = debug_payload
            return out_fail

        q_start = np.asarray(ik_out["q"], dtype=np.float32).reshape(-1)
        out = self.plan_dual(q_start, goal_b, **plan_kwargs)
        out["ik"] = ik_out
        if debug_payload is not None:
            debug_payload["ik_detail"] = ik_out.get("detail")
            # 额外：对 goal 也跑一次 IK，判断 MotionGen 的 IK_FAIL 是否来自目标端不可达/约束不满足
            try:
                ik_goal = self.ik_dual(goal_b, seed_q=ik_out.get("q"), num_seeds=ik_num_seeds)
            except Exception as e:
                ik_goal = {"success": False, "q": None, "detail": f"IK_GOAL_EXCEPTION:{e}"}
            debug_payload["ik_goal_success"] = bool(ik_goal.get("success"))
            debug_payload["ik_goal_detail"] = ik_goal.get("detail")

            # 再细分：只移动单侧（另一侧保持起点），看看是哪只手导致 goal IK 不可行
            try:
                ik_goal_left_only = self.ik_dual(
                    {"left": goal_b["left"], "right": start_b["right"]},
                    seed_q=ik_out.get("q"),
                    num_seeds=ik_num_seeds,
                )
            except Exception as e:
                ik_goal_left_only = {"success": False, "q": None, "detail": f"IK_GOAL_LEFT_ONLY_EXCEPTION:{e}"}
            try:
                ik_goal_right_only = self.ik_dual(
                    {"left": start_b["left"], "right": goal_b["right"]},
                    seed_q=ik_out.get("q"),
                    num_seeds=ik_num_seeds,
                )
            except Exception as e:
                ik_goal_right_only = {"success": False, "q": None, "detail": f"IK_GOAL_RIGHT_ONLY_EXCEPTION:{e}"}
            debug_payload["ik_goal_left_only_success"] = bool(ik_goal_left_only.get("success"))
            debug_payload["ik_goal_left_only_detail"] = ik_goal_left_only.get("detail")
            debug_payload["ik_goal_right_only_success"] = bool(ik_goal_right_only.get("success"))
            debug_payload["ik_goal_right_only_detail"] = ik_goal_right_only.get("detail")

            # 对照：禁用碰撞的 IK（只用于诊断）
            try:
                ik_goal_left_only_nc = self.ik_dual_no_collision(
                    {"left": goal_b["left"], "right": start_b["right"]},
                    seed_q=ik_out.get("q"),
                    num_seeds=ik_num_seeds,
                )
            except Exception as e:
                ik_goal_left_only_nc = {"success": False, "q": None, "detail": f"IK_GOAL_LEFT_ONLY_NC_EXCEPTION:{e}"}
            debug_payload["ik_goal_left_only_no_collision_success"] = bool(ik_goal_left_only_nc.get("success"))
            debug_payload["ik_goal_left_only_no_collision_detail"] = ik_goal_left_only_nc.get("detail")

            # 进一步诊断：同一方向下缩小左臂位移幅度，判断是否“刚好越界/太大”
            try:
                dp_left = (
                    np.asarray(goal_b["left"]["position"], dtype=np.float64).reshape(3)
                    - np.asarray(start_b["left"]["position"], dtype=np.float64).reshape(3)
                )
                sweep_scales = [1.0, 0.6, 0.4, 0.2, 0.1]
                sweep: List[dict[str, Any]] = []
                for s in sweep_scales:
                    g_left = {
                        "position": np.asarray(start_b["left"]["position"], dtype=np.float64).reshape(3) + dp_left * float(s),
                        "quaternion": np.asarray(goal_b["left"]["quaternion"], dtype=np.float64).reshape(4),
                    }
                    r = self.ik_dual_no_collision(
                        {"left": g_left, "right": start_b["right"]},
                        seed_q=ik_out.get("q"),
                        num_seeds=ik_num_seeds,
                    )
                    sweep.append(
                        {
                            "scale": float(s),
                            "delta_norm_m": float(np.linalg.norm(dp_left * float(s))),
                            "success": bool(r.get("success")),
                            "detail": r.get("detail"),
                        }
                    )
                debug_payload["ik_left_delta_sweep_no_collision"] = sweep
            except Exception as e:
                debug_payload["ik_left_delta_sweep_no_collision"] = [{"error": str(e)}]

            debug_payload["plan_status"] = out.get("status")
            debug_payload["plan_detail"] = out.get("detail")
            lr = getattr(self, "last_result", None)
            if lr is not None:
                st = getattr(lr, "status", None)
                debug_payload["motion_gen_result_status"] = st.name if hasattr(st, "name") else str(st)
            out["_debug"] = debug_payload
        return out

    # ---------------------------------------------------------------------
    # Auto-read variants (Realman): use injected state_reader each call
    # ---------------------------------------------------------------------

    def set_default_realman_scene_world_auto(
        self,
        *,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
    ) -> None:
        """自动读取 Realman 状态并设置默认障碍（桌子+6箱）。"""
        st = self._read_realman_state()
        self.set_default_realman_scene_world(
            root_pos_w=st.root_pos_w,
            root_quat_wxyz=st.root_quat_wxyz,
            platform_joint_value=st.platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )

    def set_world_from_world_auto(
        self,
        obstacles_world: List[Dict[str, np.ndarray]],
        *,
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
    ) -> None:
        """自动读取 Realman 状态并更新障碍（World 系输入）。"""
        st = self._read_realman_state()
        self.set_world_from_world(
            obstacles_world,
            root_pos_w=st.root_pos_w,
            root_quat_wxyz=st.root_quat_wxyz,
            platform_joint_value=st.platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
        )

    def plan_one_ee_world_to_world_auto(
        self,
        *,
        start_pose_world: Dict[str, np.ndarray],
        goal_pose_world: Dict[str, np.ndarray],
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """自动读取 Realman 状态：World 系起点/终点（位姿）→ IK → 规划（单臂/单末端）。"""
        st = self._read_realman_state()
        return self.plan_one_ee_world_to_world(
            start_pose_world=start_pose_world,
            goal_pose_world=goal_pose_world,
            root_pos_w=st.root_pos_w,
            root_quat_wxyz=st.root_quat_wxyz,
            platform_joint_value=st.platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
            **kwargs,
        )

    def plan_dual_world_to_world_auto(
        self,
        *,
        start_poses_world: Dict[str, Dict[str, np.ndarray]],
        goal_poses_world: Dict[str, Dict[str, np.ndarray]],
        arm_base_offset_in_root_xyz: Tuple[float, float, float] = (0.0, -0.11663, 0.271),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """自动读取 Realman 状态：双臂 World 系起点/终点（位姿）→ IK → 规划。"""
        st = self._read_realman_state()
        return self.plan_dual_world_to_world(
            start_poses_world=start_poses_world,
            goal_poses_world=goal_poses_world,
            root_pos_w=st.root_pos_w,
            root_quat_wxyz=st.root_quat_wxyz,
            platform_joint_value=st.platform_joint_value,
            arm_base_offset_in_root_xyz=arm_base_offset_in_root_xyz,
            **kwargs,
        )

    def reset(self, reset_seed: bool = True) -> None:
        if hasattr(self.motion_gen, "reset"):
            self.motion_gen.reset(reset_seed=reset_seed)

    @staticmethod
    def plan_grippers(now_val: float, target_val: float, num_step: int = 200) -> dict[str, Any]:
        return plan_grippers_linear(now_val, target_val, num_step=num_step)

    def plan_dual(
        self,
        start_joint_positions: np.ndarray,
        goal_poses: Dict[str, Dict[str, np.ndarray]],
        max_attempts: int = 60,
        timeout: float = 10.0,
        enable_graph: bool = True,
        enable_opt: bool = True,
    ) -> dict[str, Any]:
        """
        双臂同时规划（``plan_batch``，batch=1）；当前自由度 = ``self.dof``。

        参数:
            start_joint_positions: (self.dof,)，顺序与 URDF 生成时一致
            goal_poses: ``{'left': {'position','quaternion'}, 'right': {...}}``（单臂场景不使用此接口），与 ``set_world`` 同坐标约定

        返回:
            RoboTwin 风格 dict：``status`` / ``position`` (T, self.dof) / ``velocity`` / ``detail`` 等
        """
        start_joint_positions = np.asarray(start_joint_positions, dtype=np.float32).reshape(-1)
        if start_joint_positions.shape[0] != self._dof:
            raise ValueError(
                f"期望起始关节 shape ({self._dof},)，得到 {start_joint_positions.shape}"
            )
        if "left" not in goal_poses or "right" not in goal_poses:
            raise KeyError("goal_poses 必须包含 'left' 与 'right'")

        link_names_list, order = self._dual_arm_link_names_and_order()

        positions: List[np.ndarray] = []
        quaternions: List[np.ndarray] = []
        for arm in order:
            pos = np.asarray(goal_poses[arm]["position"], dtype=np.float64).reshape(3)
            quat = np.asarray(goal_poses[arm]["quaternion"], dtype=np.float64).reshape(4)
            pc, qc = self._transform_pose(pos, quat)
            positions.append(pc.astype(np.float32))
            quaternions.append(qc.astype(np.float32))

        pos_arr = np.stack(positions, axis=0)[np.newaxis, :, :]
        quat_arr = np.stack(quaternions, axis=0)[np.newaxis, :, :]

        plan_config = MotionGenPlanConfig(
            enable_graph=enable_graph,
            enable_opt=enable_opt,
            max_attempts=max_attempts,
            timeout=timeout,
        )

        # 张量须在非 inference_mode 下创建，否则 cuRobo 内部 cost 无法 backward
        with _curobo_autograd_context():
            start_t = self.tensor_args.to_device(start_joint_positions[np.newaxis, :])
            start_state = JointState.from_position(start_t)
            positions_tensor = self.tensor_args.to_device(pos_arr)
            quaternions_tensor = self.tensor_args.to_device(quat_arr)
            goal_pose = Pose(position=positions_tensor, quaternion=quaternions_tensor)
            result = self.motion_gen.plan_batch(start_state, goal_pose, plan_config)
        self.last_result = result
        plan_dict = motion_gen_batch_result_to_plan_dict(result, batch_index=0)
        self.last_plan_dict = plan_dict
        return plan_dict

    def plan(
        self,
        start_joint_positions: np.ndarray,
        goal_poses: Dict[str, Dict[str, np.ndarray]],
        dt: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Tuple[bool, np.ndarray]]:
        """
        兼容旧 API：默认返回 **dict**。

        若 ``legacy_tuple_return=True`` 传入 kwargs（不推荐），则仍返回 ``(success, trajectory)``。
        ``dt`` 已弃用：插值步长由构造参数 ``interpolation_dt`` 决定；传入时仅触发 ``UserWarning``。
        """
        import warnings

        if dt is not None and abs(dt - self.interpolation_dt) > 1e-6:
            warnings.warn(
                "plan(..., dt=...) 已弃用；请用 CuroboPlanner(..., interpolation_dt=...) 设置插值步长。",
                UserWarning,
                stacklevel=2,
            )
        legacy = kwargs.pop("legacy_tuple_return", False)
        out = self.plan_dual(start_joint_positions, goal_poses, **kwargs)
        if legacy:
            ok = out["status"] == "Success"
            traj = out["position"] if ok else np.array([])
            return ok, traj
        return out

    def plan_one_ee(
        self,
        start_joint_positions: np.ndarray,
        goal_pose: Dict[str, np.ndarray],
        max_attempts: int = 60,
        timeout: float = 10.0,
        enable_graph: bool = True,
        enable_opt: bool = True,
    ) -> dict[str, Any]:
        """
        单末端运动规划：适用于 kinematics 中 ``link_names`` 仅含一个末端的配置（由加载的 YAML / ``robot_config`` 决定，``self.dof`` 常为 7）。

        参数:
            start_joint_positions: ``(self.dof,)``
            goal_pose: ``{'position': (3,), 'quaternion': (4,)}``（与 ``set_world`` / ``plan_dual`` 同坐标约定）
        """
        start_joint_positions = np.asarray(start_joint_positions, dtype=np.float32).reshape(-1)
        if start_joint_positions.shape[0] != self._dof:
            raise ValueError(
                f"plan_one_ee 期望起始关节 shape ({self._dof},)，得到 {start_joint_positions.shape}"
            )
        pos = np.asarray(goal_pose["position"], dtype=np.float64).reshape(3)
        quat = np.asarray(goal_pose["quaternion"], dtype=np.float64).reshape(4)
        pc, qc = self._transform_pose(pos, quat)
        pos_arr = pc.astype(np.float32).reshape(1, 1, 3)
        quat_arr = qc.astype(np.float32).reshape(1, 1, 4)

        plan_config = MotionGenPlanConfig(
            enable_graph=enable_graph,
            enable_opt=enable_opt,
            max_attempts=max_attempts,
            timeout=timeout,
        )

        with _curobo_autograd_context():
            start_t = self.tensor_args.to_device(start_joint_positions[np.newaxis, :])
            start_state = JointState.from_position(start_t)
            positions_tensor = self.tensor_args.to_device(pos_arr)
            quaternions_tensor = self.tensor_args.to_device(quat_arr)
            goal_pose_t = Pose(position=positions_tensor, quaternion=quaternions_tensor)
            result = self.motion_gen.plan_batch(start_state, goal_pose_t, plan_config)
        self.last_result = result
        plan_dict = motion_gen_batch_result_to_plan_dict(result, batch_index=0)
        self.last_plan_dict = plan_dict
        return plan_dict

    def plan_single_arm(
        self,
        start_joint_positions: np.ndarray,
        goal_pose: Dict[str, np.ndarray],
        arm: str,
        fixed_arm_goal_pose: Dict[str, np.ndarray],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        只驱动一侧手臂时：调用方提供 **完整 14 维起始关节**，以及 **固定侧末端目标位姿**
        （通常来自仿真当前 FK / body_state），避免在封装内做 FK。

        参数:
            start_joint_positions: (14,)
            goal_pose: 移动臂 ``{'position':(3,), 'quaternion':(4,)}``
            arm: ``'left'`` 或 ``'right'``
            fixed_arm_goal_pose: 另一侧末端目标（与 ``goal_pose`` 相同结构）
        """
        arm = arm.lower()
        if arm not in ("left", "right"):
            raise ValueError("arm 必须为 'left' 或 'right'")
        other = "right" if arm == "left" else "left"
        goal_poses = {
            arm: {
                "position": np.asarray(goal_pose["position"], dtype=np.float64),
                "quaternion": np.asarray(goal_pose["quaternion"], dtype=np.float64),
            },
            other: {
                "position": np.asarray(fixed_arm_goal_pose["position"], dtype=np.float64),
                "quaternion": np.asarray(fixed_arm_goal_pose["quaternion"], dtype=np.float64),
            },
        }
        return self.plan_dual(start_joint_positions, goal_poses, **kwargs)

    def get_interpolated_trajectory(self) -> Optional[np.ndarray]:
        if self.last_plan_dict is not None and self.last_plan_dict["status"] == "Success":
            return self.last_plan_dict["position"]
            return None
        
    def get_optimized_trajectory(self) -> Optional[np.ndarray]:
        if self.last_result is None:
            return None
        try:
            if not self.last_result.success[0].item():
                return None
        except Exception:
            return None
        op = getattr(self.last_result, "optimized_plan", None)
        if op is None:
            return None
        pos = op.position
        if isinstance(pos, torch.Tensor) and pos.dim() == 3:
            pos = pos[0]
        return pos.detach().float().cpu().numpy()
    
    def is_success(self) -> bool:
        return self.last_plan_dict is not None and self.last_plan_dict.get("status") == "Success"
    
    @property
    def solve_time(self) -> Optional[float]:
        if self.last_result is None:
            return None
        st = getattr(self.last_result, "solve_time", None)
        if st is None:
            return None
        if isinstance(st, torch.Tensor):
            return float(st.flatten()[0].item()) * 1000.0
        return float(st) * 1000.0
