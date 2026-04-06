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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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

from .result_utils import motion_gen_batch_result_to_plan_dict, plan_grippers_linear
from .robot_spec import RobotSpec
from .world_spec import WorldSpec


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
                cache_dir = os.path.expanduser("~/.cache/curobo_realman")
                os.makedirs(cache_dir, exist_ok=True)
                # v2: 与常见 Realman+panda_hand URDF 末端命名一致（旧缓存含 left_ee 会 KeyError）
                eff_cache = os.path.join(cache_dir, "realman_config_v2.yaml")
            if eff_cache is None:
                raise ValueError(
                    "须提供 robot_config、cache_path / RobotSpec.cache_path，"
                    "或将 use_curobo_cache=True 以使用默认 ~/.cache/curobo_realman/realman_config_v2.yaml（须已预生成）。"
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

        self.last_result: Any = None
        self.last_plan_dict: Optional[dict[str, Any]] = None

        self.rotation_transform = self._get_rotation_transform()

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

        positions: List[np.ndarray] = []
        quaternions: List[np.ndarray] = []
        for arm in ("left", "right"):
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
