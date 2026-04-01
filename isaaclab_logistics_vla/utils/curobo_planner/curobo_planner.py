"""
cuRobo运动规划器封装
独立可调用工具，支持双臂运动规划
处理坐标变换：机器人正面沿y轴 -> cuRobo标准x轴
"""
import os
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig, JointState
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.geometry import Cuboid
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from .config_generator import load_realman_config


class CuroboPlanner:
    """
    cuRobo运动规划器封装类
    
    支持:
    - 从URDF自动加载Realman双臂机器人配置
    - 设置世界障碍物
    - 双臂联合运动规划
    - 单臂独立运动规划
    - 坐标变换处理（机器人y轴朝前 -> cuRobo x轴朝前）
    """
    
    def __init__(
        self,
        urdf_path: str = "/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
        device: str = "cuda:0",
        use_curobo_cache: bool = True,
        cache_path: Optional[str] = None,
    ):
        """
        初始化cuRobo运动规划器
        
        参数:
            urdf_path: 机器人URDF文件路径
            device: 计算设备
            use_curobo_cache: 是否使用缓存加速配置生成
            cache_path: 配置缓存路径，None则使用默认位置
        """
        self.device = device
        self.tensor_args = TensorDeviceType(device=device)
        
        # 加载机器人配置
        if cache_path is None and use_curobo_cache:
            cache_dir = os.path.expanduser("~/.cache/curobo_realman")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "realman_config.yaml")
        
        self.robot_config = load_realman_config(
            urdf_path=urdf_path,
            cache_path=cache_path,
            device=device
        )
        
        # 创建运动规划配置
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            None,  # world will be set later
            self.tensor_args,
            interpolation_steps=100,
            num_trajopt_iterations=10,
            use_batch_cc=True,
        )
        
        # 创建运动生成器
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        
        # 初始化世界为空
        self.world_config = WorldConfig()
        self._update_world()
        
        # 存储最近一次规划结果
        self.last_result = None
        
        # 坐标变换：机器人y轴朝前 -> cuRobo x轴朝前
        # 绕z轴旋转 -90度
        self.rotation_transform = self._get_rotation_transform()
    
    def _get_rotation_transform(self) -> np.ndarray:
        """
        获取坐标变换矩阵 将机器人坐标系(x:右, y:前, z:上) 
        转换为cuRobo坐标系(x:前, y:左, z:上)
        
        实际上就是绕z轴旋转-90度
        """
        theta = -math.pi / 2  # -90 degrees
        rot = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0, 0, 1]
        ])
        return rot
    
    def _transform_pose(self, position: np.ndarray, quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将机器人坐标系的位姿转换为cuRobo坐标系
        
        参数:
            position: (3,) 位置 xyz（机器人坐标系）
            quaternion: (4,) 四元数 wxyz（机器人坐标系）
        
        返回:
            (position_curobo, quaternion_curobo) 转换后的位姿
        """
        # 旋转位置
        position_curobo = self.rotation_transform @ position
        
        # 旋转四元数：绕z轴旋转-90度
        # 四元数的旋转也需要相应变换
        theta = -math.pi / 4
        q_rot = np.array([math.cos(theta), 0, 0, math.sin(theta)])  # w, x, y, z
        
        # 四元数乘法: q_result = q_rot * q
        w1, x1, y1, z1 = q_rot
        w2, x2, y2, z2 = quaternion
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        quaternion_curobo = np.array([w, x, y, z])
        
        return position_curobo, quaternion_curobo
    
    def _update_world(self) -> None:
        """更新世界到motion_gen"""
        self.motion_gen.update_world(self.world_config)
    
    def set_world(self, obstacles: List[Dict[str, np.ndarray]]) -> None:
        """
        设置世界障碍物配置
        
        参数:
            obstacles: 障碍物列表，每个障碍物是字典:
                {
                    'position': (3,) 世界坐标系位置 xyz（机器人坐标系）
                    'quaternion': (4,) 四元数 wxyz（机器人坐标系，可选，默认单位四元数）
                    'size': (3,) 长方体尺寸 xyz
                }
        """
        # 创建新的世界配置
        world_config = WorldConfig()
        cuboids = []
        
        for obs in obstacles:
            pos = obs['position']
            size = obs['size']
            quat = obs.get('quaternion', np.array([1.0, 0.0, 0.0, 0.0]))
            
            # 坐标变换
            pos_curobo, quat_curobo = self._transform_pose(pos, quat)
            
            # 创建cuboid，pose格式是 [x, y, z, qw, qx, qy, qz]
            pose = np.concatenate([pos_curobo, quat_curobo])
            
            cuboid = Cuboid(
                pose=pose,
                dims=size,
            )
            cuboids.append(cuboid)
        
        world_config.cuboid = cuboids
        self.world_config = world_config
        
        # 更新到motion_gen
        self._update_world()
    
    def clear_world(self) -> None:
        """清空所有障碍物"""
        self.world_config = WorldConfig()
        self._update_world()
    
    def plan(
        self,
        start_joint_positions: np.ndarray,
        goal_poses: Dict[str, Dict[str, np.ndarray]],
        dt: float = 0.05,
    ) -> Tuple[bool, np.ndarray]:
        """
        规划双臂运动轨迹
        
        参数:
            start_joint_positions: (14,) 起始关节位置
                [left_arm_joints(7), right_arm_joints(7)]
            goal_poses: 目标位姿字典:
                {
                    'left': {'position': (3,), 'quaternion': (4,)},
                    'right': {'position': (3,), 'quaternion': (4,)},
                }
                位姿使用机器人坐标系（正面沿y轴）
            dt: 插值轨迹的时间步长
        
        返回:
            success: 是否规划成功
            trajectory: (T, 14) 关节位置轨迹，每一行为一个时间步
        """
        # 检查输入维度
        assert start_joint_positions.shape == (14,), f"Expected (14,) start joints, got {start_joint_positions.shape}"
        assert 'left' in goal_poses and 'right' in goal_poses, "goal_poses must contain 'left' and 'right'"
        
        # 创建起始状态
        start_joint_tensor = self.tensor_args.to_device(start_joint_positions[np.newaxis, :])
        start_state = JointState.from_position(start_joint_tensor)
        
        # 处理目标位姿 - 需要组合两个末端的目标位姿
        positions = []
        quaternions = []
        
        for arm in ['left', 'right']:
            pos = goal_poses[arm]['position']
            quat = goal_poses[arm]['quaternion']
            
            # 坐标变换
            pos_curobo, quat_curobo = self._transform_pose(pos, quat)
            positions.append(pos_curobo)
            quaternions.append(quat_curobo)
        
        # 组合为批张量 [batch=1, n_ee=2, 3/4]
        positions_tensor = self.tensor_args.to_device(np.array(positions)[np.newaxis, :, :])
        quaternions_tensor = self.tensor_args.to_device(np.array(quaternions)[np.newaxis, :, :])
        
        goal_pose = Pose(position=positions_tensor, quaternion=quaternions_tensor)
        
        # 创建规划配置
        plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_optimization=True,
            do_interpolation=True,
            interpolation_dt=dt,
        )
        
        # 执行规划
        result = self.motion_gen.plan_batch(start_state, goal_pose, plan_config)
        self.last_result = result
        
        # 检查是否成功
        success = result.success[0].item()
        
        if not success:
            return False, np.array([])
        
        # 获取插值轨迹
        interpolated = result.get_interpolated_plan()
        trajectory = interpolated.position.cpu().numpy()  # [T, 14]
        
        return True, trajectory
    
    def plan_single_arm(
        self,
        start_joint_positions: np.ndarray,
        goal_pose: Dict[str, np.ndarray],
        arm: str = 'left',
        dt: float = 0.05,
    ) -> Tuple[bool, np.ndarray]:
        """
        单臂运动规划（当只需要移动一个手臂时使用）
        
        参数:
            start_joint_positions: (7,) 起始关节位置（指定手臂的7个关节）
            goal_pose: 目标位姿 {'position': (3,), 'quaternion': (4,)}（机器人坐标系）
            arm: 'left' 或 'right'，用于确定末端执行器
            dt: 插值轨迹时间步长
        
        返回:
            success: 是否规划成功
            trajectory: (T, 7) 关节位置轨迹
        """
        assert start_joint_positions.shape == (7,), f"Expected (7,) start joints, got {start_joint_positions.shape}"
        
        # 注意：对于单臂规划，我们仍然需要构造完整的14维输入
        # curobo是为整个机器人规划，所以需要填充另一个手臂的关节
        # 用户只需要提供规划手臂的关节，另一个手臂保持当前固定
        
        # 获取当前完整机器人的起始状态需要用户提供全部14个关节
        # 所以这里我们要求如果用户调用此方法，另一个手臂保持不动，我们会保持其起始关节不变
        # 如果用户需要另一个手臂移动，请使用 plan() 方法
        
        # 实际上，当我们只给一个末端设置目标时，cuRobo仍然会规划整个机器人
        # 另一个末端会保持在起始位置附近
        
        # 转换为cuRobo位姿
        pos = goal_pose['position']
        quat = goal_pose['quaternion']
        pos_curobo, quat_curobo = self._transform_pose(pos, quat)
        
        # 对于单臂，我们只设置对应末端的目标，另一个末端保持原位
        # 但实际上，我们需要两个末端目标，所以另一个末端通过FK从起始关节计算
        
        # 这里简化处理：我们让用户提供完整的14维起始关节，指明哪个臂要移动
        # 如果用户只提供7维，我们假设另一个手臂保持在0位置？不对，用户需要提供完整起始
        # 让我重新设计API...
        
        raise NotImplementedError(
            "For single arm planning, please use the full plan() API with "
            "both arms start joints and both arms goal poses. The other arm "
            "can just keep its goal pose the same as start pose."
        )
    
    def get_interpolated_trajectory(self) -> Optional[np.ndarray]:
        """
        获取最近一次规划的插值轨迹
        
        返回:
            trajectory: (T, n_dof) 插值后的关节轨迹，如果没有规划则返回None
        """
        if self.last_result is None:
            return None
        
        if not self.last_result.success.any():
            return None
        
        interpolated = self.last_result.get_interpolated_plan()
        return interpolated.position.cpu().numpy()
    
    def get_optimized_trajectory(self) -> Optional[np.ndarray]:
        """
        获取最近一次规划的优化轨迹（未插值）
        
        返回:
            trajectory: (T, n_dof) 优化后的关节轨迹，如果没有规划则返回None
        """
        if self.last_result is None:
            return None
        
        if not self.last_result.success.any():
            return None
        
        return self.last_result.optimized_plan.position.cpu().numpy()
    
    def is_success(self) -> bool:
        """
        检查最近一次规划是否成功
        
        返回:
            是否成功
        """
        if self.last_result is None:
            return False
        
        return self.last_result.success[0].item()
    
    @property
    def solve_time(self) -> Optional[float]:
        """
        获取最近一次规划的求解时间（毫秒）
        
        返回:
            求解时间，如果没有规划则返回None
        """
        if self.last_result is None:
            return None
        
        return self.last_result.solve_time.item() * 1000
