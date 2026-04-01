"""
从URDF生成cuRobo机器人配置
"""
import os
from typing import Optional, Tuple
import yaml

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.util.curobo_robot_world import CuroboRobotWorld


def generate_robot_config_from_urdf(
    urdf_path: str,
    output_path: Optional[str] = None,
    base_link: str = "base_link",
    left_ee_link: str = "left_ee",
    right_ee_link: str = "right_ee",
    tensor_args: TensorDeviceType = None,
) -> RobotConfig:
    """
    从URDF生成cuRobo机器人配置
    
    参数:
        urdf_path: URDF文件路径
        output_path: 输出配置文件路径(可选)
        base_link: 基座链接名称
        left_ee_link: 左臂末端执行器链接名称
        right_ee_link: 右臂末端执行器链接名称
        tensor_args: 张量设备类型
    
    返回:
        RobotConfig对象
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    
    # 使用cuRobo从URDF创建机器人配置
    # 对于双臂机器人，我们需要获取所有关节和两个末端连杆
    robot_config = RobotConfig.from_basic(
        urdf_path,
        base_link,
        [left_ee_link, right_ee_link],
        tensor_args,
        self_collision_check=True,
    )
    
    # 如果提供了输出路径，保存为YAML
    if output_path is not None:
        # 将配置转换为字典并保存
        config_dict = robot_config.model_dump()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    return robot_config


def load_realman_config(
    urdf_path: str = "/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
    cache_path: Optional[str] = None,
    device: str = "cuda:0",
) -> RobotConfig:
    """
    加载Realman双臂机器人配置
    
    参数:
        urdf_path: URDF文件路径
        cache_path: 缓存配置路径，如果存在则直接加载，否则生成后保存
        device: 计算设备
    
    返回:
        RobotConfig对象
    """
    tensor_args = TensorDeviceType(device=device)
    
    # 如果缓存存在，直接加载
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        robot_config = RobotConfig.from_dict(config_dict, tensor_args)
        return robot_config
    
    # 否则从URDF生成
    robot_config = generate_robot_config_from_urdf(
        urdf_path=urdf_path,
        output_path=cache_path,
        base_link="base_link",
        left_ee_link="left_ee",
        right_ee_link="right_ee",
        tensor_args=tensor_args
    )
    
    return robot_config


if __name__ == "__main__":
    # 测试生成Realman配置
    cache_dir = os.path.expanduser("~/.cache/curobo_realman")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "realman_config.yaml")
    
    print(f"Generating Realman robot config from URDF...")
    robot_config = load_realman_config(
        urdf_path="/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
        cache_path=cache_path
    )
    
    print(f"Config generated successfully!")
    print(f"Saved to: {cache_path}")
    print(f"Number of joints: {len(robot_config.kinematics.joint_limits)}")
