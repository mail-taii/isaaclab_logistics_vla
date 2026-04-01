"""
cuRobo运动规划器使用示例
展示如何调用CuroboPlanner进行双臂运动规划
"""
import numpy as np
from curobo_planner import CuroboPlanner


def example_basic_planning():
    """基础规划示例"""
    print("=== Basic cuRobo Planning Example ===\n")
    
    # 1. 创建规划器
    print("Initializing CuroboPlanner...")
    planner = CuroboPlanner(
        urdf_path="/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
        device="cuda:0"
    )
    print("Initialization complete!\n")
    
    # 2. 设置障碍物
    print("Setting up obstacles...")
    obstacles = [
        {
            # 桌子障碍物示例
            'position': np.array([0.0, 0.5, 0.1]),  # 世界坐标系 (x: 右, y: 前, z: 上)
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
            'size': np.array([1.0, 0.6, 0.3]),  # x方向长度, y方向长度, z方向高度
        },
        {
            # 一个随机障碍物
            'position': np.array([0.2, 0.3, 0.5]),
            'size': np.array([0.1, 0.1, 0.5]),
        }
    ]
    
    planner.set_world(obstacles)
    print(f"Added {len(obstacles)} obstacles\n")
    
    # 3. 设置起始关节位置
    print("Setting start joint positions...")
    # 14个关节: [左臂7个 + 右臂7个]
    start_joints = np.array([
        0.0, -0.6, 0.0, -1.2, 0.0, 0.6, 0.0,  # left arm
        0.0, -0.6, 0.0, -1.2, 0.0, 0.6, 0.0,  # right arm
    ])
    print(f"Start joints shape: {start_joints.shape}\n")
    
    # 4. 设置目标位姿
    print("Setting goal poses...")
    goal_poses = {
        'left': {
            'position': np.array([-0.2, 0.5, 0.3]),  # (x, y, z) in world frame
            'quaternion': np.array([0.7071, 0.0, 0.7071, 0.0]),  # wxyz
        },
        'right': {
            'position': np.array([0.2, 0.5, 0.3]),
            'quaternion': np.array([0.7071, 0.0, 0.7071, 0.0]),
        }
    }
    print(f"Left goal position: {goal_poses['left']['position']}")
    print(f"Right goal position: {goal_poses['right']['position']}\n")
    
    # 5. 执行规划
    print("Planning...")
    success, trajectory = planner.plan(start_joints, goal_poses, dt=0.05)
    
    if success:
        print(f"Planning SUCCESSFUL!")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Number of waypoints: {trajectory.shape[0]}")
        print(f"Solve time: {planner.solve_time:.2f} ms")
        print(f"\nFirst waypoint joints:\n{trajectory[0]}")
        print(f"\nLast waypoint joints:\n{trajectory[-1]}")
    else:
        print("Planning FAILED!")
    
    return success, trajectory


def example_empty_world():
    """无障碍物规划示例"""
    print("\n=== Empty World Planning Example ===\n")
    
    planner = CuroboPlanner()
    planner.clear_world()
    
    start_joints = np.array([
        0.0, -0.5, 0.0, -1.5, 0.0, 0.8, 0.0,
        0.0, -0.5, 0.0, -1.5, 0.0, 0.8, 0.0,
    ])
    
    goal_poses = {
        'left': {
            'position': np.array([-0.3, 0.6, 0.4]),
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
        },
        'right': {
            'position': np.array([0.3, 0.6, 0.4]),
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
        }
    }
    
    success, trajectory = planner.plan(start_joints, goal_poses)
    print(f"Success: {success}")
    if success:
        print(f"Trajectory length: {len(trajectory)} steps")
    
    return success, trajectory


def example_in_benchmark():
    """
    在你的benchmark中如何使用
    
    典型使用场景：当你需要从当前位姿规划到目标位姿时
    """
    print("\n=== Example Integration in Benchmark ===\n")
    
    # 1. 在evaluator初始化时创建规划器（只创建一次）
    from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner
    
    curobo_planner = CuroboPlanner()
    
    # 2. 在每个episode开始时，设置场景障碍物
    # 从你的环境中获取障碍物信息：
    obstacles = []
    for obs_asset in command_term.obstacle_assets:
        pos = obs_asset.data.root_pos_w[env_idx].cpu().numpy()
        quat = obs_asset.data.root_quat_w[env_idx].cpu().numpy()
        scale = obs_asset.cfg.scale  # 尺寸
        obstacles.append({
            'position': pos,
            'quaternion': quat,
            'size': scale,
        })
    
    curobo_planner.set_world(obstacles)
    
    # 3. 获取当前起始关节位置
    current_joints = robot.data.joint_pos[env_idx, :14].cpu().numpy()  # 取前14个关节（双臂7+7）
    
    # 4. 设置目标位姿
    # 你的目标计算...
    goal_poses = {
        'left': {
            'position': left_target_pos,  # (3,) numpy array
            'quaternion': left_target_quat,  # (4,) numpy array wxyz
        },
        'right': {
            'position': right_target_pos,
            'quaternion': right_target_quat,
        }
    }
    
    # 5. 规划
    success, trajectory = curobo_planner.plan(current_joints, goal_poses)
    
    if success:
        # trajectory: (T, 14) - 每一行是一个时间步的关节位置
        # 你可以遍历轨迹，逐点执行：
        for t in range(trajectory.shape[0]):
            # 填充到17维动作空间：
            # 0-6: 左臂, 7-13: 右臂, 14: left_gripper, 15: right_gripper, 16: platform
            action = np.zeros(17)
            action[0:14] = trajectory[t]
            action[14] = 0.04  # open gripper example
            action[15] = 0.04
            # 执行action...
            # env.step(action)
            
        print(f"Executed trajectory with {trajectory.shape[0]} steps")
    else:
        print("Planning failed, handle error...")
    
    print("Integration example complete")


if __name__ == "__main__":
    # 运行基础示例
    success, trajectory = example_basic_planning()
    
    # 运行空世界示例
    example_empty_world()
    
    # 打印集成示例代码框架
    print("\n" + "="*50)
    print("For benchmark integration example, see example_in_benchmark()")
