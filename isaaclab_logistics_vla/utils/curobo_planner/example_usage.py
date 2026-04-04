"""
cuRobo 运动规划器使用示例（RoboTwin 风格：结果 dict + numpy）。

运行（需在安装 curobo 与 GPU 环境下）::

    python -m isaaclab_logistics_vla.utils.curobo_planner.example_usage
"""
from __future__ import annotations

import numpy as np

from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner


def example_basic_planning() -> tuple[bool, np.ndarray | None]:
    print("=== Basic cuRobo Planning Example (dict API) ===\n")

    print("Initializing CuroboPlanner...")
    planner = CuroboPlanner(
        urdf_path="/home/junzhe/Benchmark/robot/realman/realman_franka_ee.urdf",
        device="cuda:0",
        interpolation_dt=0.05,
    )
    print("Initialization complete!\n")

    print("Setting up obstacles...")
    obstacles = [
        {
            "name": "table",
            "position": np.array([0.0, 0.5, 0.1]),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
            "size": np.array([1.0, 0.6, 0.3]),
        },
        {
            "name": "box_1",
            "position": np.array([0.2, 0.3, 0.5]),
            "size": np.array([0.1, 0.1, 0.5]),
        },
    ]
    planner.set_world(obstacles)
    print(f"Added {len(obstacles)} obstacles\n")

    start_joints = np.array(
        [
            0.0,
            -0.6,
            0.0,
            -1.2,
            0.0,
            0.6,
            0.0,
            0.0,
            -0.6,
            0.0,
            -1.2,
            0.0,
            0.6,
            0.0,
        ],
        dtype=np.float32,
    )

    goal_poses = {
        "left": {
            "position": np.array([-0.2, 0.5, 0.3]),
            "quaternion": np.array([0.7071, 0.0, 0.7071, 0.0]),
        },
        "right": {
            "position": np.array([0.2, 0.5, 0.3]),
            "quaternion": np.array([0.7071, 0.0, 0.7071, 0.0]),
        },
    }

    print("Planning...")
    out = planner.plan_dual(start_joints, goal_poses)

    if out["status"] == "Success":
        trajectory = out["position"]
        assert trajectory is not None
        print("Planning SUCCESSFUL!")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Solve time: {planner.solve_time:.2f} ms" if planner.solve_time else "")
        print(f"First waypoint:\n{trajectory[0]}")
        print(f"Last waypoint:\n{trajectory[-1]}")
        return True, trajectory

    print(f"Planning FAILED: {out.get('detail', '')}")
    return False, None


def example_empty_world() -> None:
    print("\n=== Empty World Planning Example ===\n")

    planner = CuroboPlanner(interpolation_dt=0.05)
    planner.clear_world()

    start_joints = np.array(
        [
            0.0,
            -0.5,
            0.0,
            -1.5,
            0.0,
            0.8,
            0.0,
            0.0,
            -0.5,
            0.0,
            -1.5,
            0.0,
            0.8,
            0.0,
        ],
        dtype=np.float32,
    )
    goal_poses = {
        "left": {
            "position": np.array([-0.3, 0.6, 0.4]),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        },
        "right": {
            "position": np.array([0.3, 0.6, 0.4]),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        },
    }

    out = planner.plan(start_joints, goal_poses)
    print(f"status={out['status']}")
    if out["status"] == "Success" and out["position"] is not None:
        print(f"Trajectory length: {out['position'].shape[0]} steps")


def example_legacy_tuple() -> None:
    """旧代码 ``success, traj = planner.plan(..., legacy_tuple_return=True)``。"""
    planner = CuroboPlanner()
    planner.clear_world()
    start_joints = np.zeros(14, dtype=np.float32)
    goal_poses = {
        "left": {"position": np.array([0.0, 0.4, 0.3]), "quaternion": np.array([1.0, 0.0, 0.0, 0.0])},
        "right": {"position": np.array([0.0, 0.4, 0.3]), "quaternion": np.array([1.0, 0.0, 0.0, 0.0])},
    }
    ok, traj = planner.plan(start_joints, goal_poses, legacy_tuple_return=True)  # type: ignore[misc]
    print(f"legacy tuple: ok={ok}, traj.shape={getattr(traj, 'shape', None)}")


def example_in_benchmark() -> None:
    """在 evaluator / 策略中集成时的推荐写法（伪代码）。"""
    print("\n=== Benchmark integration (sketch) ===\n")
    print(
        """
    from isaaclab_logistics_vla.utils.curobo_planner import CuroboPlanner

    planner = CuroboPlanner(device="cuda:0", interpolation_dt=0.05)
    planner.set_world(obstacles)  # 从仿真构建 cuboid 列表

    out = planner.plan_dual(current_joints_14, goal_poses)
    if out["status"] == "Success":
        traj = out["position"]  # (T, 14) float32 numpy
        # 逐步 env.step 或写回 action buffer
    grip = CuroboPlanner.plan_grippers(0.0, 1.0, num_step=200)
    """
    )


if __name__ == "__main__":
    example_basic_planning()
    example_empty_world()
    example_legacy_tuple()
    example_in_benchmark()
