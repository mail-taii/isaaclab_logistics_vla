"""
Bunny 遥操作数据接收端：通过 ROS2 订阅 Bunny 发布的左右臂关节角，
供 run_bunny_teleop 主循环非阻塞读取。
Bunny 端需在 publish_periodically 中发布 /bunny_teleop/left_qpos 与 /bunny_teleop/right_qpos。
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np


# 默认取前 7 维为 arm，与 xarm7 一致
ARM_DOF = 7


class BunnyQposListener:
    """
    后台线程中运行 rclpy.spin，订阅 /bunny_teleop/left_qpos 和 /bunny_teleop/right_qpos
    (std_msgs/Float64MultiArray)，将最新数据写入 _latest_left / _latest_right。
    """

    def __init__(
        self,
        left_topic: str = "/bunny_teleop/left_qpos",
        right_topic: str = "/bunny_teleop/right_qpos",
        arm_dof: int = ARM_DOF,
    ):
        self.left_topic = left_topic
        self.right_topic = right_topic
        self.arm_dof = arm_dof
        self._latest_left: Optional[np.ndarray] = None
        self._latest_right: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _cb_left(self, msg):
        with self._lock:
            self._latest_left = np.array(msg.data, dtype=np.float64)

    def _cb_right(self, msg):
        with self._lock:
            self._latest_right = np.array(msg.data, dtype=np.float64)

    def start(self):
        # 优先使用 Isaac Sim 内置的 rclpy（Python 3.11），避免因 source /opt/ros/humble 导致加载系统 Python 3.10 版本
        import sys
        try:
            import isaacsim
            _rclpy_path = os.path.join(
                os.path.dirname(isaacsim.__file__), "exts", "isaacsim.ros2.bridge", "humble", "rclpy"
            )
            if os.path.isdir(_rclpy_path) and _rclpy_path not in sys.path:
                sys.path.insert(0, _rclpy_path)
        except Exception:
            pass

        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
        except ImportError as e:
            raise ImportError(
                "Bunny teleop 需要 ROS2 与 rclpy。请确保启动脚本启用了 isaacsim.ros2.bridge"
                "（run_bunny_teleop.py 已通过 --kit_args 添加）。勿用 pip 安装 rclpy，也勿 source /opt/ros/humble。"
            ) from e

        def run_spin():
            rclpy.init(args=[])
            node = Node("bunny_qpos_listener")
            node.create_subscription(
                Float64MultiArray,
                self.left_topic,
                self._cb_left,
                10,
            )
            node.create_subscription(
                Float64MultiArray,
                self.right_topic,
                self._cb_right,
                10,
            )
            while not self._stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.05)
            node.destroy_node()
            rclpy.shutdown()

        self._thread = threading.Thread(target=run_spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_latest(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回 (left_arm_7, right_arm_7)，未收到时为 None。只取前 arm_dof 维作为臂关节。"""
        with self._lock:
            left = self._latest_left
            right = self._latest_right
        if left is not None and len(left) >= self.arm_dof:
            left = left[: self.arm_dof].copy()
        else:
            left = None
        if right is not None and len(right) >= self.arm_dof:
            right = right[: self.arm_dof].copy()
        else:
            right = None
        return left, right
