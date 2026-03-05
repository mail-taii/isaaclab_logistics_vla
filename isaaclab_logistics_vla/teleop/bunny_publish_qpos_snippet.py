# 将以下代码片段添加到 bunny_teleop_server 的 BimanualRobotTeleopNode.publish_periodically 中，
# 在 self.teleop_server.send_teleop_cmd(qpos, ee_pose) 之后，以便 isaaclab_logistics_vla 遥操作模块订阅。
#
# 添加位置：bunny_teleop_server/nodes/bimanual_teleop_server_node.py
# 在 publish_periodically 方法末尾、send_teleop_cmd 之后插入：

"""
# --- 供 isaaclab_logistics_vla 遥操作模块订阅（ROS2）---
if not hasattr(self, "_qpos_pub_left"):
    from std_msgs.msg import Float64MultiArray
    self._qpos_pub_left = self.create_publisher(
        Float64MultiArray, "/bunny_teleop/left_qpos", 10
    )
    self._qpos_pub_right = self.create_publisher(
        Float64MultiArray, "/bunny_teleop/right_qpos", 10
    )
msg_left = Float64MultiArray()
msg_left.data = qpos[0].tolist()
msg_right = Float64MultiArray()
msg_right.data = qpos[1].tolist()
self._qpos_pub_left.publish(msg_left)
self._qpos_pub_right.publish(msg_right)
"""
