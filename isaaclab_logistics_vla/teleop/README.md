# Bunny 遥操作模块

本模块在 **不经过评估流程** 的前提下，用 [Bunny-VisionPro](https://github.com/Dingry/BunnyVisionPro)（bunny_teleop_server）的 AVP 手部遥操作数据，**直接驱动** isaaclab_logistics_vla 中与评估相同的 Isaac Lab 场景，实现实时遥操作仿真。

## 架构

- **Bench**：同时跑
  1. **bunny_teleop_server**（Vision Pro 手部 → 检测 → retargeting → 关节角），并**向 ROS2 发布**左右臂 qpos；
  2. **本模块** `scripts/run_bunny_teleop.py`：Isaac Lab 仿真 + 订阅上述 qpos → 映射为 realman 动作 → `env.step(action)`。
- **AVP**：运行 BunnyVisionPro 客户端，把手部数据发给 bench 上的 bunny_teleop_server。

## 依赖

- 与 isaaclab_logistics_vla 相同（Isaac Lab、任务与机器人配置）。
- **ROS2**：使用 Isaac Sim 内置的 `isaacsim.ros2.bridge` 与 rclpy（无需 `source /opt/ros/humble`，也勿用 pip 安装 rclpy）。启动脚本已自动启用该扩展。

## 1. Bunny 端：发布 qpos 到 ROS2

bunny_teleop_server 默认不向 ROS 发布关节角，需在 **publish_periodically** 中增加一次发布，供本模块订阅。

在 `bunny_teleop_server/nodes/bimanual_teleop_server_node.py` 的 `publish_periodically` 里，在 `self.teleop_server.send_teleop_cmd(qpos, ee_pose)` 之后添加：

```python
# 供 isaaclab_logistics_vla 遥操作模块订阅（ROS2）
if not hasattr(self, "_qpos_pub_left"):
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    self._qpos_pub_left = self.create_publisher(Float64MultiArray, "/bunny_teleop/left_qpos", 10)
    self._qpos_pub_right = self.create_publisher(Float64MultiArray, "/bunny_teleop/right_qpos", 10)
msg_left = Float64MultiArray()
msg_left.data = qpos[0].tolist()
msg_right = Float64MultiArray()
msg_right.data = qpos[1].tolist()
self._qpos_pub_left.publish(msg_left)
self._qpos_pub_right.publish(msg_right)
```

或单独写一个 ROS2 节点，订阅 Bunny 内部使用的 topic（若存在）再转发为 `/bunny_teleop/left_qpos`、`/bunny_teleop/right_qpos`，只要消息类型为 `std_msgs/Float64MultiArray` 且前 7 维为左/右臂关节角即可。

## 2. 运行遥操作

在 bench 上，先按 Bunny 文档启动 **vision server** 和 **robot server**（并确保已按上一步发布 qpos），再在 isaaclab_logistics_vla 仓库根目录执行：

```bash
conda activate env_isaaclab
./scripts/run_bunny_teleop.sh \
  --task_scene_name Spawn_ds_st_sparse_EnvCfg \
  --asset_root_path /home/junzhe/Benchmark \
  --sim_device cuda:0
```

也可直接 `python scripts/run_bunny_teleop.py ...`，若遇 `librcl_action.so` 等库加载错误，请改用上述 shell 脚本（会在 Python 启动前设置 `LD_LIBRARY_PATH`）。

- `--task_scene_name`：与 `evaluate_vla.py` 一致的任务场景名。
- `--left_topic` / `--right_topic`：若未改 Bunny 端话题名，用默认 `/bunny_teleop/left_qpos`、`/bunny_teleop/right_qpos` 即可。
- `--control_hz`：控制循环频率，默认 60。
- `--sim_device`：与 evaluate_vla 一致，指定仿真 GPU。

Ctrl+C 结束。

## 3. 动作映射说明

- Bunny 输出为 **xarm7 + Ability Hand** 的左右臂+手关节角；本模块只取每侧**前 7 维**作为臂关节。
- 映射到 **realman** 动作空间（与 `RealmanFrankaEE_ActionsCfg` 一致）：  
  左 7 + 右 7 + 左夹爪 1 + 右夹爪 1 + 平台 1 = 17 维。夹爪与平台当前为默认值（夹爪 0.5，平台 0），后续可按需从 Bunny 手部或配置扩展。

## 4. 常见问题

- **收不到数据**：确认 Bunny 端已发布上述两个话题（`ros2 topic list` / `ros2 topic echo`），且 Isaac Sim 与 Bunny 在同一 DDS 域（默认 `ROS_DOMAIN_ID=0`，跨机需一致）。
- **ModuleNotFoundError / librcl_action.so: cannot open shared object file**：启动脚本会自动设置 `ROS_DISTRO`、`RMW_IMPLEMENTATION`、`LD_LIBRARY_PATH`。若仍报错，**在激活 conda 后、运行脚本前**执行（必须在实际启动 Python 之前设置）：
  ```bash
  export ROS_DISTRO=humble
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH
  ```
  然后运行 `python scripts/run_bunny_teleop.py ...`
- **延迟大**：保证 AVP 与 bench 在同一局域网；控制循环可用 `--control_hz` 提高（如 90），并避免仿真或渲染过载。
