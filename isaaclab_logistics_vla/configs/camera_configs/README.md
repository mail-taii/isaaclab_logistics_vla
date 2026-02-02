# 按机器人区分的相机配置（任务通用）

本目录与具体任务无关，**所有任务共用**：每个机器人一套相机，prim_path 必须绑定该机器人的 link 名称（如 realman 的 `head_link2`、`panda_left_hand`，UR10e 的 `ee_link`、`tool0` 等），否则相机挂载位置错误或无法渲染。

## 接入新机器人时

1. **在本目录下新增一个模块**，如 `ur10e.py`，定义该类机器人的三个相机（head_camera、ee_camera、top_camera）：
   - `head_camera`：挂在机器人“头部”或固定 link 上，prim_path 如 `{ENV_REGEX_NS}/Robot/your_head_link/camera_link/head_camera`
   - `ee_camera`：挂在末端执行器 link 上，prim_path 如 `{ENV_REGEX_NS}/Robot/your_ee_link/ee_camera`
   - `top_camera`：可固定在世界系（不绑机器人），如 `{ENV_REGEX_NS}/top_camera`

2. **在 `__init__.py` 的 `CAMERA_CONFIG_REGISTRY` 中注册**：
   ```python
   "ur10e": ur10e.UR10eCameraConfig,
   ```

3. **在 `evaluation/robot_registry.py` 中**为该机器人设置 `camera_config_key="ur10e"` 和 `scene_robot_key="ur10e"`（或对应 register 中的键）。

4. 任意任务运行评估时传 `--robot_id <你的 robot_id>`，场景会按 robot_id 自动加载该机器人的 asset 和相机绑定。

## 参考

- `realman.py`：Realman 的 head_link2、panda_left_hand 绑定。
- `evaluation/robot_registry.py`：`camera_config_key`、`scene_robot_key` 说明。
