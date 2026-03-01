# 按机器人区分的动作配置（任务通用）

本目录与 **configs/robot_configs**、**configs/camera_configs** 并列：机器人 asset 与 eeframe 在 robot_configs，相机在 camera_configs，**动作空间（关节名、scale、gripper 等）在此目录**，便于单独修改而不动机器人/相机配置。

- 每个机器人一套 ActionCfg（如 `RealmanFrankaEE_ActionsCfg`、`UR5eActionsCfg`），通过 `@register.add_action_configs("xxx_actionscfg")` 注册。
- 任务（如 OrderEnvCfg）在 `__post_init__` 中按 `robot_registry.action_config_key` 调用 `register.load_action_configs(key)()` 取用。

## 接入新机器人时

1. **在本目录下新增一个模块**，如 `ur10e.py`，定义该机器人的动作配置类（关节名、scale、gripper 等），并用 `@register.add_action_configs("ur10e_actionscfg")` 注册。
2. **在 `__init__.py` 中 import 该模块**，以触发注册。
3. **在 `evaluation/robot_registry.py`** 中为该机器人设置 `action_config_key="ur10e_actionscfg"`。

## 参考

- `realman.py`：Realman 双臂 + 左右夹爪 + platform。
- `ur5e.py`：UR5e 六关节臂。
- `evaluation/robot_registry.py`：`action_config_key` 说明。
