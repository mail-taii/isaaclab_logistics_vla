"""
按机器人区分的动作配置（任务通用）：每个机器人一套 MDP 动作空间（关节名、scale 等）。
与 camera_configs 类似，独立成包便于修改；通过 register.add_action_configs 注册，
env_cfg 按 robot_registry 的 action_config_key 用 register.load_action_configs(key)() 取用。
"""
from __future__ import annotations

# 导入各模块以触发 @register.add_action_configs 注册
from . import realman
from . import ur5e

__all__ = ["realman", "ur5e"]
