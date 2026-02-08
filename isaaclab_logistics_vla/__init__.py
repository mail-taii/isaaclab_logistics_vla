# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing logistics VLA benchmark tasks for robot learning."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_LOGISTICS_VLA_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_LOGISTICS_VLA_METADATA = toml.load(
    os.path.join(ISAACLAB_LOGISTICS_VLA_EXT_DIR, "config", "extension.toml")
)
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_LOGISTICS_VLA_METADATA["package"]["version"]

##
# Register Gym environments.
##

from .utils import import_packages

# Import all configs in this package
# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)

import isaaclab_logistics_vla.configs.robot_configs.realman_config
import isaaclab_logistics_vla.configs.robot_configs.ur5e_config
import isaaclab_logistics_vla.configs.action_configs  # 动作配置独立成包，与 camera_configs 一致
