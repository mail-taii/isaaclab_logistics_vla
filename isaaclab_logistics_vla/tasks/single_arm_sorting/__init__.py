# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single arm sorting task for logistics VLA benchmark.

This task implements task 3.1.1-1: Single arm sorting of light random small SKU packages.
"""

from .single_arm_sorting_env_cfg import SingleArmSortingEnvCfg
from .single_arm_sorting_env import SingleArmSortingEnv

__all__ = ["SingleArmSortingEnvCfg", "SingleArmSortingEnv"]

