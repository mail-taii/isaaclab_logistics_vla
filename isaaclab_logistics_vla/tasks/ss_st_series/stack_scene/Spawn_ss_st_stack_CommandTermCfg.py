import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg
from .Spawn_ss_st_stack_CommandTerm import Spawn_ss_st_stack_CommandTerm

@configclass
class Spawn_ss_st_stack_CommandTermCfg(OrderCommandTermCfg):
    class_type: type = Spawn_ss_st_stack_CommandTerm

    max_active_skus: int = 5              # 最多选几种 SKU（实际由 assign 随机确定）
    max_stacks: int = 4                   # 最多几摞
    max_per_stack: int = 4                # 每摞最多放几个
