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

    num_active_skus: int = 3              # 本局选几种 SKU
    max_instances_per_sku: int = 2        # 每种 SKU 最多选几个
    max_stack_height: int = 4             # 单摞最大物品数量
    distractor_mode: str = "stack"        # 干扰物模式: "stack" | "scatter"
