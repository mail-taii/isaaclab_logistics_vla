import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg
from .Spawn_ms_st_stack_CommandTerm import Spawn_ms_st_stack_CommandTerm


@configclass
class Spawn_ms_st_stack_CommandTermCfg(OrderCommandTermCfg):
    class_type: type = Spawn_ms_st_stack_CommandTerm

    # 多源参数
    num_active_skus: int = 3           # 本局选几种 SKU（含 1 种干扰物）
    max_instances_per_sku: int = 3     # 每种 SKU 最多选几个实例
    min_source_box: int = 2            # 最少使用几个原料箱
    max_source_box: int = 3            # 最多使用几个原料箱
    max_redundant_ratio: float = 0.7

    # 堆叠参数
    max_stacks: int = 4                # 每个原料箱最多几摞
    max_per_stack: int = 4             # 每摞最多放几个
