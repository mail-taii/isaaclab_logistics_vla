import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg
from .Spawn_ss_mt_stack_CommandTerm import Spawn_ss_mt_stack_CommandTerm


@configclass
class Spawn_ss_mt_stack_CommandTermCfg(OrderCommandTermCfg):
    class_type: type = Spawn_ss_mt_stack_CommandTerm

    # 多目标（订单）参数
    min_target_orders: int = 2             # 最少几笔订单
    max_target_orders: int = 3             # 最多几笔订单
    max_skus_per_order: int = 3            # 每笔订单最多选几种 SKU
    max_items_per_order: int = 5           # 每笔订单最多需要几件物品

    # 堆叠参数
    max_stacks: int = 4                    # 原料箱中最多几摞
    max_per_stack: int = 4                 # 每摞最多放几个
    max_redundant_ratio: float = 0.7       # 冗余物品采样概率上界
