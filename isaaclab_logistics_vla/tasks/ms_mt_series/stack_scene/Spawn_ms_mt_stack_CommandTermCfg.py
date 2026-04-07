import math

from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg
from .Spawn_ms_mt_stack_CommandTerm import Spawn_ms_mt_stack_CommandTerm


@configclass
class Spawn_ms_mt_stack_CommandTermCfg(OrderCommandTermCfg):
    class_type: type = Spawn_ms_mt_stack_CommandTerm

    # 多源参数（复用 ms_st 风格）
    min_source_box: int = 2            # 最少使用几个原料箱
    max_source_box: int = 3            # 最多使用几个原料箱

    # 多目标（订单）参数（复用 ss_mt 风格）
    min_target_orders: int = 2         # 最少几笔订单
    max_target_orders: int = 3         # 最多几笔订单
    max_skus_per_order: int = 3        # 每笔订单最多选几种 SKU
    max_items_per_order: int = 5       # 每笔订单最多需要几件物品

    # 堆叠与冗余
    max_stacks: int = 4                # 每个原料箱最多几摞
    max_per_stack: int = 4             # 每摞最多放几个
    max_redundant_ratio: float = 0.7   # 冗余物品采样概率上界
