import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks.OrderCommandTermCfg import OrderCommandTermCfg
from .Spawn_ss_st_dense_CommandTerm import Spawn_ss_st_dense_CommandTerm

@configclass
class Spawn_ss_st_dense_CommandTermCfg(OrderCommandTermCfg):
    class_type:type = Spawn_ss_st_dense_CommandTerm

    num_active_skus: int = 3         # 本局选几种 SKU
    max_instances_per_sku: int = 6  # 每种 SKU 选几个

    #shape_groups: list[str] = MISSING  # Dense Scene 特有：物体形状分组列表

    #bj_to_tray_id:   # Dense Scene 特有：物体对应的子容器 ID 1/2分别对应两个子容器（托盘/非托盘），-1 表示不分配
    tray_or_not: list[bool] = [1,0,0] # 是否是托盘场景（影响目标分配逻辑）
