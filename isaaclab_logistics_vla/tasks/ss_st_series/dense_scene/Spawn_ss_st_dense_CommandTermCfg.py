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
    max_instances_per_sku: int = 5   # 每种 SKU 选几个

    shape_groups: list[str] = MISSING  # Dense Scene 特有：物体形状分组列表

    obj_to_sub_container_id: int = MISSING  # Dense Scene 特有：物体对应的子容器 ID
