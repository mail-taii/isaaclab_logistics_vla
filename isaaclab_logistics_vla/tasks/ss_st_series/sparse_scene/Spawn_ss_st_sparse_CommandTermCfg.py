import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks.BaseOrderCommandTermCfg import OrderCommandTermCfg
from .Spawn_ss_st_sparse_CommandTerm import Spawn_ss_st_sparse_CommandTerm
from .Spawn_ss_st_sparse_CommandTerm import Spawn_ss_st_sparse_with_obstacles_CommandTerm

@configclass
class Spawn_ss_st_sparse_CommandTermCfg(OrderCommandTermCfg):
    class_type:type = Spawn_ss_st_sparse_CommandTerm

    num_active_skus: int = 3         # 本局选几种 SKU
    max_instances_per_sku: int = 2   # 每种 SKU 选几个

@configclass
class Spawn_ss_st_sparse_with_obstacles_CommandTermCfg(OrderCommandTermCfg):
    """具有障碍物场景的订单指令配置类。"""
    
    # 指向具体的逻辑实现类
    class_type: type = Spawn_ss_st_sparse_with_obstacles_CommandTerm

    # --- 任务特定逻辑配置 ---
    # 其中一排3个槽位由1个大障碍物占据
    # 剩下一排3个槽位由3个物品占据
    num_active_skus: int = 3         # 本局选 3 种 SKU
    max_instances_per_sku: int = 1   # 每种 SKU 选 1 个实例，总计 3 个物品
    
    obstacles: list[str] = MISSING 

    