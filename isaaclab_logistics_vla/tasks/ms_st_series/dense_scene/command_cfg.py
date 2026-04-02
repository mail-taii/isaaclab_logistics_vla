from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

from .Spawn_ms_st_dense_CommandTermCfg import Spawn_ms_st_dense_CommandTermCfg
from .scene_cfg import DENSE_SCENE_OBJECTS
ASSET_NAME = "robot"
BODY_NAME = "base_link_underpan"

@configclass
class Spawn_ms_st_dense_CommandsCfg:
    """Command terms for the MDP."""

    order_info = Spawn_ms_st_dense_CommandTermCfg(
        asset_name = ASSET_NAME,
        body_name=BODY_NAME,
        objects=DENSE_SCENE_OBJECTS,
        source_boxes = ['s_box_1','s_box_2','s_box_3'],
        target_boxes = ['t_box_1','t_box_2','t_box_3'],
        num_active_skus = 3,
        max_instances_per_sku = 6
    )