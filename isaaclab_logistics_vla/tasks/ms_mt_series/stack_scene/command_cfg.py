from isaaclab.utils import configclass

from .Spawn_ms_mt_stack_CommandTermCfg import Spawn_ms_mt_stack_CommandTermCfg
from .scene_cfg import STACK_SCENE_OBJECTS

ASSET_NAME = "robot"
BODY_NAME = "base_link_underpan"


@configclass
class Spawn_ms_mt_stack_CommandsCfg:
    """Command terms for the MDP."""

    order_info = Spawn_ms_mt_stack_CommandTermCfg(
        asset_name=ASSET_NAME,
        body_name=BODY_NAME,
        objects=STACK_SCENE_OBJECTS,
        source_boxes=['s_box_1', 's_box_2', 's_box_3'],
        target_boxes=['t_box_1', 't_box_2', 't_box_3'],
        min_source_box=2,
        max_source_box=3,
        min_target_orders=2,
        max_target_orders=3,
        max_skus_per_order=3,
        max_items_per_order=5,
        max_stacks=4,
        max_per_stack=4,
    )
