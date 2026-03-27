from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

from .Spawn_ss_st_stack_CommandTermCfg import Spawn_ss_st_stack_CommandTermCfg
from .scene_cfg import STACK_SCENE_OBJECTS

ASSET_NAME = "robot"
BODY_NAME = "base_link_underpan"

@configclass
class Spawn_ss_st_stack_CommandsCfg:
    """Command terms for the MDP."""

    order_info = Spawn_ss_st_stack_CommandTermCfg(
        asset_name=ASSET_NAME,
        body_name=BODY_NAME,
        objects=STACK_SCENE_OBJECTS,
        source_boxes=['s_box_1', 's_box_2', 's_box_3'],
        target_boxes=['t_box_1', 't_box_2', 't_box_3'],
        max_active_skus=5,
        max_stacks=4,
        max_per_stack=4,
    )
