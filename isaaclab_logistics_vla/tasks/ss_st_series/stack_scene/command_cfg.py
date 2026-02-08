from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

from .Spawn_ss_st_stack_CommandTermCfg import Spawn_ss_st_stack_CommandTermCfg

ASSET_NAME = "robot"
BODY_NAME = "base_link_underpan"

@configclass
class Spawn_ss_st_stack_CommandsCfg:
    """Command terms for the MDP."""

    order_info = Spawn_ss_st_stack_CommandTermCfg(
        asset_name=ASSET_NAME,
        body_name=BODY_NAME,
        objects=['cracker_box', 'sugar_box', 'plastic_package', 'sf_big', 'sf_small'],  # 仅方盒类物品
        source_boxes=['s_box_1', 's_box_2', 's_box_3'],  # 3个原料箱，随机选一个
        target_boxes=['t_box_1', 't_box_2', 't_box_3'],
        num_active_skus=3,
        max_instances_per_sku=2,
        max_stack_height=4,
        distractor_mode="stack"
    )
