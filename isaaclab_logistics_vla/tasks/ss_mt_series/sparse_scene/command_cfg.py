from isaaclab.utils import configclass
from .Spawn_ss_mt_sparse_CommandTermCfg import Spawn_ss_mt_sparse_CommandTermCfg


ASSET_NAME = "robot"

BODY_NAME = "base_link_underpan"

@configclass
class Spawn_ss_mt_sparse_CommandsCfg:

    order_info = Spawn_ss_mt_sparse_CommandTermCfg(
        asset_name=ASSET_NAME,
        body_name=BODY_NAME,
        objects=['cracker_box', 'sugar_box', 'tomato_soup_can'],
        source_boxes=['s_box_1', 's_box_2', 's_box_3'],     
        target_boxes=['t_box_1', 't_box_2', 't_box_3'],   
        num_active_skus=3,         
        max_instances_per_sku=2,
    )