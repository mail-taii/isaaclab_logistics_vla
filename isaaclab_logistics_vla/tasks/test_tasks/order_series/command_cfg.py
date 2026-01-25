from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

from isaaclab_logistics_vla.tasks.test_tasks.order_series.OrderCommandCfg import OrderCommandCfg

ASSET_NAME = "robot"
BODY_NAME = "base_link_underpan"

a1 = 'world_anchor'
b1 = 'WorldAnchor'

LEFT_START_POS =  [-1.35,-0.125,1.0] 
LEFT_END_POS = [-1.17,-0.125,1.09] 
RIGHT_START_POS = [-1.35,0.46,1.0] 
RIGHT_END_POS = [-1.17, 0.46, 1.09]

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    order_info = OrderCommandCfg(
        asset_name = ASSET_NAME,
        body_name=BODY_NAME,
        objects=['o_cracker_box_1','o_suger_box_1','o_tomato_soup_can_1'],
        source_boxes = ['s_box_1','s_box_2','s_box_3'],
        target_boxes = ['t_box_1','t_box_2','t_box_3'],
    )

