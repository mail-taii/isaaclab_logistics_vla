from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
#     left_pick_object = RewTerm(func=mdp.goal_specific_ee_distance,
#                                weight=1.0,
#                                params={"std": 0.05,"command_name":'left_pick_object_pose',
#                                        "hand_name": "left_ee_tcp",
#                                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
#                                        'is_world':True})
    
#     left_place_object = RewTerm(func=mdp.goal_specific_ee_distance,
#                                 weight=1.0,
#                                params={"std": 0.05,"command_name":'left_place_object_pose',
#                                        "hand_name": "left_ee_tcp",
#                                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),'is_world':True})
    
#     right_pick_object = RewTerm(func=mdp.goal_specific_ee_distance,
#                                 weight=1.0,
#                                params={"std": 0.05,"command_name":'right_pick_object_pose',
#                                        "hand_name": "right_ee_tcp",
#                                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),'is_world':True})
    
#     right_place_object = RewTerm(func=mdp.goal_specific_ee_distance,
#                                  weight=1.0,
#                                params={"std": 0.05,"command_name":'right_place_object_pose',
#                                        "hand_name": "right_ee_tcp",
#                                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),'is_world':True})
    pass

    