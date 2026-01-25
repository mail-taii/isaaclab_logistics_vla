from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register


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

    left_pick_object_pose = mdp.UniformPoseCommandCfg(
        asset_name=a1,
        body_name=b1,  
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(LEFT_START_POS[0], LEFT_START_POS[0]), pos_y=(LEFT_START_POS[1], LEFT_START_POS[1]), pos_z=(LEFT_START_POS[2],LEFT_START_POS[2]), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

    left_place_object_pose = mdp.UniformPoseCommandCfg(
        asset_name=a1,
        body_name=b1,  
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(LEFT_END_POS[0], LEFT_END_POS[0]), pos_y=(LEFT_END_POS[1], LEFT_END_POS[1]), pos_z=(LEFT_END_POS[2],LEFT_END_POS[2]), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


    right_pick_object_pose = mdp.UniformPoseCommandCfg(
        asset_name=a1,
        body_name=b1,  
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(RIGHT_START_POS[0], RIGHT_START_POS[0]), pos_y=(RIGHT_START_POS[1], RIGHT_START_POS[1]), pos_z=(RIGHT_START_POS[2],RIGHT_START_POS[2]), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

    right_place_object_pose = mdp.UniformPoseCommandCfg(
        asset_name=a1,
        body_name=b1,  
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(RIGHT_END_POS[0], RIGHT_END_POS[0]), pos_y=(RIGHT_END_POS[1], RIGHT_END_POS[1]), pos_z=(RIGHT_END_POS[2],RIGHT_END_POS[2]), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )