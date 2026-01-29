from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register

@configclass
class Spawn_ss_st_sparse_EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_joints_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 

            "position_range": (0.0, 0.0),
            
            "velocity_range": (0.0, 0.0),
        },
    )