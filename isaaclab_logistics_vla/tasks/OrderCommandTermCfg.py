import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks.OrderCommandTerm import OrderCommandTerm

@configclass
class OrderCommandTermCfg(CommandTermCfg):

    class_type:type = OrderCommandTerm
    resampling_time_range = [1e5,1e5]
    asset_name: str = MISSING

    body_name: str = MISSING

    objects: list[str] = MISSING
    source_boxes:list[str] = MISSING
    target_boxes: list[str] = MISSING    

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    debug_vis = False

    from_json: int = 2