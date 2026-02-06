from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.utils.constant import ASSET_ROOT_PATH


@configclass
class Spawn_ss_st_sparse_EventCfg:
    """Configuration for events."""

    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    reset_joints_position = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 

            "position_range": (0.0, 0.0),
            
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_box_texture = EventTermCfg(
        func=mdp.randomize_unified_visual_texture, # 调用该函数
        mode="reset",
        params={
            # 1. 指定要改谁
            "target_asset_names": ["s_box_1", "s_box_2", "s_box_3"],
            
            # 2. 指定纹理图片池
            "texture_paths": [
                f"{ASSET_ROOT_PATH}/texture/1.png",
                f"{ASSET_ROOT_PATH}/texture/2.png",
            ],
        }
    )

@configclass
class Spawn_ss_st_sparse_with_obstacles_EventCfg:
    """带有障碍物场景的任务事件配置。"""

    # 基础场景重置
    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    # 机器人关节重置
    reset_joints_position = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 容器（原料箱）的视觉随机化
    randomize_box_texture = EventTermCfg(
        func=mdp.randomize_unified_visual_texture,
        mode="reset",
        params={
            # 指定所有原料箱
            "target_asset_names": ["s_box_1", "s_box_2", "s_box_3"],
            
            # 纹理图片池路径
            "texture_paths": [
                f"{ASSET_ROOT_PATH}/texture/1.png",
                f"{ASSET_ROOT_PATH}/texture/2.png",
            ],
        }
    )

    # 大障碍物的颜色或纹理也发生变化
    randomize_obstacle_texture = EventTermCfg(
        func=mdp.randomize_unified_visual_texture,
        mode="reset",
        params={
            "target_asset_names": ["large_obstacle"],
            "texture_paths": [f"{ASSET_ROOT_PATH}/texture/obstacle_wood.png"],
        }
    )

