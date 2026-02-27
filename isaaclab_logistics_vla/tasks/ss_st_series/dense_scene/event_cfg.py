from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.utils.constant import ASSET_ROOT_PATH
from isaaclab_logistics_vla.utils.object_position import set_asset_relative_position

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *

import torch


@configclass
class Spawn_ss_st_dense_EventCfg:
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

    setup_trays = EventTermCfg(
        func=lambda env, env_ids: mdp.update_tray_positions(
            env, 
            env_ids, 
            # 这里的 env 是 lambda 传入的参数，而不是全局变量
            tray_or_not=env.cfg.commands.order_info.tray_or_not 
        ),
        mode="reset"
    )

    plastic_texture_randomizer = EventTermCfg(
        func=mdp.set_package_visual_texture,
        mode="startup",
        params={
            "target_asset_names": ["empty_plastic_package_1", "empty_plastic_package_2", "empty_plastic_package_3",
                                   "empty_plastic_package_4", "empty_plastic_package_5", "empty_plastic_package_6"],
            "texture_paths": [
                f"{ASSET_ROOT_PATH}/props/Collected_empty_plastic_package/textures/20260209_234859.png",
            ],
            #"words": "订单号 123456\n发往 2号订单箱子"
            
        },
    )

