from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.utils.constant import ASSET_ROOT_PATH

@configclass
class Spawn_ss_mt_sparse_EventCfg:
    # 1. 场景基础重置
    # 将所有物体拉回 SceneCfg 中定义的 init_state (通常是地下掩埋点)
    # 这会触发 CommandTerm 随后进行物体的重新采样和摆放
    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    # 2. 机器人关节重置
    reset_joints_position = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 3. 容器纹理多样性随机化 (Domain Randomization)
    randomize_box_texture = EventTermCfg(
        func=mdp.randomize_unified_visual_texture,
        mode="reset",
        params={
            # 在 SS-MT 任务中，包含 1 个源箱和 3 个目标箱
            "target_asset_names": ["s_box_1", "s_box_2", "s_box_3", "t_box_1", "t_box_2", "t_box_3"],
            
            "texture_paths": [
                f"{ASSET_ROOT_PATH}/texture/1.png",
                f"{ASSET_ROOT_PATH}/texture/2.png",
            ],
        }
    )

