from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from typing import ClassVar

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks.base_scene_cfg import BaseOrderSceneCfg
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.configs.camera_configs import get_camera_config

SKU_DEFINITIONS = {
    "cracker_box": (CRACKER_BOX_PARAMS['USD_PATH'],2),
    "sugar_box":   (SUGER_BOX_PARAMS['USD_PATH'],2),
    "tomato_soup_can": (TOMATO_SOUP_CAN_PARAMS['USD_PATH'],2),
}


@configclass
class Spawn_ss_st_sparse_SceneCfg(BaseOrderSceneCfg):
    """Sparse scene 任务场景：机器人 + 三路相机（head/ee/top）+ 动态 SKU 物体。"""
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos  = (0.96781,2.28535,0.216)
    robot.init_state.rot = (1,0,0,0)

    replicate_physics=False

    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

    # 相机：与 order_series 一致，从 realman 相机配置获取（绑定 head_link2、panda_left_hand、顶视）
    _cameras: ClassVar = get_camera_config("realman")
    head_camera = _cameras.head_camera
    ee_camera = _cameras.ee_camera
    top_camera = _cameras.top_camera

for sku_name, (usd_path, count) in SKU_DEFINITIONS.items():
    for i in range(count):
        # 实例名: cracker_box_0, cracker_box_1 ...
        instance_name = f"{sku_name}_{i}"
        
        # 定义 Config
        obj_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                scale=(1.0, 1.0, 1.0),
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    sleep_threshold=0.05
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(100, 100, 0),rot=(1, 0, 0, 0)),
        )
        
        # [关键] 动态注入到 MySceneCfg 类中
        # 这样 Isaac Lab 解析时就能看到这些属性
        setattr(Spawn_ss_st_sparse_SceneCfg, instance_name, obj_cfg)