from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks.base_scene_cfg import BaseOrderSceneCfg
from isaaclab_logistics_vla.utils.constant import *

SKU_DEFINITIONS = {
    "cracker_box": (CRACKER_BOX_PARAMS['USD_PATH'],6),
    "sugar_box":   (SUGER_BOX_PARAMS['USD_PATH'],6),
    "tomato_soup_can": (TOMATO_SOUP_CAN_PARAMS['USD_PATH'],6),
    "CN_big": (CN_BIG_PARAMS['USD_PATH'],6),
    "SF_small": (SF_SMALL_PARAMS['USD_PATH'],6),
    "empty_plastic_package": (EMPTY_PLASTIC_PACKAGE_PARAMS['USD_PATH'],6),
    "SF_big": (SF_BIG_PARAMS['USD_PATH'],6),
}


@configclass
class Spawn_ss_st_dense_SceneCfg(BaseOrderSceneCfg):
    
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos  = (0.96781,2.28535,0.216)
    robot.init_state.rot = (1,0,0,0)

    replicate_physics=False

    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

for sku_name, (usd_path, count) in SKU_DEFINITIONS.items():
    for i in range(count):
        # 实例名: cracker_box_0, cracker_box_1 ...
        instance_name = f"{sku_name}_{i}"
        
        if sku_name == "cracker_box" or sku_name == "sugar_box" or sku_name == "tomato_soup_can":
            # 定义 Config
            obj_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
                spawn=UsdFileCfg(
                    usd_path=usd_path,
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=schemas.RigidBodyPropertiesCfg(
                        sleep_threshold=0.05
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1158019, 133614, 0),rot=(1, 0, 0, 0)),
            )
        else:
        #if sku_name == "CN_big" | sku_name == "SF_small" | sku_name == "empty_plastic_package" | sku_name == "SF_big":
            obj_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
                spawn=UsdFileCfg(
                    usd_path=usd_path,
                    scale=(0.3, 0.3, 0.3),
                    rigid_props=schemas.RigidBodyPropertiesCfg(
                        sleep_threshold=0.05
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1158019, 133614, 0),rot=(1, 0, 0, 0)),
            )
        
        # [关键] 动态注入到 MySceneCfg 类中
        # 这样 Isaac Lab 解析时就能看到这些属性
        setattr(Spawn_ss_st_dense_SceneCfg, instance_name, obj_cfg)

ASSET_ROOT_PATH = os.getenv("ASSET_ROOT_PATH", "")
for i in range(6):
    tray_cfg= RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/tray_{i}",
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT_PATH}/env/Collected_Blue_Tray/SM_Crate_A08_Blue_01.usd",
            scale=(0.3*0.018, 0.35*0.018, 0.4*0.018),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1158019, 1158019, 0),rot=(1, 0, 0, 0)),
    )
    setattr(Spawn_ss_st_dense_SceneCfg, f"tray_{i}", tray_cfg)