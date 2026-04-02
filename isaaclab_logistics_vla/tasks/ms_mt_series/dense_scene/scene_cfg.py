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
from isaaclab_logistics_vla.utils.constant_new import *

'''
SKU_DEFINITIONS = {
    "cracker_box": (CRACKER_BOX_PARAMS['USD_PATH'],18),
    "sugar_box":   (SUGER_BOX_PARAMS['USD_PATH'],18),
    "tomato_soup_can": (TOMATO_SOUP_CAN_PARAMS['USD_PATH'],18),
    "CN_big": (CN_BIG_PARAMS['USD_PATH'],18),
    "SF_small": (SF_SMALL_PARAMS['USD_PATH'],18),
    "empty_plastic_package": (EMPTY_PLASTIC_PACKAGE_PARAMS['USD_PATH'],18),
    "SF_big": (SF_BIG_PARAMS['USD_PATH'],18),
}
'''

DENSE_SCENE_OBJECTS = ['bottle_base3','bottle_base4']

# 每种 SKU 的默认实例数（可按需调整，也可在 constant.py 每个 PARAMS 里加 STACK_COUNT 覆盖）
DEFAULT_SKU_COUNT = 3

# SKU 定义: (usd_path, count, scale)，由 STACK_SCENE_OBJECTS + constant.SKU_CONFIG 自动生成
SKU_DEFINITIONS = {}
for _sku_name in DENSE_SCENE_OBJECTS:
    _params = SKU_CONFIG[_sku_name]
    SKU_DEFINITIONS[_sku_name] = (
        _params['USD_PATH'],
        _params.get('DENSE_COUNT', DEFAULT_SKU_COUNT),
        _params.get('DENSE_SCALE', 1.0),
    )

@configclass
class Spawn_ms_mt_dense_SceneCfg(BaseOrderSceneCfg):
    
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos  = (0.96781,2.28535,0.216)
    robot.init_state.rot = (1,0,0,0)

    replicate_physics=False

    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

delta = 0
for sku_name, (usd_path, count, scale) in SKU_DEFINITIONS.items():
    for i in range(count):
        # 实例名: cracker_box_0, cracker_box_1 ...
        instance_name = f"{sku_name}_{i}"

        obj_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                scale=(scale, scale, scale),
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    sleep_threshold=0.1
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(10000+delta, 10000+delta, 0),rot=(1, 0, 0, 0)),
        )

        delta+=1
        # [关键] 动态注入到 MySceneCfg 类中
        # 这样 Isaac Lab 解析时就能看到这些属性
        setattr(Spawn_ms_mt_dense_SceneCfg, instance_name, obj_cfg)


ASSET_ROOT_PATH = os.getenv("ASSET_ROOT_PATH", "")
for i in range(6):
    tray_cfg= RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/tray_{i}",
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT_PATH}/env/Collected_Blue_Tray/SM_Crate_A08_Blue_01.usd",
            scale=(0.3, 0.35, 0.4),
            mass_props=schemas.MassPropertiesCfg(mass=1.0),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(30, 30, 0),rot=(1, 0, 0, 0)),
    )
    setattr(Spawn_ms_mt_dense_SceneCfg, f"tray_{i}", tray_cfg)