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

# -----------------------------------------------------------------------
# 在这里声明本场景使用的 SKU 列表。
# command_cfg.py 会从这里导入，不要反向引用 command_cfg。
# -----------------------------------------------------------------------
STACK_SCENE_OBJECTS = ['cracker_box', 'sugar_box', 'plastic_package',  'sf_small',
                        'rubikscube_base0','rubikscube_base1','rubikscube_base2',
                        'phone_base0','phone_base1','phone_base2','phone_base3','phone_base4',
                        'remotecontrol_base0','remotecontrol_base1','remotecontrol_base2','remotecontrol_base3','remotecontrol_base4','remotecontrol_base5','remotecontrol_base6',
                         'playingcards_base0','playingcards_base1','playingcards_base2',
                          'notebook_base0','notebook_base1','notebook_base2',
                          'soap_base1',
                          'teabox_base0','teabox_base1','teabox_base2','teabox_base3','teabox_base4','teabox_base5',
                          'coffeebox_base0','coffeebox_base1','coffeebox_base2','coffeebox_base3','coffeebox_base4','coffeebox_base5',
                          'smallspeaker_base1',
                          'woodenblock_base0',
                    ]

# 每种 SKU 的默认实例数（可按需调整，也可在 constant.py 每个 PARAMS 里加 STACK_COUNT 覆盖）
DEFAULT_SKU_COUNT = 4

# SKU 定义: (usd_path, count, scale)，由 STACK_SCENE_OBJECTS + constant.SKU_CONFIG 自动生成
SKU_DEFINITIONS = {}
for _sku_name in STACK_SCENE_OBJECTS:
    _params = SKU_CONFIG[_sku_name]
    SKU_DEFINITIONS[_sku_name] = (
        _params['USD_PATH'],
        _params.get('STACK_COUNT', DEFAULT_SKU_COUNT),
        _params.get('STACK_SCALE', 1.0),
    )


@configclass
class Spawn_ms_st_stack_SceneCfg(BaseOrderSceneCfg):
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.96781, 2.28535, 0.216)
    robot.init_state.rot = (1, 0, 0, 0)

    replicate_physics = False

    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()


# 动态注入 SKU 实例
for sku_name, (usd_path, count, scale) in SKU_DEFINITIONS.items():
    for i in range(count):
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(100, 100, 0), rot=(1, 0, 0, 0)),
        )

        setattr(Spawn_ms_st_stack_SceneCfg, instance_name, obj_cfg)
