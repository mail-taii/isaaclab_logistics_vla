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

STACK_SCENE_OBJECTS = ['whiteboard_eraser_base0']

# 每种 SKU 的默认实例数（可按需调整，也可在 constant.py 每个 PARAMS 里加 STACK_COUNT 覆盖）
DEFAULT_SKU_COUNT = 2

# SKU 定义: (usd_path, count, scale)，由 STACK_SCENE_OBJECTS + constant.SKU_CONFIG 自动生成
SKU_DEFINITIONS = {}
for _sku_name in STACK_SCENE_OBJECTS:
    _params = SKU_CONFIG[_sku_name]
    SKU_DEFINITIONS[_sku_name] = (
        _params['USD_PATH'],
        _params.get('SPARSE_COUNT', DEFAULT_SKU_COUNT),
        _params.get('SPARSE_SCALE', 1.0),
    )

@configclass
class Spawn_ss_st_sparse_SceneCfg(BaseOrderSceneCfg):

    large_obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/large_obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.15, 0.30), 
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1) # 默认红色
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.5,
                angular_damping=0.5,
                max_depenetration_velocity=0.5
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )


# --- 动态注入 SKU 物品实例 ---
for sku_name, (usd_path, count, scale) in SKU_DEFINITIONS.items():
    for i in range(count):
        # 实例名: cracker_box_0, cracker_box_1 ...
        instance_name = f"{sku_name}_{i}"
        
        # 定义 Config
        obj_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                scale=(scale, scale, scale), 
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    sleep_threshold=0.05
                ),
            ),
            # 同样将所有 SKU 默认放在地下，等待 _assign_objects_boxes 的分配和召唤
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -100.0), rot=(1, 0, 0, 0)),
        )
        
        # 动态注入到唯一的 MySceneCfg 类中
        setattr(Spawn_ss_st_sparse_SceneCfg, instance_name, obj_cfg)