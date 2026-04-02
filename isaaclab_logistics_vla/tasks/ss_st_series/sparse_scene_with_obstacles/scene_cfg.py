from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks.base_scene_cfg import BaseOrderSceneCfg
from isaaclab_logistics_vla.utils.constant import *

# 1. 物品定义保持不变 (对应一排的3个槽位)
SKU_DEFINITIONS = {
    "cracker_box": (CRACKER_BOX_PARAMS['USD_PATH'], 2),
    "sugar_box":   (SUGER_BOX_PARAMS['USD_PATH'], 2),
    "tomato_soup_can": (TOMATO_SOUP_CAN_PARAMS['USD_PATH'], 2),
}

@configclass
class Spawn_ss_st_sparse_with_obstacles_SceneCfg(BaseOrderSceneCfg):
    
    # 机器人配置
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos  = (0.96781, 2.28535, 0.216)
    robot.init_state.rot = (1, 0, 0, 0)

    replicate_physics = False

    # 末端执行器参考系
    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

    # --- 障碍物随机化配置 ---
    large_obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/large_obstacle",
        spawn=sim_utils.CuboidCfg(
            # 这里的 size 会在每个环境实例创建时作为初始值
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

# 2. 动态注入 SKU 物品实例
for sku_name, (usd_path, count) in SKU_DEFINITIONS.items():
    for i in range(count):
        instance_name = f"{sku_name}_{i}"
        
        obj_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                scale=(1.0, 1.0, 1.0),
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    sleep_threshold=0.05
                ),
            ),
            # 初始位置设在远处，等待 CommandTerm 采样
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -10.0), rot=(1, 0, 0, 0)),
        )
        
        # 动态注入到带后缀的类名中
        setattr(Spawn_ss_st_sparse_with_obstacles_SceneCfg, instance_name, obj_cfg)