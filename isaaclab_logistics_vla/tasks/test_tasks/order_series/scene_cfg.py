import random
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.scene import InteractiveSceneCfg

from isaaclab_logistics_vla.utils.register import register

OBJ_USD_OPTIONS = [
    "/home/mail-robo/Collected_003_cracker_box/003_cracker_box.usd",
    "/home/mail-robo/Collected_004_sugar_box/004_sugar_box.usd",
    "/home/mail-robo/Collected_005_tomato_soup_can/005_tomato_soup_can.usda"
]

BOX_USD_PATH = "/home/mail-robo/Box.usd"

@configclass
class OrderSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.96781, 2.28535, 0.216)
    robot.init_state.rot = (1, 0, 0, 0)

    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    world_anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorldAnchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    )

    e_conveyer = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/o_conveyer",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/urdf/G.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, max_depenetration_velocity=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1.42335), rot=(1, 0, 0, 0)),
    )

    e_desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/o_desk",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/urdf/F.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, max_depenetration_velocity=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35625, 3.55143, -0.12113), rot=(0.5, 0.5, -0.5, -0.5)),
    )

    # -------------------------------------------------------------------------
    # 3. 箱子定义 (Source & Target)
    # -------------------------------------------------------------------------
    s_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/s_box_1",                                                                                                            
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.57989, 1.33474, 0.755),rot=(1, 0, 0, 0)),
    )
    s_box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/s_box_2",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.025, 1.33614, 0.755),rot=(1, 0, 0, 0)),
    )
    s_box_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/s_box_3",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51025, 1.33614, 0.755),rot=(1, 0, 0, 0)),
    )

    t_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/t_box_1",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05,
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.57989, 3.4429, 0.66),rot=(1, 0, 0, 0)),
    )
    t_box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/t_box_2",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05,
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.025, 3.4429, 0.66),rot=(1, 0, 0, 0)),
    )
    t_box_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/t_box_3",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05,
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.510, 3.4429, 0.66),rot=(1, 0, 0, 0)),
    )

    # -------------------------------------------------------------------------
    # 4. 随机生成 9 个物品 (带清理逻辑)
    # -------------------------------------------------------------------------
    # 使用下划线前缀，提示这些是临时变量
    for _i in range(9):
        _chosen_usd_path = random.choice(OBJ_USD_OPTIONS)
        _obj_name = f"o_item_{_i}"
        
        _obj_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{_obj_name}",
            spawn=UsdFileCfg(
                usd_path=_chosen_usd_path,
                scale=(1, 1, 1),
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    sleep_threshold=0.05,
                    max_depenetration_velocity=1.0 
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(10.0 + _i * 0.5, 1.33614, 0.86), 
                rot=(1, 0, 0, 0)
            ),
        )
        
        # 注入到类属性
        vars()[_obj_name] = _obj_cfg
    
    # 必须显式删除循环变量，防止 Isaac Lab 把它当成资源
    # 如果不删除，class 内部会残留 _i=8，导致 Unknown asset config type 报错
    del _i, _chosen_usd_path, _obj_name, _obj_cfg