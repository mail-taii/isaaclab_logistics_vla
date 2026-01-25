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



@configclass
class OrderSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos  = (0.96781,2.28535,0.216)
    robot.init_state.rot = (1,0,0,0)

    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    world_anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorldAnchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01), # 非常小，看不见
            visible=False,           # 或者直接设为不可见
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, 
                kinematic_enabled=True # ✅ 关键：设为运动学物体，固定不动
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), # ✅ 放在世界原点
            rot=(1.0, 0.0, 0.0, 0.0) # 无旋转
        )
    )

    e_conveyer = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/o_conveyer",
        spawn=UsdFileCfg(
            usd_path=f"/home/wst/model_files/benchmark/urdf/G.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                
                # 防止之前提到的“炸飞”问题
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1.42335),rot=(1,0,0,0)),
    )

    e_desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/o_desk",
        spawn=UsdFileCfg(
            usd_path=f"/home/wst/model_files/benchmark/urdf/F.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35625, 3.55143, -0.12113),rot=(0.5,0.5,-0.5,-0.5)),
    )

    s_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/s_box_1",                                                                                                            
        spawn=UsdFileCfg(
            usd_path=f"/home/wst/model_files/benchmark/Box.usd",
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
            usd_path=f"/home/wst/model_files/benchmark/Box.usd",
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
            usd_path=f"/home/wst/model_files/benchmark/Box.usd",
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
            usd_path=f"/home/wst/model_files/benchmark/Box.usd",
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
            usd_path=f"/home/wst/model_files/benchmark/Box.usd",
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
            usd_path=f"/home/wst/model_files/benchmark/Box.usd",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05,
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.510, 3.4429, 0.66),rot=(1, 0, 0, 0)),
    )

    o_cracker_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/o_cracker_box_1",
        spawn=UsdFileCfg(
            usd_path=f"/home/wst/isaacsim_assets/Assets/Isaac/5.0/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.58019, 1.33614, 0.86797),rot=(1, 0, 0, 0)),
    )

    o_suger_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/o_suger_box_1",
        spawn=UsdFileCfg(
            usd_path=f"/home/wst/isaacsim_assets/Assets/Isaac/5.0/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(11.58019, 1.33614, 0.86797),rot=(1, 0, 0, 0)),
    )

    o_tomato_soup_can_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/o_tomato_soup_can_1",
        spawn=UsdFileCfg(
            usd_path=f"/home/wst/isaacsim_assets/Assets/Isaac/5.0/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(11.58019, 1.33614, 0.86797),rot=(1, 0, 0, 0)),
    )