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
from isaaclab_logistics_vla.utils.constant import *
from isaaclab.sensors.camera import CameraCfg
from isaaclab_logistics_vla.utils.util import *


@configclass
class BaseOrderSceneCfg(InteractiveSceneCfg):
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
        prim_path="{ENV_REGEX_NS}/e_conveyer",
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT_PATH}/env/conveyer.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                
                # 防止之前提到的“炸飞”问题
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1.42335),rot=(1,0,0,0)),
    )

    e_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/e_table",
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT_PATH}/env/table.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                
                # 防止之前提到的“炸飞”问题
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.9, 3.5, 0),rot=(1,0,0,0)),
    )

    # e_desk = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/o_desk",
    #     spawn=UsdFileCfg(
    #         usd_path=f"/home/daniel/fff/model_files/benchmark/urdf/F.usd",
    #         mass_props= sim_utils.MassPropertiesCfg(mass=20.0),
    #         scale=(1, 1, 1),
    #         rigid_props=schemas.RigidBodyPropertiesCfg(
    #             sleep_threshold=0.05, 
    #         ),
            
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35625, 3.55143, -0.12113),rot=(0.5,0.5,-0.5,-0.5)),
    # )

    # --- 机器人与参考系 ---
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos  = (0.96781, 2.28535, 0.216)
    robot.init_state.rot = (1, 0, 0, 0)

    replicate_physics = False

    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

    # 1.头部摄像头
    head_camera: CameraCfg = CameraCfg(
        # 挂载在头部最高的连杆上
        prim_path="{ENV_REGEX_NS}/Robot/head_link2/head_camera", 
        update_period=0.0,
        height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0, 0, 1.7), 
            rot=(-0.00378, -0.01704, 0.21641, 0.97615),
            convention="opengl"
        ),
    )

    # 2.左手腕摄像头
    left_wrist_camera: CameraCfg = CameraCfg(
        # 挂载在左手夹爪基座上
        prim_path="{ENV_REGEX_NS}/Robot/panda_left_hand/left_camera", 
        update_period=0.0,
        height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.36109, 0.31991, 0.7884), 
            rot=(0.31974, -0.21564, 0.25905, 0.88553),
            convention="opengl"
        ),
    )

    # 3.右手腕摄像头
    right_wrist_camera: CameraCfg = CameraCfg(
        # 挂载在右手夹爪基座上
        prim_path="{ENV_REGEX_NS}/Robot/panda_right_hand/right_camera", 
        update_period=0.0,
        height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.60966, 0.19795, 0.74869), 
            rot=(0.27194, -0.16637, -0.22036, -0.92185),
            convention="opengl"
        ),
    )

    s_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/s_box_1",                                                                                                            
        spawn=UsdFileCfg(
            usd_path=f"{WORK_BOX_PARAMS['USD_PATH']}",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.57989, 1.33474, 0.750),rot=(1, 0, 0, 0)),
    )

    s_box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/s_box_2",
        spawn=UsdFileCfg(
            usd_path=f"{WORK_BOX_PARAMS['USD_PATH']}",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.025, 1.33614, 0.725),rot=(1, 0, 0, 0)),
    )

    s_box_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/s_box_3",
        spawn=UsdFileCfg(
            usd_path=f"{WORK_BOX_PARAMS['USD_PATH']}",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51025, 1.33614, 0.750),rot=(1, 0, 0, 0)),
    )

    t_box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/t_box_1",
        spawn=UsdFileCfg(
            usd_path=f"{WORK_BOX_PARAMS['USD_PATH']}",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05,
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.57989, 3.4429, 0.82),rot=(1, 0, 0, 0)),
    )

    t_box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/t_box_2",
        spawn=UsdFileCfg(
            usd_path=f"{WORK_BOX_PARAMS['USD_PATH']}",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05,
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.025, 3.4429, 0.82),rot=(1, 0, 0, 0)),
    )

    t_box_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/t_box_3",
        spawn=UsdFileCfg(
            usd_path=f"{WORK_BOX_PARAMS['USD_PATH']}",
            mass_props= sim_utils.MassPropertiesCfg(mass=5.0),
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05,
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.510, 3.4429, 0.82),rot=(1, 0, 0, 0)),
    )