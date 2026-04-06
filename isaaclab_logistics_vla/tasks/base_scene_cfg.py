from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac

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

    # 顶视相机：用于 VLM-agent 的唯一 tool（参考 cjx 分支 topview / top_camera）
    # 注意：渲染需要 AppLauncher(enable_cameras=True) 或等价配置。
    top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 2.0, 4.0),
            rot=euler_to_quat_isaac(r=0, p=180, y=0, return_tensor=False),
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=1024,
        height=768,
        data_types=["rgb"],
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
    #         usd_path=f"{ASSET_ROOT_PATH}/urdf/F.usd",
    #         mass_props= sim_utils.MassPropertiesCfg(mass=20.0),
    #         scale=(1, 1, 1),
    #         rigid_props=schemas.RigidBodyPropertiesCfg(
    #             sleep_threshold=0.05,
    #         ),
    #
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35625, 3.55143, -0.12113),rot=(0.5,0.5,-0.5,-0.5)),
    # )

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