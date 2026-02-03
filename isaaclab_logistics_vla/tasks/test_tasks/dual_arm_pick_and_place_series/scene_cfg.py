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
class DualArmPickAndPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
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

    shelf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/shelf",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/test_1_5.usd",
            scale=(0.001, 0.001, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                
                # 防止之前提到的“炸飞”问题
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0),rot=(0.5, 0.5, -0.5, -0.5)),
    )

    box_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box_1",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5335, -0.12, 0.84),rot=(0.70711, 0, 0, 0.70711)),
    )

    box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box_2",
        spawn=UsdFileCfg(
            usd_path=f"/home/daniel/fff/model_files/benchmark/Box.usd",
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05, 
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.5335, 0.497, 0.84),rot=(0.70711, 0, 0, 0.70711)),
    )