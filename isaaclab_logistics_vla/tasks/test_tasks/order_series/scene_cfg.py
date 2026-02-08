from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from typing import ClassVar

from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.configs.camera_configs import get_camera_config
from isaaclab_logistics_vla.evaluation.robot_registry import get_robot_eval_config


def get_order_scene_cfg(robot_id: str, num_envs: int = 1, env_spacing: float = 5.0) -> "OrderSceneCfg":
    """
    根据 robot_id 返回该机器人对应的场景配置**实例**（含机器人 asset 与相机绑定）。
    在 OrderSceneCfg 实例上仅替换 robot、ee_frame、三路相机，保证 InteractiveScene 迭代 __dict__ 时
    实体顺序正确（先 articulation 再 sensors），且 prim 父节点先于子节点存在。
    """
    cfg = get_robot_eval_config(robot_id)
    robot_key = cfg.scene_robot_key or "realman_franka_ee"
    camera_key = cfg.camera_config_key
    cameras = get_camera_config(camera_key)

    # 使用完整 OrderSceneCfg 实例，只替换机器人与相机相关字段
    scene_cfg = OrderSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    scene_cfg.robot = register.load_robot(robot_key)().replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.robot.init_state.pos = getattr(cfg, "init_pos", None) or (0.96781, 2.28535, 0.216)
    scene_cfg.robot.init_state.rot = getattr(cfg, "init_rot", None) or (1, 0, 0, 0)
    scene_cfg.ee_frame = register.load_eeframe_configs(f"{robot_key}_eeframe")()
    scene_cfg.head_camera = cameras.head_camera
    scene_cfg.ee_camera = cameras.ee_camera
    scene_cfg.top_camera = cameras.top_camera
    return scene_cfg


@configclass
class OrderSceneCfg(InteractiveSceneCfg):
    """Order 任务默认场景（Realman）：机器人 + 相机 + 场景物体。其他机器人请通过 get_order_scene_cfg(robot_id) 获取。"""
    robot: ArticulationCfg = register.load_robot("realman_franka_ee")().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.96781, 2.28535, 0.216)
    robot.init_state.rot = (1, 0, 0, 0)
    ee_frame: FrameTransformerCfg = register.load_eeframe_configs("realman_franka_ee_eeframe")()

    # 相机：从按机器人区分的相机注册表获取（realman 的 link 绑定）
    _cameras: ClassVar = get_camera_config("realman")  # ClassVar：避免被 InteractiveScene 当作实体解析
    head_camera = _cameras.head_camera
    ee_camera = _cameras.ee_camera
    top_camera = _cameras.top_camera

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
            usd_path="/home/junzhe/scene/G.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/F.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Collected_003_cracker_box/003_cracker_box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Collected_004_sugar_box/004_sugar_box.usd",  # 使用本地场景资源
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
            usd_path="/home/junzhe/scene/Collected_005_tomato_soup_can/005_tomato_soup_can.usda",  # 使用本地场景资源（注意是.usda格式）
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05
            ),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(11.58019, 1.33614, 0.86797),rot=(1, 0, 0, 0)),
    )

    o_cracker_box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/o_cracker_box_2",
        spawn=UsdFileCfg(
            usd_path="/home/junzhe/scene/Collected_003_cracker_box/003_cracker_box.usd",
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 1.33614, 0.86797), rot=(1, 0, 0, 0)),
    )

    o_suger_box_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/o_suger_box_2",
        spawn=UsdFileCfg(
            usd_path="/home/junzhe/scene/Collected_004_sugar_box/004_sugar_box.usd",
            scale=(1, 1, 1),
            rigid_props=schemas.RigidBodyPropertiesCfg(
                sleep_threshold=0.05
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.5, 1.33614, 0.86797), rot=(1, 0, 0, 0)),
    )