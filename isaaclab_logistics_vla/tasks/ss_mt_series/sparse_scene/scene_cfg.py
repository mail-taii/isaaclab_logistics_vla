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

# 定义本任务涉及的 SKU 池及其最大实例数
SKU_DEFINITIONS = {
    "cracker_box": (CRACKER_BOX_PARAMS['USD_PATH'], 2),
    "sugar_box":   (SUGER_BOX_PARAMS['USD_PATH'], 2),
    "tomato_soup_can": (TOMATO_SOUP_CAN_PARAMS['USD_PATH'], 2),
}

@configclass
class Spawn_ss_mt_sparse_SceneCfg(BaseOrderSceneCfg):
    """单原料箱至多目标箱 (SS-MT) 稀疏场景配置。"""

    # --- 机器人配置 ---
    robot: ArticulationCfg = register.load_robot('realman_franka_ee')().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos  = (0.96781, 2.28535, 0.216)
    robot.init_state.rot = (1, 0, 0, 0)

    # 禁用物理复制以支持每个环境独立的 Scale 修改（针对障碍物随机化需求）
    replicate_physics = False

    # 末端执行器参考系
    ee_frame: FrameTransformerCfg = register.load_eeframe_configs('realman_franka_ee_eeframe')()

    # --- 场景布局补充 ---
    # 提示：BaseOrderSceneCfg 通常已包含默认的 s_box_1, t_box_1 等
    # 在 SS-MT 任务中，我们需要确保 t_box_2 甚至 t_box_3 存在于场景中
    # 如果基类未定义，可以在此处显式添加或覆盖

# --- 动态注入 SKU 物品实例 ---
for sku_name, (usd_path, count) in SKU_DEFINITIONS.items():
    for i in range(count):
        instance_name = f"{sku_name}_{i}"
        
        # 定义 Config
        obj_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
            spawn=UsdFileCfg(
                usd_path=usd_path,
                scale=(1.0, 1.0, 1.0),
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    sleep_threshold=0.05
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1158019, 133614, 0),rot=(1, 0, 0, 0)),
        )
        
        # [关键] 动态注入到 MySceneCfg 类中
        # 这样 Isaac Lab 解析时就能看到这些属性
        setattr(Spawn_ss_mt_sparse_SceneCfg, instance_name, obj_cfg)