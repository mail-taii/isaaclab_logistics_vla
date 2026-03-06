import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .scene_cfg import Spawn_ss_mt_sparse_SceneCfg
from .observation_cfg import ObservationsCfg
from .command_cfg import Spawn_ss_mt_sparse_CommandsCfg
from .reward_cfg import Spawn_ss_mt_sparse_RewardCfg
from .event_cfg import Spawn_ss_mt_sparse_EventCfg

from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks import mdp

@configclass
class TerminationsCfg:
    """SS-MT 任务的终止条件配置。"""

    # 1. 超时重置
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2. 订单全成功重置
    # 在 SS-MT 中，只有当所有目标箱的物品都放置正确，才触发此终止
    order_success = DoneTerm(
        func=mdp.check_order_completion,
        params={
            "command_name": "order_info", 
            # 阈值设为 0.999 代表要求所有 target_id != -1 的物品必须全部完成
            "threshold": 0.999, 
        },
    )

@configclass
class CurriculumCfg:
    pass

@register.add_env_configs('Spawn_ss_mt_sparse_EnvCfg')
@configclass
class Spawn_ss_mt_sparse_EnvCfg(ManagerBasedRLEnvCfg):
    """单源-多目标 (SS-MT) 物流分拣环境的主配置类。"""

    # --- 场景设置 ---
    # 增加 env_spacing 确保多目标箱布局时环境不重叠
    scene: Spawn_ss_mt_sparse_SceneCfg = Spawn_ss_mt_sparse_SceneCfg(num_envs=4, env_spacing=8.0)

    # --- 基础 MDP 构成 ---
    observations: ObservationsCfg = ObservationsCfg()
    actions = register.load_action_configs('realman_franka_ee_actionscfg')()
    commands: Spawn_ss_mt_sparse_CommandsCfg = Spawn_ss_mt_sparse_CommandsCfg()

    # --- 奖励与事件 ---
    rewards: Spawn_ss_mt_sparse_RewardCfg = Spawn_ss_mt_sparse_RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Spawn_ss_mt_sparse_EventCfg = Spawn_ss_mt_sparse_EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
