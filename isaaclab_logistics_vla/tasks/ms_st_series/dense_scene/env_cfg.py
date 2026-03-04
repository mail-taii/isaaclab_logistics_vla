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

from .scene_cfg import Spawn_ms_st_dense_SceneCfg
from .observation_cfg import ObservationsCfg
from .command_cfg import Spawn_ms_st_dense_CommandsCfg
from .reward_cfg import Spawn_ms_st_dense_RewardCfg
from .event_cfg import Spawn_ms_st_dense_EventCfg

from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks import mdp

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 保留超时重置：这是必须的，否则环境永远不停
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    order_success = DoneTerm(
        func=mdp.check_order_completion,
        params={
            "command_name": "order_info", 
            "threshold": 0.999, 
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@register.add_env_configs('Spawn_ms_st_dense_EnvCfg')
@configclass
class Spawn_ms_st_dense_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    # Scene settings
    scene: Spawn_ms_st_dense_SceneCfg = Spawn_ms_st_dense_SceneCfg(num_envs=4,env_spacing = 7.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions  = register.load_action_configs('realman_franka_ee_actionscfg')()
    commands: Spawn_ms_st_dense_CommandsCfg = Spawn_ms_st_dense_CommandsCfg()
    # MDP settings
    rewards: Spawn_ms_st_dense_RewardCfg = Spawn_ms_st_dense_RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Spawn_ms_st_dense_EventCfg = Spawn_ms_st_dense_EventCfg()
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

        # ======== ⬇️ 新增的 PhysX GPU 缓冲区扩容配置 ⬇️ ========
        # 解决 Patch buffer overflow detected 报错 (直接给到约200万，满足报错提示的85万+)
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024 * 2  
        
        # 接触面暴增，接触点总量也需要同步放大
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 * 4 
        
        # 密集场景防爆必备，配合你上面的 aggregate_pairs 容量
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 2
