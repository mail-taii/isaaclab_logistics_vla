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

from .scene_cfg import Spawn_ss_st_sparse_SceneCfg
from .observation_cfg import ObservationsCfg
from .reward_cfg import Spawn_ss_st_sparse_RewardCfg
from .event_cfg import Spawn_ss_st_sparse_EventCfg
from .command_cfg import Spawn_ss_st_sparse_CommandsCfg
from .command_cfg import Spawn_ss_st_sparse_with_obstacles_CommandsCfg

from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks import mdp

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 终止条件：当订单完成率 >= 0.999 时触发成功重置
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

@register.add_env_configs('Spawn_ss_st_sparse_EnvCfg')
@configclass
class Spawn_ss_st_sparse_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment (No Obstacles)."""
    
    scene: Spawn_ss_st_sparse_SceneCfg = Spawn_ss_st_sparse_SceneCfg(num_envs=1, env_spacing=7.0)
    
    observations: ObservationsCfg = ObservationsCfg()
    actions = register.load_action_configs('realman_franka_ee_actionscfg')()
    commands: Spawn_ss_st_sparse_CommandsCfg = Spawn_ss_st_sparse_CommandsCfg()
    rewards: Spawn_ss_st_sparse_RewardCfg = Spawn_ss_st_sparse_RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Spawn_ss_st_sparse_EventCfg = Spawn_ss_st_sparse_EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0 
        
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

@register.add_env_configs('Spawn_ss_st_sparse_with_obstacles_EnvCfg')
@configclass
class Spawn_ss_st_sparse_with_obstacles_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment (With Obstacles)."""
    
    scene: Spawn_ss_st_sparse_SceneCfg = Spawn_ss_st_sparse_SceneCfg(num_envs=2, env_spacing=7.0)
    
    observations: ObservationsCfg = ObservationsCfg()
    actions = register.load_action_configs('realman_franka_ee_actionscfg')()
    
    commands: Spawn_ss_st_sparse_with_obstacles_CommandsCfg = Spawn_ss_st_sparse_with_obstacles_CommandsCfg()
    
    rewards: Spawn_ss_st_sparse_RewardCfg = Spawn_ss_st_sparse_RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: Spawn_ss_st_sparse_EventCfg = Spawn_ss_st_sparse_EventCfg()
    
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        
        self.episode_length_s = 20.0         
        
        self.sim.dt = 0.01      # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625