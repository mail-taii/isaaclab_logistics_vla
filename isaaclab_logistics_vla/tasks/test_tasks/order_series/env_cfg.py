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

from isaaclab_logistics_vla import ISAACLAB_LOGISTICS_VLA_EXT_DIR

from isaaclab_logistics_vla.tasks.test_tasks.order_series.command_cfg import CommandsCfg
from isaaclab_logistics_vla.tasks.test_tasks.order_series.observation_cfg import ObservationsCfg
from isaaclab_logistics_vla.tasks.test_tasks.order_series.event_cfg import EventCfg
from isaaclab_logistics_vla.tasks.test_tasks.order_series.reward_cfg import RewardsCfg
from isaaclab_logistics_vla.tasks.test_tasks.order_series.scene_cfg import OrderSceneCfg, get_order_scene_cfg
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.evaluation.robot_registry import get_robot_eval_config
from isaaclab_logistics_vla.tasks import mdp


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 保留超时重置：这是必须的，否则环境永远不停
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@register.add_env_configs('OrderEnvCfg')
@configclass
class OrderEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment. robot_id 决定场景中的机器人与相机绑定。"""
    robot_id: str = "realman_dual_left_arm"
    # Scene settings：按 robot_id 选择机器人 asset 与相机配置
    scene: OrderSceneCfg = OrderSceneCfg(num_envs=1, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions = register.load_action_configs("realman_franka_ee_actionscfg")()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization. 按 robot_id 切换场景（机器人 + 相机绑定）与动作配置。"""
        cfg = get_robot_eval_config(self.robot_id)
        self.scene = get_order_scene_cfg(
            self.robot_id,
            num_envs=self.scene.num_envs,
            env_spacing=self.scene.env_spacing,
        )
        if getattr(cfg, "action_config_key", None):
            self.actions = register.load_action_configs(cfg.action_config_key)()
        # general settings
        self.decimation = 2
        self.episode_length_s = 50
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625