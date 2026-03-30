from typing import TYPE_CHECKING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class VisionCfg(ObsGroup):
        """Observations for vision group."""

        # 1.头部相机
        head_rgb = ObsTerm(
            func=get_image_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb"}
        )
        
        # 2.左手腕相机
        left_wrist_rgb = ObsTerm(
            func=get_image_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb"}
        )
        
        # 3.右手腕相机
        right_wrist_rgb = ObsTerm(
            func=get_image_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            # 关闭展平操作，保留图像原本的(H, W, C)3D空间结构
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    vision: VisionCfg = VisionCfg()     #将视觉数据暴露给外部的强化学习或VLA模型