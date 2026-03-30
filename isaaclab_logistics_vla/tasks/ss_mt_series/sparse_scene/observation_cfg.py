from typing import TYPE_CHECKING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_logistics_vla.tasks import mdp
from isaaclab_logistics_vla.utils.register import register
from isaaclab_logistics_vla.tasks.BaseOrderCommandTerm import *
from isaaclab_logistics_vla.tasks.base_observation_cfg import BaseObservationsCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

@configclass
class ObservationsCfg(BaseObservationsCfg):
    pass