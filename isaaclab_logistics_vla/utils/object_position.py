import math
import re
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions
from isaaclab.envs import ManagerBasedEnv


def _set_asset_global_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset: RigidObject | Articulation,
    global_pos: torch.Tensor,
    global_quat: torch.Tensor
):
    """
    直接写入世界坐标，不处理任何偏移。
    """
    # 写入位姿
    asset.write_root_pose_to_sim(
        torch.cat([global_pos, global_quat], dim=-1), 
        env_ids=env_ids
    )
    # 清零速度
    asset.write_root_velocity_to_sim(
        torch.zeros(len(env_ids), 6, device=env.device), 
        env_ids=env_ids
    )

def set_asset_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset: RigidObject | Articulation,
    position,
    quat=None
):
    """
    强制将asset按给定位置和角度放置在指定环境的指定位置
    position:(N,3) 
    quat:(N,4)
    """

    global_pos = position + env.scene.env_origins[env_ids]
    if quat is None:
        quat = torch.tensor([1, 0, 0, 0], device=env.device).repeat(len(env_ids), 1)
    
    _set_asset_global_pose(env,env_ids,asset,global_pos,quat)

def set_asset_relative_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    target_asset: RigidObject | Articulation,    # 物体 A (要放置的)
    reference_asset: RigidObject | Articulation, # 物体 B (参考物)
    relative_pos: torch.Tensor,        
    relative_quat: torch.Tensor = None   # A 相对于 B 的旋转 (w, x, y, z)
):
    """
    在指定环境中，强制将target_asset放在reference_asset的某相对位置
    elative_pos: torch.Tensor,          # A 相对于 B 的位置 (x, y, z) Shape(N,3)
    relative_quat: torch.Tensor = None   # A 相对于 B 的旋转 (w, x, y, z) Shape(N,4)
    """
    asset_A = target_asset
    asset_B = reference_asset

    pos_B_w = asset_B.data.root_pos_w[env_ids]
    quat_B_w = asset_B.data.root_quat_w[env_ids]

    if relative_quat is None:
        relative_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)

    # 确保 relative_pos 也是 batch 的形状 (如果传入的是单个坐标)
    if relative_pos.dim() == 1:
        relative_pos = relative_pos.repeat(len(env_ids), 1)

    pos_A_w, quat_A_w = math_utils.combine_frame_transforms(
        t01=pos_B_w,  q01=quat_B_w,      # 父坐标系 (B)
        t12=relative_pos, q12=relative_quat # 相对变换 (A in B)
    )

    _set_asset_global_pose(env,env_ids,target_asset,pos_A_w,quat_A_w)