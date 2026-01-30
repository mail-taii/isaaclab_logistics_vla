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


def check_object_in_box(
    env_ids: torch.Tensor,
    target_asset: RigidObject,  
    box_asset,     
    box_size
):
    pos_obj = target_asset.data.root_pos_w[env_ids]

    pos_box = box_asset.data.root_pos_w[env_ids]

    quat_box = box_asset.data.root_quat_w[env_ids]

    relative_pos_world = pos_obj - pos_box  #计算世界坐标下的相对向量
    #箱子坐标系下，箱子中心点到物体的向量
    pos_local = math_utils.quat_apply_inverse(quat_box, relative_pos_world)  

    x_length, y_length, z_length = box_size

    in_x = torch.abs(pos_local[:, 0]) < (x_length / 2)

    in_y = torch.abs(pos_local[:, 1]) < (y_length / 2)

    in_z = (pos_local[:, 2] > 0) & (pos_local[:, 2] < z_length * 1.5) # 稍微允许高一点点

    return in_x & in_y & in_z


def get_rotated_aabb_size(dim_x, dim_y, dim_z, euler_deg, device='cpu'):
    """
    输入: 物体原始尺寸 (x, y, z) 和 欧拉角 (相对于父坐标系的度数)
    输出：在父级坐标系 X, Y, Z 轴上的投影长度。
    """
    dims = torch.tensor([dim_x, dim_y, dim_z], device=device, dtype=torch.float32)
    angles = torch.tensor(euler_deg, device=device, dtype=torch.float32)

    q:torch.Tensor = math_utils.quat_from_euler_xyz(torch.deg2rad(angles[0]), torch.deg2rad(angles[1]), torch.deg2rad(angles[2]))

    basis_vectors = torch.diag(dims)   # shape: (3, 3) -> 每一行代表一个轴向量: [x,0,0], [0,y,0], [0,0,z]

    rot_vectors = math_utils.quat_apply(q.repeat(3, 1), basis_vectors)

    new_dims = torch.abs(rot_vectors).sum(dim=0)
    
    return new_dims[0].item(), new_dims[1].item(), new_dims[2].item()
