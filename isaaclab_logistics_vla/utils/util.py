from typing import TYPE_CHECKING
import torch
from abc import abstractmethod
from typing import TYPE_CHECKING

from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_logistics_vla.utils.object_position import *
from isaaclab_logistics_vla.utils.constant import *
from isaaclab_logistics_vla.utils.util import *
from isaaclab_logistics_vla.utils.path_utils import *

from scipy.spatial.transform import Rotation as R
import torch
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def get_image_from_sensor(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, data_type: str):
    """
    从指定的相机传感器获取图像张量，返回的Tensor形状为(num_envs, H, W, C)
    """
    return env.scene.sensors[sensor_cfg.name].data.output[data_type]

def euler2quat(axis = 'z',degree = 0):
    '''
    绕axis轴旋转degree 度后得到的四元数(w, x, y, z)
    '''
    r = R.from_euler('z', degree, degrees=True)
    quat = r.as_quat() # 注意 scipy 默认返回 (x, y, z, w)
    # 我们需要转换成 Isaac Lab 的 (w, x, y, z)
    isaac_quat = (quat[3], quat[0], quat[1], quat[2])
    return isaac_quat

def euler_to_quat_isaac(r, p, y):
    """
    输入: r, p, y  角度制
    输出: (w, x, y, z) 格式的四元数
    """
    # 如果输入是标量，自动转换成 tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(r, torch.Tensor):
        
        r = torch.tensor([r], device=device)
        p = torch.tensor([p], device=device)
        y = torch.tensor([y], device=device)
        
    r = torch.deg2rad(r)
    p = torch.deg2rad(p)
    y = torch.deg2rad(y)
    quat:torch.Tensor = math_utils.quat_from_euler_xyz(r, p, y)
    quat = quat.to(device)
    return quat