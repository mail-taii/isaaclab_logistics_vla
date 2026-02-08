import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from isaaclab.utils import math as math_utils

def euler2quat(axis = 'z',degree = 0):
    '''
    绕axis轴旋转degree 度后得到的四元数(w, x, y, z)
    '''
    r = R.from_euler(axis, degree, degrees=True)
    quat = r.as_quat() # 注意 scipy 默认返回 (x, y, z, w)
    # 我们需要转换成 Isaac Lab 的 (w, x, y, z)
    isaac_quat = (quat[3], quat[0], quat[1], quat[2])
    return isaac_quat

def camera_rot_look_along_parent_x():
    """
    相机挂于父 link 时，使视线与父系 +X（基座/头部指向）一致。
    先绕 X 轴 180° 修正默认朝上，再绕 Y 轴 -90° 使视线沿 +X。
    返回 (w, x, y, z) 格式四元数，用于 CameraCfg.OffsetCfg.rot。
    """
    r = R.from_euler("xy", [180.0, -90.0], degrees=True)
    q = r.as_quat()  # scipy (x, y, z, w)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def camera_rot_look_along_parent_z():
    """
    相机挂于父 link 时，使视线与父系 +Z（末端 TCP 指向）一致。
    绕 X 轴 180° 修正默认朝上后，视线即沿 +Z，无需额外旋转。
    返回 (w, x, y, z) 格式四元数，用于 CameraCfg.OffsetCfg.rot。
    """
    return (0.0, 1.0, 0.0, 0.0)


def euler_to_quat_isaac(r, p, y, return_tensor=False):
    """
    输入: r, p, y  角度制
    输出: (w, x, y, z) 格式的四元数
    
    参数:
        return_tensor: 是否返回 PyTorch 张量，默认为 False（返回普通元组）
    """
    # 检查是否所有输入都是标量
    all_scalars = not isinstance(r, torch.Tensor) and not isinstance(p, torch.Tensor) and not isinstance(y, torch.Tensor)
    
    if all_scalars and not return_tensor:
        # 使用 scipy 计算四元数，返回普通元组
        from scipy.spatial.transform import Rotation as R
        r_obj = R.from_euler('xyz', [r, p, y], degrees=True)
        quat = r_obj.as_quat()  # (x, y, z, w)
        # 转换为 (w, x, y, z) 格式
        return (quat[3], quat[0], quat[1], quat[2])
    
    # 如果需要返回张量，或者输入是张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保所有输入都是张量
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, device=device)
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, device=device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, device=device)
    
    r = torch.deg2rad(r)
    p = torch.deg2rad(p)
    y = torch.deg2rad(y)
    quat:torch.Tensor = math_utils.quat_from_euler_xyz(r, p, y)
    quat = quat.to(device)
    return quat