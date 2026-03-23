"""
UR5e 机器人相机配置：prim_path 绑定 UR5e 的 link（base_link、tool0 等）。
"""
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac


class UR5eCameraConfig:
    """UR5e：基座相机、末端相机、顶视相机（无头部，用 base_link 作“前视”）。"""

    # 前视相机：挂在 base_link，向前看
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.2, 0.4),
            rot=euler_to_quat_isaac(r=-135, p=0, y=180, return_tensor=False),
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        width=640,
        height=480,
        data_types=["rgb"],
    )

    # 末端相机：挂在 wrist_3_link（UrdfConverter 可能合并 fixed joint，tool0 未必单独存在）
    ee_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/ee_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),
            rot=euler_to_quat_isaac(r=90, p=0, y=0, return_tensor=False),
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 1.0e5),
        ),
        width=640,
        height=480,
        data_types=["rgb"],
    )

    # 顶视相机：世界系固定
    top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 2.0, 4.0),
            rot=euler_to_quat_isaac(r=0, p=180, y=0, return_tensor=False),
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=1024,
        height=768,
        data_types=["rgb"],
    )
