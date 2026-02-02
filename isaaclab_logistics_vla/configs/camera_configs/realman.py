"""
Realman 机器人相机配置：prim_path 绑定 realman 的 link（head_link2、panda_left_hand 等）。
"""
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac


class RealmanCameraConfig:
    """Realman 机器人：头部相机、末端相机、顶视相机。"""

    # 头部相机：挂在 head_link2/camera_link 节点，向前看
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link2/camera_link/head_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=euler_to_quat_isaac(r=45, p=180, y=0),
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
    )

    # 末端相机：挂在 panda_left_hand，视线与末端 +Z（TCP 指向）一致
    ee_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_left_hand/ee_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.08),
            rot=euler_to_quat_isaac(r=60, p=180, y=0),
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 1.0e5),
        ),
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
    )

    # 顶视相机：固定在世界系上方，俯视整个工作区
    top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 4.0, 4.0),
            rot=euler_to_quat_isaac(r=0, p=180, y=0),
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=1024,
        height=768,
        data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
    )
