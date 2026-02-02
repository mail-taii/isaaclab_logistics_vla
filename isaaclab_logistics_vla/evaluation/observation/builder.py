from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

from isaaclab.envs import ManagerBasedRLEnv

from .schema import (
    MetaInfo,
    ObservationDict,
    ObservationRequire,
    PointCloudData,
    RobotState,
    VisionData,
)


@dataclass
class EpisodeContext:
    """用于补充 meta 信息的简单上下文."""

    task_name: str
    episode_id: int


class ObservationBuilder:
    """把 IsaacLab 环境里的数据整理成统一的 ObservationDict。

    当前已支持：
    - meta（task_name / episode_id / step_id / num_envs）
    - robot_state（qpos/qvel/qacc）
    - vision（多相机 rgb/depth/seg + intrinsic）
    - point_cloud（可选，从 depth+intrinsic 在相机坐标系下生成）
    """

    def __init__(self, env: ManagerBasedRLEnv):
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.unwrapped.num_envs

    def build(
        self,
        ctx: EpisodeContext,
        step_id: int,
        require: Optional[ObservationRequire] = None,
        camera_names: Optional[List[str]] = None,
    ) -> ObservationDict:
        """构建一次完整观测。"""
        if require is None:
            require = ObservationRequire()

        obs: ObservationDict = {}

        # 1) meta
        meta: MetaInfo = {
            "task_name": ctx.task_name,
            "episode_id": ctx.episode_id,
            "step_id": step_id,
            "num_envs": self.num_envs,
        }
        obs["meta"] = meta

        # 2) robot_state：优先从 observation_manager 的 policy group 里拿
        robot_state: RobotState = self._build_robot_state()
        obs["robot_state"] = robot_state

        # 3) vision：根据 require 开关和 camera_names 采集多视角图像
        if require.require_rgb or require.require_depth or require.require_seg:
            vision = self._build_vision(require, camera_names)
            if vision:
                obs["vision"] = vision

        # 4) point cloud：在相机坐标系下，从 depth+intrinsic 生成
        if require.require_pcd and "vision" in obs:
            pcd = self._build_point_cloud(obs["vision"], require)
            if pcd:
                obs["point_cloud"] = pcd

        return obs

    # --------------------------------------------------------------------- #
    # 内部工具：从 env 中获取机器人状态
    # --------------------------------------------------------------------- #

    def _build_robot_state(self) -> RobotState:
        """从 env 中读取 qpos/qvel/qacc.

        优先尝试：
        - env.unwrapped.observation_manager.compute_group("policy")

        否则 fallback 到：
        - env.unwrapped.scene["robot"].data.joint_* 张量
        """
        env = self._env.unwrapped

        qpos: torch.Tensor
        qvel: torch.Tensor

        # 优先走 observation_manager/policy group
        if hasattr(env, "observation_manager"):
            obs_group = env.observation_manager.compute_group("policy")
            # 注意：当 ObservationGroupCfg.concatenate_terms=True 时，compute_group 返回的是一个拼接后的 Tensor，
            #       这时不能对它做 `"joint_pos" in obs_group` 这种字典判断。
            if isinstance(obs_group, dict) and "joint_pos" in obs_group and "joint_vel" in obs_group:
                qpos = obs_group["joint_pos"]
                qvel = obs_group["joint_vel"]
            else:
                # 回退到 scene 里的机器人数据
                robot = env.scene["robot"]
                qpos = robot.data.joint_pos
                qvel = robot.data.joint_vel
        else:
            robot = env.scene["robot"]
            qpos = robot.data.joint_pos
            qvel = robot.data.joint_vel

        # 目前 IsaacLab 默认没有直接的加速度观测，这里先填 0，后续如有需要可在 env 中补充 term
        qacc = torch.zeros_like(qpos)

        return RobotState(qpos=qpos, qvel=qvel, qacc=qacc)

    # --------------------------------------------------------------------- #
    # 视觉观测：rgb / depth / seg + intrinsic
    # --------------------------------------------------------------------- #

    def _iter_cameras(
        self, camera_names: Optional[List[str]]
    ) -> List[Tuple[str, object]]:
        """根据名称筛选 scene.sensors 中的相机传感器."""
        env = self._env.unwrapped
        sensors: Dict[str, object] = getattr(env.scene, "sensors", {})

        if not sensors:
            return []

        if camera_names is None:
            # 默认：挑出名字里带 camera 的传感器
            return [(name, sensor) for name, sensor in sensors.items() if "camera" in name.lower()]
        else:
            out = []
            for name in camera_names:
                if name in sensors:
                    out.append((name, sensors[name]))
            return out

    def _build_vision(
        self, require: ObservationRequire, camera_names: Optional[List[str]]
    ) -> VisionData:
        """从相机传感器中构造 VisionData."""
        cameras = self._iter_cameras(camera_names)
        if not cameras:
            return {}

        rgb_list: List[torch.Tensor] = []
        depth_list: List[torch.Tensor] = []
        seg_list: List[torch.Tensor] = []
        intrinsic_list: List[torch.Tensor] = []

        camera_name_order: List[str] = []

        for name, sensor in cameras:
            data = getattr(sensor, "data", None)
            if data is None:
                log.debug("相机 %s 的 data 为 None", name)
                continue

            camera_name_order.append(name)

            log.debug("相机 %s 数据属性: %s", name, [attr for attr in dir(data) if not attr.startswith('_')])

            rgb_data = None
            if hasattr(data, "output"):
                rgb_data = data.output
                log.debug("使用 data.output")
                if isinstance(rgb_data, dict):
                    log.debug("data.output 字典键: %s", list(rgb_data.keys()))
                    if "rgb" in rgb_data:
                        rgb_data = rgb_data["rgb"]
                        log.debug("从 output 字典获取 rgb")
                    elif "image" in rgb_data:
                        rgb_data = rgb_data["image"]
                        log.debug("从 output 字典获取 image")
            elif hasattr(data, "image"):
                rgb_data = data.image
                log.debug("使用 data.image")
            elif hasattr(data, "rgb"):
                rgb_data = data.rgb
                log.debug("使用 data.rgb")

            if require.require_rgb and rgb_data is not None and hasattr(rgb_data, "shape"):
                rgb_list.append(rgb_data)
                log.debug("相机 %s 获取到 RGB 数据，形状: %s", name, rgb_data.shape)
            else:
                log.debug("相机 %s 无法获取 RGB 数据 - require_rgb: %s", name, require.require_rgb)
                if rgb_data is not None:
                    log.debug("rgb_data 类型: %s", type(rgb_data))
                    if isinstance(rgb_data, dict):
                        log.debug("rgb_data 字典键: %s", list(rgb_data.keys()))

            if require.require_depth and hasattr(data, "distance_to_image_plane"):
                depth_list.append(data.distance_to_image_plane)  # (num_envs, H, W)

            if require.require_seg and hasattr(data, "instance_segmentation"):
                seg_list.append(data.instance_segmentation)  # (num_envs, H, W)

            if hasattr(data, "intrinsic_matrices"):
                intrinsic_list.append(data.intrinsic_matrices)  # (num_envs, 3, 3)

        vision: VisionData = {}
        if not camera_name_order:
            log.debug("没有可用的相机")
            return vision

        vision["cameras"] = camera_name_order

        if rgb_list:
            shapes = [tensor.shape for tensor in rgb_list]
            log.debug("所有相机 RGB 形状: %s", shapes)

            if len(set(shapes)) > 1:
                log.debug("相机尺寸不一致，进行尺寸调整")
                target_shape = shapes[0]

                import torch.nn.functional as F
                resized_list = []
                for i, tensor in enumerate(rgb_list):
                    if tensor.shape != target_shape:
                        _, h, w, c = target_shape
                        current_tensor = tensor.permute(0, 3, 1, 2).float()
                        resized_tensor = F.interpolate(current_tensor, size=(h, w), mode='bilinear', align_corners=False)
                        resized_tensor = resized_tensor.clamp(0, 255).byte()
                        resized_tensor = resized_tensor.permute(0, 2, 3, 1)
                        resized_list.append(resized_tensor)
                        log.debug("相机 %s 从 %s 调整为 %s", i, tensor.shape, resized_tensor.shape)
                    else:
                        resized_list.append(tensor)
                rgb_list = resized_list

            vision["rgb"] = torch.stack(rgb_list, dim=0)
            log.debug("最终 RGB 数据形状: %s", vision["rgb"].shape)
        else:
            log.debug("RGB 列表为空，无法构建 vision['rgb']")
            
        if depth_list:
            vision["depth"] = torch.stack(depth_list, dim=0)
        if seg_list:
            vision["segmentation"] = torch.stack(seg_list, dim=0)
        if intrinsic_list:
            vision["intrinsic"] = torch.stack(intrinsic_list, dim=0)

        log.debug("最终 vision 字段: %s", list(vision.keys()))
        return vision

    # --------------------------------------------------------------------- #
    # 点云：从 depth + intrinsic 在相机坐标系下生成
    # --------------------------------------------------------------------- #

    def _build_point_cloud(
        self, vision: VisionData, require: ObservationRequire
    ) -> PointCloudData:
        """从 depth + intrinsic 生成简单点云（相机坐标系），不依赖 extrinsic."""
        if "depth" not in vision or "intrinsic" not in vision:
            return {}

        depth = vision["depth"]          # (C, num_envs, H, W)
        intrinsic = vision["intrinsic"]  # (C, num_envs, 3, 3)

        C, num_envs, H, W = depth.shape

        # 预先生成像素网格（H, W）
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        all_env_points: List[torch.Tensor] = []

        for env_id in range(num_envs):
            env_points: List[np.ndarray] = []
            for cam_id in range(C):
                depth_img = depth[cam_id, env_id].cpu().numpy()  # (H, W)
                K = intrinsic[cam_id, env_id].cpu().numpy()      # (3, 3)

                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]

                z = depth_img
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)  # (N, 3)

                # 简单过滤掉 z<=0 的点
                valid = points_cam[:, 2] > 0
                points_cam = points_cam[valid]

                env_points.append(points_cam)

            if env_points:
                merged = np.concatenate(env_points, axis=0)  # (N, 3)
                all_env_points.append(torch.from_numpy(merged).to(depth.device))
            else:
                all_env_points.append(torch.zeros((0, 3), device=depth.device))

        # 打包成 (num_envs, N_i, 3) 的 list，再在外面由调用方决定怎么对齐 N
        # 这里先简单用 list of tensors 的形式，保持灵活
        # schema 中定义的是 Tensor，但真正在序列化时可以把每个 env 单独转 list
        pcd: PointCloudData = {
            "masked_point_cloud": torch.nested.nested_tensor(all_env_points),
            "frame": require.pcd_frame,
        }
        return pcd

