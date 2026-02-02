import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from isaaclab.app import AppLauncher


def _summarize_tensor(t):
    import torch

    if not isinstance(t, torch.Tensor):
        return {"type": type(t).__name__}
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }


def _summarize_observation(obs):
    """把 ObservationDict 转成易读结构（只看有没有 / shape / dtype），用于快速 sanity check."""
    summary = {}

    meta = obs.get("meta")
    if meta is not None:
        summary["meta"] = dict(meta)

    rs = obs.get("robot_state")
    if rs is not None:
        rs_sum = {}
        for k, v in rs.items():
            rs_sum[k] = _summarize_tensor(v)
        summary["robot_state"] = rs_sum

    vision = obs.get("vision")
    if vision is not None:
        v_sum = {}
        cams = vision.get("cameras")
        if cams is not None:
            v_sum["cameras"] = list(cams)
        for key in ["rgb", "depth", "segmentation", "intrinsic", "extrinsic"]:
            if key in vision:
                v_sum[key] = _summarize_tensor(vision[key])
        summary["vision"] = v_sum

    pcd = obs.get("point_cloud")
    if pcd is not None:
        p_sum = {}
        if "masked_point_cloud" in pcd:
            mpc = pcd["masked_point_cloud"]
            import torch

            p_sum["masked_point_cloud"] = {
                "type": type(mpc).__name__,
                "is_nested": torch.is_nested_tensor(mpc),
            }
        if "frame" in pcd:
            p_sum["frame"] = pcd["frame"]
        summary["point_cloud"] = p_sum

    return summary


def _pad_to_grid(frames: List[np.ndarray], grid_hw: Tuple[int, int], cell_hw: Tuple[int, int]) -> np.ndarray:
    """把多路图像拼成网格（不足补黑，分辨率自动对齐到 cell_hw）。

    - frames: list of (H,W,3) uint8，允许每路相机分辨率不同
    - cell_hw: 目标单元格分辨率 (cell_h, cell_w)，通常取第一路相机的分辨率
    """
    grid_h, grid_w = grid_hw
    cell_h, cell_w = cell_hw
    target_n = grid_h * grid_w

    def _fit_to_cell(img: np.ndarray) -> np.ndarray:
        """把任意分辨率 img 填充/裁剪成 (cell_h, cell_w, 3)。"""
        h, w = img.shape[:2]
        canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        # 取重叠区域尺寸
        copy_h = min(h, cell_h)
        copy_w = min(w, cell_w)
        # 简单从左上角对齐复制（够用；如果以后需要可以改成居中对齐）
        canvas[:copy_h, :copy_w, :] = img[:copy_h, :copy_w, :]
        return canvas

    # 先把所有帧 resize/crop 到统一 cell 尺寸
    frames = [_fit_to_cell(f) for f in frames]

    if len(frames) < target_n:
        pad = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        frames = frames + [pad] * (target_n - len(frames))
    else:
        frames = frames[:target_n]

    rows = []
    idx = 0
    for _ in range(grid_h):
        row = np.concatenate(frames[idx : idx + grid_w], axis=1)
        rows.append(row)
        idx += grid_w
    return np.concatenate(rows, axis=0)


def _get_rgb_from_camera(cam) -> "np.ndarray":
    """从 IsaacLab Camera 传感器里取一帧 RGB（uint8, HxWx3）。

    IsaacLab 2.2.1: cam.data 是 CameraData，图像在 cam.data.output 字典里（没有 cam.data.rgb 属性）。
    """
    data = getattr(cam, "data", None)
    if data is None:
        raise AttributeError("camera has no .data")
    output = getattr(data, "output", None)
    if not isinstance(output, dict):
        raise AttributeError("camera.data has no .output dict")

    # 常见键：rgb / rgba（部分实现会先给 rgba，再派生 rgb）
    if "rgb" in output and output["rgb"] is not None:
        rgb = output["rgb"]
    elif "rgba" in output and output["rgba"] is not None:
        rgb = output["rgba"][..., :3]
    else:
        raise KeyError(f"camera.data.output keys = {list(output.keys())}, missing rgb/rgba")

    # rgb: (num_envs, H, W, 3) torch.Tensor
    frame = rgb[0].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=None, help="录制时长（秒）。若设置，则 steps = round(seconds * fps)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="./camera_videos")
    # 这里保留 headless/enable_cameras 参数，是为了和 AppLauncher 接口对齐
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--enable_cameras", action="store_true", default=True)  # 默认启用相机
    args_cli, _ = parser.parse_known_args()
    if args_cli.seconds is not None:
        args_cli.steps = max(1, int(round(args_cli.fps * args_cli.seconds)))

    # AppLauncher 会根据 args_cli.headless / args_cli.enable_cameras 配置 IsaacSim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch
    import imageio

    from isaaclab_logistics_vla.tasks.test_tasks.order_series.env_cfg import OrderEnvCfg
    from isaaclab_logistics_vla.tasks.test_tasks.order_series.observation_cfg import (
        ObservationsCfg,
    )
    from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv
    from isaaclab_logistics_vla.evaluation.observation.builder import (
        EpisodeContext,
        ObservationBuilder,
    )
    from isaaclab_logistics_vla.evaluation.observation.schema import ObservationRequire

    env_cfg = OrderEnvCfg()
    env_cfg.sim.device = args_cli.device
    # 挂上 observation 配置，方便 ObservationBuilder 从 observation_manager 里拿到关节信息
    env_cfg.observations = ObservationsCfg()
    # 这里我们已经在 env_cfg 里把 num_envs 固定为 1

    env = VLAIsaacEnv(cfg=env_cfg)
    env.reset()

    # ObservationBuilder：用于从当前 env 抽取标准化 ObservationDict
    builder = ObservationBuilder(env)
    ctx = EpisodeContext(task_name="order_series", episode_id=0)
    require = ObservationRequire(
        require_rgb=True,
        require_depth=True,
        require_seg=True,
        require_pcd=True,
        pcd_frame="camera",
    )

    # 找到相机传感器
    sensors: Dict[str, object] = getattr(env.unwrapped.scene, "sensors", {})
    cam_names = [name for name in sensors.keys() if "camera" in name.lower()]
    cam_names = sorted(cam_names)
    if len(cam_names) == 0:
        raise RuntimeError("未找到任何相机传感器（scene.sensors 中不含 camera）。请确认 scene_cfg.py 已添加 head_camera/ee_camera。")

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每路相机单独录制 + 九宫格拼接录制
    writers = {}
    grid_writer = None

    # 先跑一步拿到分辨率
    # 用零动作推进。动作维度不要从 env.action_space 读（可能是 1），
    # 而是直接使用底层 ActionManager 的 total_action_dim，避免维度不匹配。
    num_actions = env.unwrapped.action_manager.total_action_dim
    actions = torch.zeros((env.num_envs, num_actions), device=env.device)
    obs, rew, terminated, truncated, info = env.step(actions)

    # 在第一步之后，构建一次完整 ObservationDict 做 sanity check
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    camera_names = sorted([name for name in sensors.keys() if "camera" in name.lower()])
    obs_full = builder.build(
        ctx=ctx,
        step_id=0,
        require=require,
        camera_names=camera_names,
    )
    from pprint import pprint

    print("=== Observation summary (step 0) from ObservationBuilder ===")
    pprint(_summarize_observation(obs_full), width=120)

    # 可选：把完整 ObservationDict 存下来，方便你在 notebook 里 torch.load 详细看
    dump_path = Path(args_cli.out_dir) / "order_series_obs_step0.pt"
    try:
        torch.save(obs_full, dump_path)
        print(f"[INFO] Saved step-0 ObservationDict to: {dump_path}")
    except Exception as e:
        print(f"[WARN] Failed to save ObservationDict: {e}")

    # 获取一帧 rgb 来确定 H/W
    first_frames = []
    for name in cam_names:
        cam = sensors[name]
        first_frames.append(_get_rgb_from_camera(cam))

    H, W = first_frames[0].shape[:2]

    for name in cam_names:
        writers[name] = imageio.get_writer(str(out_dir / f"{name}.mp4"), fps=args_cli.fps)
    grid_writer = imageio.get_writer(str(out_dir / "grid_3x3.mp4"), fps=args_cli.fps)

    # 写入第一帧
    for name, frame in zip(cam_names, first_frames):
        writers[name].append_data(frame)
    grid_writer.append_data(_pad_to_grid(first_frames, grid_hw=(3, 3), cell_hw=(H, W)))

    # 主循环：录制 frames
    for _ in range(args_cli.steps - 1):
        obs, rew, terminated, truncated, info = env.step(actions)

        frames = []
        for name in cam_names:
            cam = sensors[name]
            frame = _get_rgb_from_camera(cam)
            frames.append(frame)
            writers[name].append_data(frame)

        grid_writer.append_data(_pad_to_grid(frames, grid_hw=(3, 3), cell_hw=(H, W)))

    # 关闭 writer
    for w in writers.values():
        w.close()
    if grid_writer is not None:
        grid_writer.close()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

