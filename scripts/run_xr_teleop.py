"""
XR(handtracking) 遥操作入口：在 isaaclab_logistics_vla 的任务场景上启用 CloudXR/OpenXR。

工作流（先确保已按 Isaac Lab 文档启动 CloudXR Runtime，并在 Isaac Sim UI 中 Start AR）：

最小命令（使用默认参数）:
```bash
conda activate env_isaaclab
cd /path/to/isaaclab_logistics_vla
./isaaclab.sh -p scripts/run_xr_teleop.py --xr
```

完整命令（显式指定常用参数；动作幅度也可用 --teleop_* 与 --control_hz 等写在一起）:
```bash
conda activate env_isaaclab
cd /path/to/isaaclab_logistics_vla

./isaaclab.sh -p scripts/run_xr_teleop.py \
  --task_scene_name Spawn_ds_st_sparse_XRTeleop_EnvCfg \
  --num_envs 1 \
  --control_hz 45 \
  --asset_root_path /home/junzhe/Benchmark \
  --device cuda:0 \
  --teleop_pos_scale 8 \
  --teleop_rot_scale 8 \
  --teleop_ik_scale 1.0 \
  --xr
```

视角仍可用环境变量：TELEOP_XR_ANCHOR_POS、TELEOP_XR_ANCHOR_ROT。
"""

import argparse
import os
from collections.abc import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Apple Vision Pro (CloudXR/OpenXR) 遥操作驱动 Logistics VLA 场景")
parser.add_argument(
    "--task_scene_name",
    type=str,
    default="Spawn_ds_st_sparse_XRTeleop_EnvCfg",
    help="XR teleop 用的任务场景名（建议用 *_XRTeleop_EnvCfg 变体）",
)
parser.add_argument("--control_hz", type=float, default=45.0, help="控制循环频率（XR 下通常 45Hz 渲染）")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量（XR 遥操作建议 1）")
parser.add_argument(
    "--asset_root_path",
    type=str,
    default="/home/junzhe/Benchmark",
    help="资产根路径",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    choices=["handtracking"],
    help="仅支持 handtracking（OpenXR/CloudXR）",
)
parser.add_argument(
    "--teleop_pos_scale",
    type=float,
    default=None,
    help="手部位移放大倍数，越大机器人动得越远（默认 8.0，也可用 TELEOP_POS_SCALE 环境变量）",
)
parser.add_argument(
    "--teleop_rot_scale",
    type=float,
    default=None,
    help="手部旋转放大倍数（默认 8.0，也可用 TELEOP_ROT_SCALE 环境变量）",
)
parser.add_argument(
    "--teleop_ik_scale",
    type=float,
    default=None,
    help="IK 末端跟随幅度（默认 1.0，也可用 TELEOP_IK_SCALE 环境变量）",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# handtracking 必须启用 XR experience
args_cli.xr = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    if not os.path.exists(args_cli.asset_root_path):
        raise FileNotFoundError(f"资产路径不存在: {args_cli.asset_root_path}")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

    import time
    import torch
    import omni.log

    import isaaclab_tasks  # noqa: F401
    import isaaclab_logistics_vla  # noqa: F401
    from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv
    from isaaclab_logistics_vla.utils.register import register

    # 扫描注册表（包含 *_XRTeleop_EnvCfg）
    register.auto_scan("isaaclab_logistics_vla.tasks")

    def _teleop_scale(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    # 命令行优先，未传则用环境变量
    teleop_pos_scale = args_cli.teleop_pos_scale if args_cli.teleop_pos_scale is not None else _teleop_scale("TELEOP_POS_SCALE", 8.0)
    teleop_rot_scale = args_cli.teleop_rot_scale if args_cli.teleop_rot_scale is not None else _teleop_scale("TELEOP_ROT_SCALE", 8.0)
    teleop_ik_scale = args_cli.teleop_ik_scale if args_cli.teleop_ik_scale is not None else _teleop_scale("TELEOP_IK_SCALE", 1.0)
    # 分轴位置缩放：若「往前伸」卡住、上下却正常，可单独放大某一轴（世界系 X/Y/Z，默认 1.0 不额外缩放）
    teleop_pos_scale_x = _teleop_scale("TELEOP_POS_SCALE_X", 1.0)
    teleop_pos_scale_y = _teleop_scale("TELEOP_POS_SCALE_Y", 1.0)
    teleop_pos_scale_z = _teleop_scale("TELEOP_POS_SCALE_Z", 1.0)
    # 位置增量坐标系：与 curobo_reach_box_policy / 场景一致为世界系 (X,Y,Z)。
    # 若 XR 运行时为 Y-up、-Z 前，则「往前伸」会变成世界 Z，导致上下能动、往前卡住。可试 TELEOP_POS_FRAME=xr_yup_negz_fwd
    teleop_pos_frame = os.environ.get("TELEOP_POS_FRAME", "").strip().lower()

    env_cfg = register.load_env_configs(args_cli.task_scene_name)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # 显式传入 scale 重建 teleop_devices；IK scale 直接写回 actions
    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device == "handtracking":
        from isaaclab_logistics_vla.configs.teleop_configs.realman_xr_handtracking import (
            build_realman_xr_handtracking_devices_cfg,
        )
        env_cfg.teleop_devices = build_realman_xr_handtracking_devices_cfg(
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
            pos_scale=teleop_pos_scale,
            rot_scale=teleop_rot_scale,
        )
    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "left_arm_ik"):
        env_cfg.actions.left_arm_ik.scale = teleop_ik_scale
        env_cfg.actions.right_arm_ik.scale = teleop_ik_scale

    # 通过环境变量覆盖 XR 视角（anchor），无需改代码即可试不同视角
    # TELEOP_XR_ANCHOR_POS=x,y,z  例如 0,0,-1.2 表示场景在眼前约 1.2m
    # TELEOP_XR_ANCHOR_ROT=w,x,y,z 四元数。常用：顶视 (0,1,0,0) 正前 (1,0,0,0) 侧视 (0.707,0,0.707,0)
    if hasattr(env_cfg, "xr") and env_cfg.xr is not None:
        from isaaclab.devices.openxr import XrCfg
        pos_s = os.environ.get("TELEOP_XR_ANCHOR_POS", "").strip()
        rot_s = os.environ.get("TELEOP_XR_ANCHOR_ROT", "").strip()
        if pos_s or rot_s:
            xr = env_cfg.xr
            new_pos = xr.anchor_pos
            new_rot = xr.anchor_rot
            if pos_s:
                parts = [float(p.strip()) for p in pos_s.split(",")]
                if len(parts) == 3:
                    new_pos = tuple(parts)
            if rot_s:
                parts = [float(p.strip()) for p in rot_s.split(",")]
                if len(parts) == 4:
                    new_rot = tuple(parts)
            env_cfg.xr = XrCfg(anchor_pos=new_pos, anchor_rot=new_rot, near_plane=getattr(xr, "near_plane", 0.15))
            print(f"[XR Teleop] 已用环境变量覆盖 XR anchor: pos={new_pos} rot={new_rot}")

    # 创建环境
    env = VLAIsaacEnv(cfg=env_cfg).unwrapped

    # XR 手部遥操作设备
    from isaaclab.devices.teleop_device_factory import create_teleop_device
    from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
    from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeter, GripperRetargeterCfg

    # 可选：运行时调参（不改代码）
    # TELEOP_POS_SCALE / TELEOP_ROT_SCALE 会覆盖 Se3RelRetargeterCfg 的 delta_*_scale_factor
    pos_scale_s = os.environ.get("TELEOP_POS_SCALE", "").strip()
    rot_scale_s = os.environ.get("TELEOP_ROT_SCALE", "").strip()
    if pos_scale_s or rot_scale_s:
        try:
            pos_scale = float(pos_scale_s) if pos_scale_s else None
            rot_scale = float(rot_scale_s) if rot_scale_s else None
        except ValueError:
            pos_scale = rot_scale = None
        if pos_scale is not None or rot_scale is not None:
            try:
                dev_cfg = env_cfg.teleop_devices.devices.get("handtracking", None)
                if dev_cfg is not None and getattr(dev_cfg, "retargeters", None):
                    for r in dev_cfg.retargeters:
                        if pos_scale is not None and hasattr(r, "delta_pos_scale_factor"):
                            r.delta_pos_scale_factor = pos_scale
                        if rot_scale is not None and hasattr(r, "delta_rot_scale_factor"):
                            r.delta_rot_scale_factor = rot_scale
                    print(
                        f"[XR Teleop] 已覆盖 retargeter scale: pos={pos_scale if pos_scale is not None else '(default)'} "
                        f"rot={rot_scale if rot_scale is not None else '(default)'}"
                    )
            except Exception:
                pass

    # 调试：若 AVP 点 Play 后仍无 START 回调，可设 TELEOP_FORCE_ACTIVE=1 强制开启遥操作，测试手部数据是否有输出
    teleoperation_active = os.environ.get("TELEOP_FORCE_ACTIVE", "").strip() in ("1", "true", "yes")
    if teleoperation_active:
        print("[XR Teleop] TELEOP_FORCE_ACTIVE=1，遥操作已强制开启（用于排查 Play 未生效）")
    should_reset = False

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("[XR Teleop] RESET")

    # 点 Play 后烧掉一帧：用当前手姿作为 retargeter 的 previous，避免第一帧 delta 巨大导致手臂交缠
    skip_first_frame_after_start = [True]

    def start_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        skip_first_frame_after_start[0] = True
        print("[XR Teleop] START（下一帧将用作零点，机器人不会动）")

    def stop_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("[XR Teleop] STOP —— 遥操作已暂停，机器人将不再跟随。请在 AVP 上再次点击 Play 恢复控制。")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "START": start_teleop,
        "STOP": stop_teleop,
        "RESET": reset_env,
        "R": reset_env,
    }

    use_custom = os.environ.get("TELEOP_USE_CUSTOM_RETARGETER", "").strip().lower() in ("1", "true", "yes")
    if use_custom:
        # 使用自定义 Realman retargeter，绕过 factory（scale 已在上方读取为 teleop_pos_scale / teleop_rot_scale）
        from isaaclab_logistics_vla.teleop.retargeters.realman_se3_rel_retargeter import (
            RealmanSe3RelRetargeterCfg,
            RealmanSe3RelRetargeter,
        )

        dev_cfg = OpenXRDeviceCfg(xr_cfg=env_cfg.xr)
        left_cfg = RealmanSe3RelRetargeterCfg(
            bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT,
            sim_device=env_cfg.sim.device,
            delta_pos_scale_factor=teleop_pos_scale,
            delta_rot_scale_factor=teleop_rot_scale,
        )
        right_cfg = RealmanSe3RelRetargeterCfg(
            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
            sim_device=env_cfg.sim.device,
            delta_pos_scale_factor=teleop_pos_scale,
            delta_rot_scale_factor=teleop_rot_scale,
        )
        left = RealmanSe3RelRetargeter(left_cfg)
        right = RealmanSe3RelRetargeter(right_cfg)
        grip_left = GripperRetargeter(
            GripperRetargeterCfg(bound_hand=OpenXRDevice.TrackingTarget.HAND_LEFT, sim_device=env_cfg.sim.device)
        )
        grip_right = GripperRetargeter(
            GripperRetargeterCfg(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, sim_device=env_cfg.sim.device)
        )
        # 顺序须与 env 一致：left_arm, right_arm, left_gripper, right_gripper
        teleop_interface = OpenXRDevice(cfg=dev_cfg, retargeters=[left, right, grip_left, grip_right])
        # 注册 XR UI 的 START/STOP/RESET 回调
        for key, cb in teleoperation_callbacks.items():
            if key in ("START", "STOP", "RESET"):
                teleop_interface.add_callback(key, cb)
        print("[XR Teleop] 使用自定义 Realman retargeter (TELEOP_USE_CUSTOM_RETARGETER=1)")
    else:
        if not hasattr(env_cfg, "teleop_devices") or args_cli.teleop_device not in env_cfg.teleop_devices.devices:
            omni.log.error(
                f"[XR Teleop] env_cfg.teleop_devices 缺少 '{args_cli.teleop_device}' 配置。"
                "请使用 *_XRTeleop_EnvCfg，或在你的 EnvCfg 中添加 teleop_devices.handtracking。"
            )
            env.close()
            return
        teleop_interface = create_teleop_device(
            args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
        )
        print(f"[XR Teleop] Using teleop device: {teleop_interface}")
    print("[XR Teleop] 请在 AVP 上点击 Play 后再动手指，否则机器人不会跟随。")
    print("[XR Teleop] 若机器人一直不动：1) 确认 AVP 上为 Play 状态（非 Stop）；2) 可设 TELEOP_FORCE_ACTIVE=1 强制开启遥操作以排查。")
    print(
        "[XR Teleop] 动作幅度：pos_scale=%.1f rot_scale=%.1f ik_scale=%.1f（可用 --teleop_pos_scale 等或环境变量）"
        % (teleop_pos_scale, teleop_rot_scale, teleop_ik_scale)
    )
    if teleop_pos_scale_x != 1.0 or teleop_pos_scale_y != 1.0 or teleop_pos_scale_z != 1.0:
        print(
            "[XR Teleop] 位置分轴缩放：X=%.1f Y=%.1f Z=%.1f（若往前伸卡住可试 TELEOP_POS_SCALE_X/Y=2）"
            % (teleop_pos_scale_x, teleop_pos_scale_y, teleop_pos_scale_z)
        )
    if teleop_pos_frame:
        print(
            "[XR Teleop] 位置增量坐标系: TELEOP_POS_FRAME=%s（与 curobo/场景世界系对齐，解决往前伸卡住）"
            % (teleop_pos_frame,)
        )
    if os.environ.get("TELEOP_DEBUG_RAW", "").strip() in ("1", "true", "yes"):
        print("[XR Teleop] TELEOP_DEBUG_RAW=1：将打印 OpenXR 原始 wrist/palm 位姿变化（若可用）")

    env.reset()
    teleop_interface.reset()

    # realman 有 platform_joint（升降柱），teleop 只输出双臂+夹爪，需补平台目标（与 init_state 一致）
    try:
        action_dim = env.unwrapped.action_manager.total_action_dim
        print(f"[XR Teleop] env action_dim={action_dim}, teleop 输出 dim 将在此后首帧打印")
    except Exception:
        action_dim = None
    PLATFORM_JOINT_DEFAULT = 0.8  # 与 realman_config.py init_state platform_joint 一致

    _logged_action_shape = False
    _last_debug_time = [time.perf_counter()]  # 用 list 以便在 closure 里更新
    _last_paused_reminder = [time.perf_counter()]
    dt = 1.0 / float(args_cli.control_hz)

    while simulation_app.is_running():
        with torch.inference_mode():
            try:
                action = teleop_interface.advance()
            except Exception as e:
                print(f"[XR Teleop] advance() 异常（可能导致 TF_PYTHON_EXCEPTION）: {e}")
                import traceback
                traceback.print_exc()
                action = torch.zeros(1, action_dim or 14, device=env.unwrapped.device, dtype=torch.float32)

            if teleoperation_active:
                # START 后烧掉一帧：advance() 已在上方调用，retargeter 的 previous 已更新，本帧用零动作 step 一次
                if skip_first_frame_after_start[0]:
                    dim = action_dim if action_dim is not None else action.shape[1]
                    zero_actions = torch.zeros(env.num_envs, dim, device=env.unwrapped.device, dtype=torch.float32)
                    if action_dim is not None and dim >= 1:
                        zero_actions[:, -1] = PLATFORM_JOINT_DEFAULT
                    try:
                        env.step(zero_actions)
                    except Exception as e:
                        print(f"[XR Teleop] 首帧零 step 异常: {e}")
                    skip_first_frame_after_start[0] = False
                else:
                    if not _logged_action_shape:
                        print(f"[XR Teleop] teleop advance() shape={action.shape}, action_dim={action_dim}")
                        _logged_action_shape = True
                    # 每约 2 秒打印一次 action 范数，若始终为 0 说明手部数据未传到本机
                    t = time.perf_counter()
                    if t - _last_debug_time[0] >= 2.0:
                        _last_debug_time[0] = t
                        an = float(action.abs().sum()) if action.numel() else 0.0
                        # 尝试拆分：两臂(前12维) + 两夹爪(后2维) 的绝对和
                        arm_sum = float(action[:12].abs().sum()) if action.numel() >= 12 else 0.0
                        grip_sum = float(action[12:14].abs().sum()) if action.numel() >= 14 else 0.0
                        print(
                            f"[XR Teleop] active=1 arm_sum_abs={arm_sum:.4f} grip_sum_abs={grip_sum:.4f} "
                            f"total={an:.4f} (若 arm≈0 且 grip≈2，通常是手部位姿未进来，只剩夹爪默认值)"
                        )
                        # 可选：打印 raw wrist/palm 位姿是否在变化
                        if os.environ.get("TELEOP_DEBUG_RAW", "").strip() in ("1", "true", "yes"):
                            try:
                                raw = teleop_interface._get_raw_data()  # type: ignore[attr-defined]
                                # raw: {HAND_LEFT: {joint: pose7}, HAND_RIGHT: {...}, HEAD: pose7}
                                def _joint_pos(d, name):
                                    p = d.get(name, None)
                                    if p is None:
                                        return None
                                    return tuple(float(x) for x in p[:3])
                                l = raw.get(getattr(teleop_interface, "TrackingTarget").HAND_LEFT, {})
                                r = raw.get(getattr(teleop_interface, "TrackingTarget").HAND_RIGHT, {})
                                lw = _joint_pos(l, "wrist")
                                lp = _joint_pos(l, "palm")
                                rw = _joint_pos(r, "wrist")
                                rp = _joint_pos(r, "palm")
                                print(f"[XR Teleop] raw left(wrist,palm)={lw},{lp} right(wrist,palm)={rw},{rp}")
                            except Exception:
                                pass
                    # 位置增量坐标系转换：与 curobo/场景世界系一致（见 curobo_reach_box_policy 世界系、planner 臂基系纯减法）
                    # 若 XR 为 Y-up、-Z=前，则「往前伸」会变成世界 Z，导致上下能动、往前卡住。试 TELEOP_POS_FRAME=xr_yup_negz_fwd
                    if teleop_pos_frame == "xr_yup_negz_fwd" and action.numel() >= 9:
                        # XR (right, up, back) → world (forward, left, up): world = (-xr_z, xr_x, xr_y)
                        a = action.clone()
                        for start in (0, 6):
                            if start + 3 <= action.numel():
                                v = a[start : start + 3]
                                a[start] = -v[2]
                                a[start + 1] = v[0]
                                a[start + 2] = v[1]
                        action = a

                    # Realman 专用：参考 lerobot-realman-vla 的 Vive→Robot 轴映射做一个可选后处理
                    # 映射矩阵默认 identity；若设 TELEOP_REALMAN_AXIS_MAP=lerobot 则使用 [-z,-x,+y]
                    axis_map = os.environ.get("TELEOP_REALMAN_AXIS_MAP", "").strip().lower()
                    if axis_map == "lerobot" and action.numel() >= 12:
                        # 对两臂的 (dx,dy,dz) 和 (rx,ry,rz) 分别做同样轴映射
                        def _map3(v: torch.Tensor) -> torch.Tensor:
                            return torch.stack([-v[2], -v[0], v[1]])
                        a = action.clone()
                        a[0:3] = _map3(a[0:3])
                        a[3:6] = _map3(a[3:6])
                        a[6:9] = _map3(a[6:9])
                        a[9:12] = _map3(a[9:12])
                        action = a

                    # 分轴位置缩放：解决「上下能动、往前伸卡住」— 可试 TELEOP_POS_SCALE_X/Y=2（世界系）
                    if (teleop_pos_scale_x != 1.0 or teleop_pos_scale_y != 1.0 or teleop_pos_scale_z != 1.0) and action.numel() >= 9:
                        scale_xyz = torch.tensor(
                            [teleop_pos_scale_x, teleop_pos_scale_y, teleop_pos_scale_z],
                            device=action.device,
                            dtype=action.dtype,
                        )
                        action = action.clone()
                        action[0:3] = action[0:3] * scale_xyz
                        action[6:9] = action[6:9] * scale_xyz

                    actions = action.repeat(env.num_envs, 1)
                    if action_dim is not None and actions.shape[1] < action_dim:
                        pad = torch.full(
                            (actions.shape[0], action_dim - actions.shape[1]),
                            PLATFORM_JOINT_DEFAULT,
                            device=actions.device,
                            dtype=actions.dtype,
                        )
                        actions = torch.cat([actions, pad], dim=1)
                    if not (torch.isfinite(actions).all()):
                        pass  # 跳过含 NaN/Inf 的帧，避免仿真炸
                    else:
                        try:
                            env.step(actions)
                        except Exception as e:
                            print(f"[XR Teleop] env.step 异常: {e}")
            else:
                # 未在 Play 状态时只渲染，机器人不跟随；每隔约 5 秒提醒一次
                t = time.perf_counter()
                if t - _last_paused_reminder[0] >= 5.0:
                    _last_paused_reminder[0] = t
                    print("[XR Teleop] 当前为 STOP 状态，机器人未跟随。请在 AVP 上点击 Play 恢复控制。")
                env.sim.render()

            if should_reset:
                env.reset()
                should_reset = False

        time.sleep(dt)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

