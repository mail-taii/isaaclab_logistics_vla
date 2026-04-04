"""
VLA 评估驱动：只负责创建环境、循环 step、收集/打印指标。
动作由外部注入的 policy 生成，评估器不包含任何策略逻辑。
策略收到的 obs 为 ObservationBuilder 产出的 ObservationDict（meta / robot_state / vision / point_cloud），
而非 env 原生的 group 观测。
"""

from .VLAIsaacEnv import VLAIsaacEnv
import os
import torch
import numpy as np
import time
import imageio
from pathlib import Path

from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab_logistics_vla.evaluation.observation.builder import EpisodeContext, ObservationBuilder
from isaaclab_logistics_vla.evaluation.observation.schema import ObservationRequire
from isaaclab_logistics_vla.evaluation.robot_registry import get_robot_eval_config

import isaaclab_logistics_vla
from isaaclab_logistics_vla.evaluation.curobo.planner import (
    WorldMode,
    CuroboPlanner,
)


def _get_action_from_policy(policy, obs):
    """从策略得到动作：支持 predict(obs) 返回 tensor，或 __call__(obs) 返回 ActionDict。"""
    if hasattr(policy, "predict"):
        out = policy.predict(obs, **{})
        return out if isinstance(out, torch.Tensor) else out.get("action", out)
    out = policy(obs)
    if isinstance(out, dict) and "action" in out:
        return out["action"]
    return out


def _make_policy_from_name(
    policy_name: str,
    env,
    trajectory_path: str = None,
    robot_eval_cfg=None,
):
    """根据名称创建策略实例，用于脚本传 --policy 字符串时。
    robot_eval_cfg: 可选，RobotEvalConfig；用于 OpenVLA 等策略时传入该机器人的 unnorm_key。
    """
    action_dim = env.unwrapped.action_manager.total_action_dim
    device = env.device
    if policy_name == "random":
        from isaaclab_logistics_vla.evaluation.models.policy.random_policy import RandomPolicy
        return RandomPolicy(action_dim=action_dim, device=device)
    if policy_name in ("openpi", "pi0", "openpi_remote"):
        from isaaclab_logistics_vla.evaluation.models.policy.openpi_remote_policy import OpenPIRemotePolicy
        return OpenPIRemotePolicy(action_dim=action_dim, device=device)
    if policy_name == "openvla_stub":
        # 仅本地调试用，不做远程调用
        from isaaclab_logistics_vla.evaluation.models.policy.openvla_stub_policy import OpenVLAStubPolicy
        return OpenVLAStubPolicy(action_dim=action_dim, device=device)
    if policy_name == "openvla":
        # 真实远程 OpenVLA：走 deploy.py 的 HTTP /act 接口；unnorm_key 按 robot_id 从注册表取，与动作维度对应
        from isaaclab_logistics_vla.evaluation.models.policy.openvla_remote_policy import OpenVLARemotePolicy
        unnorm_key = getattr(robot_eval_cfg, "unnorm_key", None) if robot_eval_cfg else None
        if unnorm_key is None:
            unnorm_key = "bridge_orig"
        return OpenVLARemotePolicy(action_dim=action_dim, device=device, unnorm_key=unnorm_key)
    if policy_name in ("rrt", "trajectory"):
        from isaaclab_logistics_vla.evaluation.models.policy.trajectory_playback_policy import TrajectoryPlaybackPolicy
        path = trajectory_path or "/home/wst/code/ompl/RRT_path.txt"
        return TrajectoryPlaybackPolicy(txt_path=path, device=device, action_dim=action_dim, lift_duration=250)
    if policy_name in ("curobo_reach_box", "reach_box"):
        # 使用 Curobo 规划左臂到前方箱子中点的简单 demo 策略
        from isaaclab_logistics_vla.evaluation.models.policy.curobo_reach_box_policy import (
            CuroboReachBoxPolicy,
        )

        if robot_eval_cfg is None:
            from isaaclab_logistics_vla.evaluation.robot_registry import get_robot_eval_config

            robot_eval_cfg = get_robot_eval_config("realman_dual_left_arm")
        platform_joint_index = None
        try:
            scene = getattr(env.unwrapped, "scene", None)
            robot = scene["robot"] if scene is not None else None
            if (
                robot is not None
                and hasattr(robot, "data")
                and hasattr(robot.data, "joint_names")
                and getattr(robot_eval_cfg, "platform_joint_name", None)
            ):
                platform_joint_index = list(robot.data.joint_names).index(robot_eval_cfg.platform_joint_name)
        except (ValueError, AttributeError):
            pass
        return CuroboReachBoxPolicy(
            action_dim=action_dim,
            device=device,
            robot_eval_cfg=robot_eval_cfg,
            platform_joint_index=platform_joint_index,
        )
    raise ValueError(f"Unknown policy name: {policy_name!r}. Use 'random', 'rrt', or 'trajectory'.")


class VLA_Evaluator:
    """纯驱动：持有一个 env 和一个 policy，run_evaluation 里只做 reset → 循环(取 obs → policy 得 action → step)。"""

    def __init__(
        self,
        env_cfg,
        policy,
        trajectory_path: str = None,
        record_video: bool = True,
        video_output_dir: str = "./videos",
        robot_id: str = "realman_dual_left_arm",
        from_json: int = 0,
    ):
        """
        Args:
            env_cfg: 环境配置（如 OrderEnvCfg()）
            policy: 策略实例，或策略名称。
            trajectory_path: 轨迹路径
            record_video: 是否录制视频
            video_output_dir: 视频输出目录
            robot_id: 评估侧机器人 ID，用于从 robot_registry 取 arm_dof / 平台关节 / Curobo 配置等。
                见 evaluation/robot_registry.py，新机器人需在 REGISTRY 中注册。
            from_json: 0=记录 JSON，1=回放 JSON，2=纯随机（与 scripts/evaluate_vla.py --from_json 对应）。
        """
        self.env = VLAIsaacEnv(cfg=env_cfg)
        # 说明：场景中的机器人由 env_cfg.scene 加载（OrderSceneCfg → register.load_robot('realman_franka_ee')），
        # 与 robot_id/robot_registry 无关。robot_registry 仅用于评估侧 IK、arm_dof、平台关节名等。
        _pkg_dir = Path(isaaclab_logistics_vla.__file__).resolve().parent
        _scene_robot_usd = _pkg_dir / "assets" / "robots" / "realman" / "realman_franka_ee.usd"
        print(f"[INFO] 场景机器人由 env_cfg 加载，USD 路径: {_scene_robot_usd} (存在: {_scene_robot_usd.exists()})")
        self._robot_eval_cfg = get_robot_eval_config(robot_id)
        if isinstance(policy, str):
            self.policy = _make_policy_from_name(
                policy, self.env, trajectory_path, robot_eval_cfg=self._robot_eval_cfg
            )
        else:
            self.policy = policy
        self.isprint = False
        self.from_json = from_json  # 0: 记录 JSON, 1: 回放 JSON, 2: 纯随机

        # Observation Builder
        self._obs_builder = ObservationBuilder(self.env)
        self._obs_require = ObservationRequire(
            require_rgb=True,
            require_depth=True,
            require_seg=True,
            require_pcd=False,
            pcd_frame="camera",
        )
        sensors = getattr(self.env.unwrapped.scene, "sensors", {})
        self._camera_names = sorted([n for n in sensors.keys() if "camera" in n.lower()]) or None
        
        # 视频录制设置
        self.record_video = record_video
        self.video_writers = {}
        self.video_output_dir = Path(video_output_dir)
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.video_initialized = False

        self._curobo_planner: CuroboPlanner | None = None
        self.arm_dof = self._robot_eval_cfg.arm_dof

        if (
            self._robot_eval_cfg.curobo_yml_name
            and self._robot_eval_cfg.curobo_asset_folder
            and self._robot_eval_cfg.curobo_urdf_name
        ):
            try:
                use_mesh = os.environ.get("CUROBO_USE_MESH_OBSTACLES", "").lower() in ("1", "true", "yes")
                use_hollow_box = os.environ.get("CUROBO_HOLLOW_BOX", "").lower() in ("1", "true", "yes")
                if use_mesh and use_hollow_box:
                    world_mode: WorldMode = "boxes_hollow"
                    print(
                        f"🔄 初始化 CuroboPlanner (robot_id={robot_id})，障碍物: 空心箱..."
                    )
                elif use_mesh:
                    world_mode = "boxes_mesh"
                    print(
                        f"🔄 初始化 CuroboPlanner (robot_id={robot_id})，障碍物: 箱子(mesh)..."
                    )
                else:
                    world_mode = "table_only"
                    print(
                        f"🔄 初始化 CuroboPlanner (robot_id={robot_id})，障碍物: 仅桌子..."
                    )

                # 与 policy 的 Curobo 使用同一 GPU，避免 retract_config 等张量跨设备
                _curobo_dev = os.environ.get("CUROBO_DEVICE")
                if _curobo_dev is not None:
                    _curobo_dev = (
                        torch.device(_curobo_dev)
                        if isinstance(_curobo_dev, str)
                        else _curobo_dev
                    )
                else:
                    _curobo_dev = self.env.device

                self._curobo_planner = CuroboPlanner(
                    self._robot_eval_cfg,
                    curobo_device=_curobo_dev,
                    world_mode=world_mode,
                    logger_name=f"evaluator_curobo_{robot_id}",
                )
            except Exception as e:
                print(f"❌ CuroboPlanner 初始化失败: {e}")
                self._curobo_planner = None
        else:
            print(
                f"[INFO] robot_id={robot_id} 未配置 Curobo（curobo_yml_name/asset/urdf 为空），EE 模式不可用。"
            )

    def _init_video_writers(self, obs_dict):
        """初始化视频写入器"""
        if not self.record_video:
            return
        
        try:
            # 从obs_dict获取vision数据
            vision = obs_dict.get("vision", {})
            cameras = vision.get("cameras", [])
            rgb = vision.get("rgb", None)

            if rgb is None or len(cameras) == 0:
                if not getattr(self, "_video_init_warned", False):
                    self._video_init_warned = True
                    reason = "vision 缺失" if not vision else ("cameras 为空" if not cameras else "rgb 为 None")
                    print(f"⚠️ 视频录制未初始化: {reason} (obs_dict keys={list(obs_dict.keys())}, vision keys={list(vision.keys()) if vision else []})")
                return
            
            if rgb is not None and len(cameras) > 0:
                # 获取时间戳
                timestamp = int(time.time())
                fps = 20  # 录制帧率
                
                # 为每个相机创建视频写入器
                for cam_idx, cam_name in enumerate(cameras):
                    # 获取图像形状
                    height, width = rgb.shape[2], rgb.shape[3]  # (相机数, 环境数, 高度, 宽度, 通道)
                    
                    # 创建视频文件名
                    video_filename = f"{cam_name}_{timestamp}.mp4"
                    video_path = self.video_output_dir / video_filename
                    
                    # 使用imageio创建视频写入器
                    video_writer = imageio.get_writer(
                        str(video_path),
                        fps=fps,
                        codec='libx264',
                        quality=9
                    )
                    
                    self.video_writers[cam_name] = video_writer
                    print(f"🎥 {cam_name} 视频录制已初始化: {video_path}")
                    print(f"📹 录制参数: {width}x{height}, {fps}fps")
                
                self.video_initialized = True
        except Exception as e:
            print(f"⚠️ 视频写入器初始化失败: {e}")
            import traceback
            traceback.print_exc()

    def _record_frame_from_obs(self, obs_dict):
        """从obs_dict录制视频帧"""
        if not self.record_video:
            return
        
        # 初始化视频写入器
        if not self.video_initialized:
            self._init_video_writers(obs_dict)
            if not self.video_initialized:
                return
        
        try:
            # 从obs_dict获取vision数据
            vision = obs_dict.get("vision", {})
            cameras = vision.get("cameras", [])
            rgb = vision.get("rgb", None)
            
            if rgb is not None and len(cameras) > 0:
                # 为每个相机录制帧
                for cam_idx, cam_name in enumerate(cameras):
                    if cam_name in self.video_writers:
                        # 获取当前相机的图像
                        frame = rgb[cam_idx, 0].cpu().numpy()  # (H, W, 3)
                        
                        # 确保图像数据类型正确
                        if frame.dtype == np.float32:
                            frame = (frame * 255).astype(np.uint8)
                        elif frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        
                        # 写入视频
                        writer = self.video_writers[cam_name]
                        writer.append_data(frame)
        except Exception as e:
            print(f"⚠️ 视频录制错误: {e}")
            import traceback
            traceback.print_exc()

    def close_video_recording(self):
        """关闭视频录制"""
        if not self.video_writers:
            if self.record_video and not self.video_initialized:
                print("🎬 未录制视频（vision/cameras 初始化失败，请查看上方 ⚠️ 提示）")
            return
        for cam_name, writer in self.video_writers.items():
            if writer is not None:
                writer.close()
                print(f"🎬 {cam_name} 视频已保存至 {self.video_output_dir}")
        self.video_writers.clear()

    def _convert_actions_by_control_mode(self, actions, obs_dict):
        """
        根据策略的控制模式转换动作格式
        """
        control_mode = getattr(self.policy, "control_mode", "joint")
        
        if control_mode == "ee":
            # EE 模式必须要有 CuroboPlanner，否则报错
            if self._curobo_planner is None:
                raise RuntimeError(
                    "EE 模式下 CuroboPlanner 未初始化成功，无法将末端动作转为关节动作。"
                    " 请检查 configs/robot_configs/ 与 Curobo 依赖。"
                )

            # 获取 robot_state（缺 qpos 无法做 IK，直接报错）
            robot_state = obs_dict.get("robot_state", {})
            if not robot_state or "qpos" not in robot_state:
                raise RuntimeError(
                    "EE 模式下 obs_dict 缺少 robot_state.qpos，无法做 MotionGen 规划。"
                    " 请确保 ObservationBuilder 提供 robot_state。"
                )

            current_qpos = robot_state["qpos"]
            robot_data = self.env.unwrapped.scene.articulations["robot"].data

            # 左臂关节索引：若有 left_arm_joint_names 则从 joint_names 查找，否则用前 arm_dof 个
            left_arm_indices = None
            left_names = getattr(self._robot_eval_cfg, "left_arm_joint_names", None)
            if left_names and hasattr(robot_data, "joint_names"):
                all_names = list(robot_data.joint_names)
                try:
                    left_arm_indices = [all_names.index(n) for n in left_names]
                except ValueError:
                    pass
            if left_arm_indices is None:
                left_arm_indices = list(range(self.arm_dof))

            # 获取/计算 target_ee_pos
            
            # 获取当前末端位置与姿态（缺一不可，否则 IK 无意义）
            ee_pos = None
            ee_quat = None
            if hasattr(robot_data, "target_pos_w") and hasattr(robot_data, "body_state_w"):
                ee_pos = robot_data.target_pos_w[:, 0, :]
                ee_quat = robot_data.body_state_w[:, -1, 3:7]
            elif hasattr(robot_data, "body_state_w"):
                ee_pos = robot_data.body_state_w[:, -1, 0:3]
                ee_quat = robot_data.body_state_w[:, -1, 3:7]
            if ee_pos is None or ee_quat is None:
                raise RuntimeError(
                    "EE 模式下无法获取末端位姿（ee_pos / ee_quat）。"
                    " robot_data 需提供 target_pos_w + body_state_w 或 body_state_w。"
                )

            # 策略给出的末端位移增量（前 3 维，单位需与 ee_pos 一致，一般为米）
            ee_delta = actions[:, :3]
            target_ee_pos_w = ee_pos + ee_delta

            # Curobo 的 URDF 基座在原点，需把目标位置从 Isaac 世界系变换到臂基座系（含平台高度）
            root_pos_w = robot_data.root_pos_w[:, :3]
            root_quat_w = robot_data.root_quat_w
            # 臂基 = root + R_root @ offset；offset 来自 arm_base_offset_in_root，z 再加上 platform_joint
            arm_base_pos_w = root_pos_w.clone()
            offset_in_root = torch.zeros(
                root_pos_w.shape[0], 3,
                dtype=root_pos_w.dtype, device=root_pos_w.device
            )
            arm_offset = getattr(self._robot_eval_cfg, "arm_base_offset_in_root", None)
            if arm_offset is not None:
                offset_in_root[:, 0] = arm_offset[0]
                offset_in_root[:, 1] = arm_offset[1]
                offset_in_root[:, 2] = arm_offset[2]
            platform_joint_name = getattr(self._robot_eval_cfg, "platform_joint_name", None)
            if platform_joint_name and hasattr(robot_data, "joint_names") and platform_joint_name in robot_data.joint_names:
                platform_idx = list(robot_data.joint_names).index(platform_joint_name)
                platform_pos = robot_data.joint_pos[:, platform_idx]  # (num_envs,)
                offset_in_root[:, 2] = offset_in_root[:, 2] + platform_pos
            if arm_offset is not None or (platform_joint_name and hasattr(robot_data, "joint_names") and platform_joint_name in robot_data.joint_names):
                arm_base_pos_w, _ = combine_frame_transforms(
                    root_pos_w, root_quat_w, offset_in_root
                )
            # 位置与姿态都变换到臂基系（Curobo 期望位姿均在基座系）
            target_ee_pos_b, ee_quat_b = subtract_frame_transforms(
                arm_base_pos_w, root_quat_w, target_ee_pos_w, ee_quat
            )
            target_ee_pos = target_ee_pos_b
            ee_quat_for_pose = ee_quat_b

            # Debug：打印世界系与基座系下的末端、目标、当前关节角
            _ee_w = ee_pos[0].detach().cpu().numpy()
            _target_w = target_ee_pos_w[0].detach().cpu().numpy()
            _target_b = target_ee_pos_b[0].detach().cpu().numpy()
            _delta = ee_delta[0].detach().cpu().numpy()
            _q = current_qpos[0, : self.arm_dof].detach().cpu().numpy()
            print("[MotionGen] 当前末端位置 世界系 (m):", _ee_w.tolist())
            print("[MotionGen] 目标末端位置 世界系 (m):", _target_w.tolist())
            print("[MotionGen] 目标末端位置 基座系 (m):", _target_b.tolist())
            print("[MotionGen] 策略位移增量 actions[:, :3]:", _delta.tolist())
            print("[MotionGen] 当前左臂关节角 (rad):", _q.tolist())

            try:
                with torch.enable_grad():
                    arm_qpos = current_qpos[:, left_arm_indices].detach().clone()
                    q_start_left = arm_qpos[0]

                    result = self._curobo_planner.plan_ee(
                        q_start=q_start_left,
                        target_pos_b=target_ee_pos[0],
                        target_quat_b=ee_quat_for_pose[0],
                        max_attempts=10,
                        timeout=2.0,
                        enable_graph=True,
                        enable_opt=False,
                    )

                    if result["status"] == "Success" and result["position"] is not None:
                        # 取规划轨迹的最后一个点（目标关节角）作为本步动作
                        raw_plan_np = result["position"]
                        sol_left = torch.from_numpy(raw_plan_np[-1]).to(
                            device=actions.device, dtype=actions.dtype
                        )
                        new_actions = actions.clone()
                        for i, idx in enumerate(left_arm_indices):
                            if idx < new_actions.shape[1]:
                                new_actions[:, idx] = sol_left[i]
                        print("[CuroboPlanner] 规划成功，已应用目标关节角")
                        return new_actions

                    # 规划失败：跳过本步，保持当前关节角
                    detail = result.get("detail", "?")
                    print(f"[CuroboPlanner] 规划失败 (detail={detail})，跳过本步，保持当前关节角")
                    new_actions = actions.clone()
                    for i, idx in enumerate(left_arm_indices):
                        if idx < new_actions.shape[1]:
                            new_actions[:, idx] = current_qpos[:, idx]
                    return new_actions

            except RuntimeError:
                raise
            except Exception as e:
                import traceback
                print(f"[MotionGen] Curobo 规划异常: {e}")
                traceback.print_exc()
                err_detail = (
                    f"当前末端 (m): {_ee_w.tolist()}, 目标 (m): {_target_w.tolist()}, "
                    f"位移增量: {_delta.tolist()}, 当前关节 (rad): {_q.tolist()}"
                )
                raise RuntimeError(
                    f"EE 模式下 Curobo MotionGen 规划异常: {e}\n  {err_detail}"
                ) from e
        
        else:
            return actions

    def run_evaluation(self):
        step_i = 0
        episode_length = 0
        self.env.reset()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        ctx = EpisodeContext(task_name="order_series", episode_id=0)

        # 相机需要至少一次 step 后才输出首帧，先执行一步零动作以预热
        if self.record_video and self._camera_names:
            _zero = torch.zeros(
                (self.env.num_envs, self.env.unwrapped.action_manager.total_action_dim),
                device=self.env.device,
                dtype=torch.float32,
            )
            self.env.step(_zero)

        # 诊断 1：Isaac 仿真“绝对真理”——打印真实 Link 高度（root + 推算臂基）
        try:
            isaac_env = self.env.unwrapped
            robot = isaac_env.scene.articulations.get("robot")
            if robot is not None and hasattr(robot, "data"):
                root_pos = robot.data.root_pos_w[0, :3].cpu().numpy()
                root_quat = robot.data.root_quat_w[0].cpu().numpy()
                offset = getattr(self._robot_eval_cfg, "arm_base_offset_in_root", None)
                platform_val = 0.0
                if getattr(self._robot_eval_cfg, "platform_joint_name", None) and hasattr(robot.data, "joint_names"):
                    names = list(robot.data.joint_names)
                    if self._robot_eval_cfg.platform_joint_name in names:
                        idx = names.index(self._robot_eval_cfg.platform_joint_name)
                        platform_val = float(robot.data.joint_pos[0, idx].cpu().item())
                if offset is not None:
                    offset_t = torch.tensor(
                        [offset[0], offset[1], offset[2] + platform_val],
                        device=robot.data.root_pos_w.device, dtype=robot.data.root_pos_w.dtype,
                    ).unsqueeze(0)
                    arm_base_w, _ = combine_frame_transforms(
                        robot.data.root_pos_w[0:1, :3],
                        robot.data.root_quat_w[0:1],
                        offset_t,
                    )
                    arm_base = arm_base_w[0].cpu().numpy()
                else:
                    arm_base = root_pos
                print("\n[诊断] Isaac 仿真真实高度 (env 0):")
                print(f"  root_pos_w (x,y,z) = {root_pos.tolist()}  → 根高度 z = {root_pos[2]:.4f} m")
                print(f"  platform_joint = {platform_val:.4f}")
                print(f"  推算臂基 arm_base_pos_w (x,y,z) = {arm_base.tolist()}  → 臂基高度 z = {arm_base[2]:.4f} m")
                print("[诊断] 若臂基 z≈1.137 为叠加；若 z≈0.866 或 0.921 则与预期不符。\n")
        except Exception as e:
            print(f"[诊断] 打印 Isaac 高度失败: {e}")

        try:
            start_time = time.time()
            last_info = {}
            last_rew = torch.tensor(0.0, device=self.env.device)
            while True:
                with torch.no_grad():
                    obs_dict = self._obs_builder.build(
                        ctx=ctx,
                        step_id=step_i,
                        require=self._obs_require,
                        camera_names=self._camera_names,
                    )
                    actions = _get_action_from_policy(self.policy, obs_dict)
                    
                    # 转换动作 (内部会临时开启梯度)
                    actions = self._convert_actions_by_control_mode(actions, obs_dict)
                    
                    obs, rew, terminated, truncated, info = self.env.step(actions)
                    last_info = info
                    last_rew = rew

                # 从obs_dict录制视频帧
                self._record_frame_from_obs(obs_dict)

                step_i += 1
                episode_length += 1

                if step_i % 100 == 0:
                    # last_rew 可能是多环境 Tensor，这里取均值做日志
                    if isinstance(last_rew, torch.Tensor):
                        rew_scalar = float(last_rew.mean().item())
                    else:
                        rew_scalar = float(last_rew)
                    print(f"  step {step_i}: policy→action→env.step ok, reward={rew_scalar:.4f}")

                # 检查是否终止（terminated/truncated 是 Tensor，需要按 env 维度汇总）
                term_flag = bool(torch.any(terminated).item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
                trunc_flag = bool(torch.any(truncated).item()) if isinstance(truncated, torch.Tensor) else bool(truncated)
                if term_flag or trunc_flag:
                    print(f"\n🎯 Episode 终止: terminated={terminated}, truncated={truncated}")
                    break
                    
                if step_i % 1000 == 0:
                    isaac_env = self.env.unwrapped
                    robot_asset = isaac_env.scene.articulations["robot"]
                    default_state_tensor = robot_asset.data.root_state_w
                    print("\n" + "=" * 50)
                    print("Default Root State of 'robot' Asset:")
                    print(f"Shape: {default_state_tensor.shape}")
                    print(f"Data:\n{default_state_tensor[:, 0:3]}")
                    print(f"Reward :\n{rew}")
                    print("=" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n⏹️  评估被用户中断")
            # 立即关闭视频录制
            print("🎬 立即关闭视频录制")
            self.close_video_recording()
        except Exception as e:
            print(f"\n❌ 评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            # 立即关闭视频录制
            print("🎬 立即关闭视频录制")
            self.close_video_recording()
        finally:
            # 确保视频录制被关闭
            print("🎬 确保视频录制被关闭")
            self.close_video_recording()