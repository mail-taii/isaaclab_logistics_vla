"""
VLA 评估驱动：只负责创建环境、循环 step、收集/打印指标。
动作由外部注入的 policy 生成，评估器不包含任何策略逻辑。
策略收到的 obs 为 ObservationBuilder 产出的 ObservationDict（meta / robot_state / vision / point_cloud），
而非 env 原生的 group 观测。
"""

from .VLAIsaacEnv import VLAIsaacEnv
import torch
import numpy as np
import time
import imageio
from pathlib import Path

from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab_logistics_vla.evaluation.observation.builder import EpisodeContext, ObservationBuilder
from isaaclab_logistics_vla.evaluation.observation.schema import ObservationRequire
from isaaclab_logistics_vla.evaluation.result.saver import ResultSaver, EpisodeReport
from isaaclab_logistics_vla.evaluation.robot_registry import get_robot_eval_config
# Curobo 逆运动学求解器（配置从本包 configs/robot_configs/ 加载，不依赖 Curobo 安装路径）
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml

import isaaclab_logistics_vla


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
        
        # ResultSaver 初始化
        self.result_saver = ResultSaver(output_dir="./results")

    
        self.ik_solver = None
        self._retract_config_list = None
        self.arm_dof = self._robot_eval_cfg.arm_dof

        if self._robot_eval_cfg.curobo_yml_name and self._robot_eval_cfg.curobo_asset_folder and self._robot_eval_cfg.curobo_urdf_name:
            try:
                print(f"🔄 初始化 Curobo IK Solver (robot_id={robot_id})...")
                tensor_args = TensorDeviceType(device=self.env.device)
                _pkg_dir = Path(isaaclab_logistics_vla.__file__).resolve().parent
                _robot_configs_dir = _pkg_dir / "configs" / "robot_configs"
                _robot_yml = _robot_configs_dir / self._robot_eval_cfg.curobo_yml_name
                config_file = load_yaml(str(_robot_yml))
                _assets_dir = _pkg_dir / "assets" / "robots" / self._robot_eval_cfg.curobo_asset_folder
                config_file["robot_cfg"]["kinematics"]["urdf_path"] = str(_assets_dir / self._robot_eval_cfg.curobo_urdf_name)
                config_file["robot_cfg"]["kinematics"]["asset_root_path"] = str(_assets_dir)
                config_file["robot_cfg"]["kinematics"]["collision_spheres"] = str(
                    _robot_configs_dir / "spheres" / self._robot_eval_cfg.curobo_yml_name
                )
                robot_cfg = RobotConfig.from_dict(config_file["robot_cfg"], tensor_args)
                ik_config = IKSolverConfig.load_from_robot_config(
                    robot_cfg,
                    None,
                    rotation_threshold=0.05,
                    position_threshold=0.01,
                    num_seeds=32,
                    self_collision_check=True,
                    self_collision_opt=True,
                    tensor_args=tensor_args,
                    use_cuda_graph=True,
                )
                self.ik_solver = IKSolver(ik_config)
                self._retract_config_list = config_file["robot_cfg"]["kinematics"].get("cspace", {}).get("retract_config")
                print("✅ Curobo IK Solver 初始化完成")
            except Exception as e:
                print(f"❌ Curobo 初始化失败: {e}")
                self.ik_solver = None
                self._retract_config_list = None
        else:
            print(f"[INFO] robot_id={robot_id} 未配置 Curobo（curobo_yml_name/asset/urdf 为空），EE 模式不可用。")

    def _init_video_writers(self, obs_dict):
        """初始化视频写入器"""
        if not self.record_video:
            return
        
        try:
            # 从obs_dict获取vision数据
            vision = obs_dict.get("vision", {})
            cameras = vision.get("cameras", [])
            rgb = vision.get("rgb", None)
            
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
        for cam_name, writer in self.video_writers.items():
            if writer is not None:
                writer.close()
                print(f"🎬 {cam_name} 视频录制已完成")
        self.video_writers.clear()
    
    def _save_evaluation_result(self, start_time, episode_length, info, rew, terminated, truncated, ctx):
        """
        保存评估结果
        
        Args:
            start_time: 评估开始时间
            episode_length: episode 长度
            info: 环境返回的信息
            rew: 奖励
            terminated: 是否正常终止
            truncated: 是否被截断
            ctx:  episode 上下文
        """
        try:
            # 计算评估时间
            eval_time = time.time() - start_time
            
            # 构建 metrics_read（这里使用 info 作为示例，实际应从环境读取）
            metrics_read = info.get("metrics", {})
            if not metrics_read:
                # 如果没有 metrics，使用简单的奖励作为示例
                metrics_read = {"total_reward": float(rew.sum().item())}
            
            # 构建 timing 信息
            timing = {
                "episode_time": eval_time,
                "steps_per_second": episode_length / eval_time if eval_time > 0 else 0
            }
            
            # 构建并保存 episode 报告
            episode_report = EpisodeReport(
                episode_id=ctx.episode_id,
                seed=None,  # 可以从 env 中获取
                success=bool(terminated and not truncated),  # 假设 terminated 表示成功
                metrics_read=metrics_read,
                timing=timing,
                task_name=ctx.task_name,
                episode_length=episode_length
            )
            
            # 保存 episode 结果
            self.result_saver.write_episode(episode_report)
            
            # 生成并保存任务报告
            self.result_saver.write_task(task_name=ctx.task_name)
            
            print(f"\n📊 评估完成:")
            print(f"  - Episode 长度: {episode_length} 步")
            print(f"  - 评估时间: {eval_time:.2f} 秒")
            print(f"  - 成功率: {'✓' if episode_report.success else '✗'}")
            print(f"  - 奖励: {metrics_read.get('total_reward', 0):.2f}")
            print(f"  - 结果已保存到: {self.result_saver.output_dir}")
            
        except Exception as e:
            print(f"❌ 保存评估结果失败: {e}")
            import traceback
            traceback.print_exc()

    def _convert_actions_by_control_mode(self, actions, obs_dict):
        """
        根据策略的控制模式转换动作格式
        """
        control_mode = getattr(self.policy, "control_mode", "joint")
        
        if control_mode == "ee":
            # EE 模式必须要有 IK，否则报错（静默返回错误动作无意义）
            if not self.ik_solver:
                raise RuntimeError(
                    "EE 模式下 Curobo IK 未初始化成功，无法将末端动作转为关节动作。"
                    " 请检查 configs/robot_configs/ 与 Curobo 依赖。"
                )

            # 获取 robot_state（缺 qpos 无法做 IK，直接报错）
            robot_state = obs_dict.get("robot_state", {})
            if not robot_state or "qpos" not in robot_state:
                raise RuntimeError(
                    "EE 模式下 obs_dict 缺少 robot_state.qpos，无法做 IK。"
                    " 请确保 ObservationBuilder 提供 robot_state。"
                )

            current_qpos = robot_state["qpos"]
            
            # 获取/计算 target_ee_pos
            robot_data = self.env.unwrapped.scene.articulations["robot"].data
            
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
            # 若有可移动平台（从注册表取 platform_joint_name），臂基 = root + [0,0,platform_height]（root 系下）
            arm_base_pos_w = root_pos_w.clone()
            platform_joint_name = getattr(self._robot_eval_cfg, "platform_joint_name", None)
            if platform_joint_name and hasattr(robot_data, "joint_names") and platform_joint_name in robot_data.joint_names:
                platform_idx = list(robot_data.joint_names).index(platform_joint_name)
                platform_pos = robot_data.joint_pos[:, platform_idx]  # (num_envs,)
                offset_in_root = torch.zeros(
                    platform_pos.shape[0], 3,
                    dtype=root_pos_w.dtype, device=root_pos_w.device
                )
                offset_in_root[:, 2] = platform_pos
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
            _ee_b = target_ee_pos_b[0].detach().cpu().numpy()
            _delta = ee_delta[0].detach().cpu().numpy()
            _target_w = target_ee_pos_w[0].detach().cpu().numpy()
            _target_b = target_ee_pos_b[0].detach().cpu().numpy()
            _q = current_qpos[0, : self.arm_dof].detach().cpu().numpy()
            print("[IK] 当前末端位置 世界系 (m):", _ee_w.tolist())
            print("[IK] 目标末端位置 世界系 (m):", _target_w.tolist())
            print("[IK] 目标末端位置 基座系 (m):", _target_b.tolist())
            print("[IK] 策略位移增量 actions[:, :3]:", _delta.tolist())
            print("[IK] 当前左臂关节角 (rad):", _q.tolist())

            try:
                with torch.enable_grad():
                    target_pose = Pose(
                        target_ee_pos.detach().clone(),
                        ee_quat_for_pose.detach().clone()
                    )

                    arm_qpos = current_qpos[:, : self.arm_dof].detach().clone()
                    degenerate_threshold = 0.01
                    is_degenerate = (
                        arm_qpos.shape[1] >= 2
                        and (arm_qpos[:, 1:].abs() < degenerate_threshold).all().item()
                    )
                    # 准备多组 seed，依次尝试以提高收敛率
                    seeds_to_try = []
                    if getattr(self, "_retract_config_list", None) is not None:
                        retract = torch.tensor(
                            self._retract_config_list,
                            dtype=arm_qpos.dtype,
                            device=arm_qpos.device,
                        ).unsqueeze(0).unsqueeze(1)  # (1, 1, 7)
                        if target_ee_pos.is_cuda:
                            retract = retract.to(target_ee_pos.device)
                        seeds_to_try.append(("retract_config", retract))
                    if arm_qpos.dim() == 2:
                        seed_current = arm_qpos.unsqueeze(1)
                    else:
                        seed_current = arm_qpos
                    if target_ee_pos.is_cuda:
                        seed_current = seed_current.to(target_ee_pos.device)
                    seeds_to_try.append(("current_qpos", seed_current))
                    # 零位 seed（7 自由度）
                    zero_seed = torch.zeros(
                        1, 1, self.arm_dof,
                        dtype=arm_qpos.dtype, device=arm_qpos.device
                    )
                    if target_ee_pos.is_cuda:
                        zero_seed = zero_seed.to(target_ee_pos.device)
                    seeds_to_try.append(("zero", zero_seed))

                    result = None
                    used_seed_name = None
                    print(f"[IK] 依次尝试 {len(seeds_to_try)} 组 seed: {[s[0] for s in seeds_to_try]}")
                    for seed_name, seed_input in seeds_to_try:
                        print(f"[IK] 尝试 seed={seed_name} ...", end=" ", flush=True)
                        result = self.ik_solver.solve_single(
                            target_pose,
                            seed_config=seed_input,
                            retract_config=seed_input
                        )
                        if result.success.item():
                            used_seed_name = seed_name
                            print("收敛")
                            break
                        print("未收敛")
                    if is_degenerate and used_seed_name:
                        print(f"[IK] 当前关节构型退化，使用 seed={used_seed_name} 收敛")

                    if result is not None and result.success.item():
                        # 当前环境 Curobo：result.solution 直接为关节解 Tensor（非 JointState.position）
                        sol_qpos = result.solution.detach()
                        if sol_qpos.dim() == 3:
                            sol_qpos = sol_qpos.squeeze(1)
                        new_actions = actions.clone()
                        if new_actions.shape[1] >= self.arm_dof:
                            new_actions[:, : self.arm_dof] = sol_qpos
                            return new_actions
                        raise RuntimeError(
                            f"IK 成功但 action 维度不足: new_actions.shape[1]={new_actions.shape[1]}, arm_dof={self.arm_dof}"
                        )

                    # 多组 seed 均未收敛：跳过本步，保持当前关节角，继续下一帧
                    print(f"[IK] 已尝试 {len(seeds_to_try)} 组 seed，均未收敛；跳过本步，保持当前关节角继续下一帧")
                    new_actions = actions.clone()
                    new_actions[:, : self.arm_dof] = current_qpos[:, : self.arm_dof]
                    return new_actions

            except RuntimeError:
                raise
            except Exception as e:
                import traceback
                print(f"[IK] Curobo 求解异常: {e}")
                traceback.print_exc()
                err_detail = (
                    f"当前末端 (m): {_ee_w.tolist()}, 目标 (m): {_target_w.tolist()}, "
                    f"位移增量: {_delta.tolist()}, 当前关节 (rad): {_q.tolist()}"
                )
                raise RuntimeError(
                    f"EE 模式下 Curobo IK 求解异常: {e}\n  {err_detail}"
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
                    print(f"  step {step_i}: policy→action→env.step ok, reward={last_rew.item():.4f}")

                # 检查是否终止
                if terminated or truncated:
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
            
            # 保存结果
            self._save_evaluation_result(start_time, episode_length, last_info, last_rew, terminated, truncated, ctx)
            
        except KeyboardInterrupt:
            print("\n⏹️  评估被用户中断")
            # 保存中断时的结果
            self._save_evaluation_result(start_time, episode_length, last_info, last_rew, False, True, ctx)
            # 立即关闭视频录制
            print("🎬 立即关闭视频录制")
            self.close_video_recording()
        except Exception as e:
            print(f"\n❌ 评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            # 保存错误时的结果
            self._save_evaluation_result(start_time, episode_length, last_info, last_rew, False, True, ctx)
            # 立即关闭视频录制
            print("🎬 立即关闭视频录制")
            self.close_video_recording()
        finally:
            # 确保视频录制被关闭
            print("🎬 确保视频录制被关闭")
            self.close_video_recording()