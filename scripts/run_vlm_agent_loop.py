import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def _normalize_trace_path(trace_jsonl: str) -> str:
    """把 trace 输出收敛到固定目录，避免污染仓库根目录。

    规则：
    - 空字符串：不启用 trace 
    - 绝对路径：保持不变
    - 相对路径：
      - 若只给了文件名（无目录），则自动写到 ./traces/<filename>
      - 否则保持相对路径结构
    """
    s = (trace_jsonl or "").strip()
    if not s:
        return ""
    p = Path(s)
    if p.is_absolute():
        return str(p)
    if p.parent == Path("."):
        return str(Path("traces") / p.name)
    return str(p)


def _sanitize_filename(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return "untitled"
    # keep it filesystem-friendly
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    cleaned = "".join(out).strip("._")
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_len]


def _pick_current_task_label(tasks_dir: str) -> tuple[str, str]:
    """Return (task_id_str, task_subject) best representing 'current executing task'."""
    try:
        from isaaclab_logistics_vla.agent.task_system import TaskManager

        tm = TaskManager(Path(tasks_dir))
        tasks = tm.list_all()
        # Prefer in_progress, then ready (pending & unblocked), else nothing
        inprog = [t for t in tasks if getattr(t, "status", "") == "in_progress"]
        if inprog:
            t = inprog[0]
            return (str(t.id), str(t.subject))
        ready = [t for t in tasks if getattr(t, "status", "") == "pending" and not (t.blockedBy or [])]
        if ready:
            t = ready[0]
            return (str(t.id), str(t.subject))
    except Exception:
        pass
    return ("no_task", "no_task")


def _save_rgb_png(image_rgb_uint8, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        import imageio  # type: ignore

    imageio.imwrite(str(out_path), image_rgb_uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--cuda_linalg_backend",
        type=str,
        default="auto",
        choices=["auto", "cusolver", "magma"],
        help=(
            "选择 torch CUDA 线性代数后端（影响 torch.linalg.*）。"
            "auto=默认；遇到 cusolver 句柄错误会自动切换到 magma 重试一次。"
        ),
    )
    parser.add_argument(
        "--cuda_oom_fallback",
        type=str,
        default="reduce_then_cpu",
        choices=["none", "reduce_camera", "cpu", "reduce_then_cpu"],
        help=(
            "创建环境时若遇到 CUDA OOM 的兜底策略："
            "reduce_camera=降低顶视相机分辨率后重试；"
            "cpu=直接切到 --device cpu 重试；"
            "reduce_then_cpu=先降分辨率再切 CPU（默认）。"
        ),
    )
    parser.add_argument("--task_scene_name", type=str, default="Spawn_ss_st_sparse_EnvCfg")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="直接指定任务指令（为空则可用 --use_virtual_instruction 自动生成）。",
    )
    parser.add_argument(
        "--asset_root_path",
        type=str,
        default=os.environ.get("ASSET_ROOT_PATH", ""),
        help="资产根路径（必须）。例如 /home/junzhe/Benchmark",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument(
        "--backend",
        type=str,
        default="dummy",
        choices=["dummy", "volc_ark", "openai_compat"],
        help="dummy=本地假模型；volc_ark=火山 Ark Coding（Anthropic 兼容）；openai_compat=OpenAI 风格 Chat Completions 兼容 API。",
    )
    parser.add_argument(
        "--volc_api_key",
        type=str,
        default="",
        help="覆盖环境变量 ANTHROPIC_API_KEY / VOLC_ARK_API_KEY（不建议写进 shell 历史，优先用 env）",
    )
    parser.add_argument(
        "--anthropic_base_url",
        type=str,
        default="",
        help="Anthropic 兼容服务根 URL，默认 https://ark.cn-beijing.volces.com/api/coding",
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="",
        help="模型名（不同 backend 默认值不同）。",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="",
        help="openai_compat 后端的 API Key（优先级高于 OPENAI_API_KEY 环境变量）。",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default="",
        help="openai_compat 后端 base_url（例如 https://api.openai.com/v1 或你的兼容网关）。",
    )
    parser.add_argument("--out_json", type=str, default="", help="可选：把运行结果写入该 JSON 文件路径。")
    parser.add_argument(
        "--trace_jsonl",
        type=str,
        default="",
        help="可选：把每一步模型输出/解析/tool 调用与结果摘要按 JSONL 追加写入该文件（不保存图片本体，仅保存 shape/dtype 摘要）。",
    )
    parser.add_argument(
        "--top_camera_width",
        type=int,
        default=1000,
        help="顶视相机宽度（默认 640，降低显存占用；设为 0 表示不覆盖场景配置）。",
    )
    parser.add_argument(
        "--top_camera_height",
        type=int,
        default=1200,
        help="顶视相机高度（默认 480，降低显存占用；设为 0 表示不覆盖场景配置）。",
    )
    parser.add_argument(
        "--enable_task_system",
        action="store_true",
        default=False,
        help="启用 harness 级持久化任务系统 tools（task_create/update/list/get）。",
    )
    parser.add_argument(
        "--tasks_dir",
        type=str,
        default="/home/junzhe/isaaclab_logistics_vla/.tasks",
        help="任务系统持久化目录（每任务一个 JSON 文件）。仅在 enable_task_system 时生效。",
    )
    parser.add_argument(
        "--print_task_board",
        action="store_true",
        default=False,
        help="在控制台打印 TaskBoard（READY/IN_PROGRESS/BLOCKED/DONE）。仅在 enable_task_system 时生效。",
    )
    parser.add_argument(
        "--save_observation_images",
        action="store_true",
        default=False,
        help="开启后：每次 get_topview_image 成功返回时自动把顶视 PNG 写到 observation_images_dir（与模型提示词无关，无需模型提「保存」）。",
    )
    parser.add_argument(
        "--observation_images_dir",
        type=str,
        default="traces/images",
        help="保存顶视图的目录（相对路径默认放到 traces/images，已被 gitignore）。",
    )
    parser.add_argument(
        "--record_topview_video",
        action="store_true",
        default=False,
        help="开启后：从 agent 主循环开始用顶视相机持续录 MP4，直到循环结束（与模型是否调用 get_topview_image 无关）。",
    )
    parser.add_argument(
        "--topview_video_path",
        type=str,
        default="",
        help="顶视录制输出路径（mp4）。留空则自动写入 traces/videos/agent_topview_时间戳.mp4。",
    )
    parser.add_argument(
        "--topview_video_fps",
        type=float,
        default=10.0,
        help="输出视频的标称帧率（每完成一轮 agent 循环追加一帧，非真实墙钟时间轴）。",
    )
    args_cli, _ = parser.parse_known_args()
    args_cli.trace_jsonl = _normalize_trace_path(args_cli.trace_jsonl)

    # 在无图形界面环境中自动切换 headless，避免 GLFW 初始化失败导致的噪声报错
    if not args_cli.headless:
        display = os.environ.get("DISPLAY", "").strip()
        wayland = os.environ.get("WAYLAND_DISPLAY", "").strip()
        if not display and not wayland:
            print("[Warn] No DISPLAY/WAYLAND detected; forcing --headless to avoid windowing errors.")
            args_cli.headless = True

    if not args_cli.asset_root_path or not os.path.exists(args_cli.asset_root_path):
        raise SystemExit("请设置 --asset_root_path 或环境变量 ASSET_ROOT_PATH 指向资产根路径。")
    os.environ["ASSET_ROOT_PATH"] = args_cli.asset_root_path

    # 必须启用相机
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch

    import isaaclab_tasks  # noqa: F401
    import isaaclab_logistics_vla  # noqa: F401
    from isaaclab_logistics_vla.utils.register import register

    register.auto_scan("isaaclab_logistics_vla.tasks")
    from isaaclab_logistics_vla.evaluation.evaluator.VLAIsaacEnv import VLAIsaacEnv
    from isaaclab_logistics_vla.agent.runner import VlmToolUseRunner
    from isaaclab_logistics_vla.agent.tools import (
        ToolManager,
        build_get_topview_image_tool,
        build_move_to_point_tool,
        build_step_action_tool,
        build_hold_action,
        read_topview_rgb_uint8,
    )
    from isaaclab_logistics_vla.agent.backend_registry import create_vlm_backend

    env_cfg = register.load_env_configs(args_cli.task_scene_name)()
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = 1
    # 覆盖顶视相机分辨率：降低 camera reset / render 的显存占用，避免 CUDA OOM
    try:
        if args_cli.top_camera_width and args_cli.top_camera_height:
            if hasattr(env_cfg.scene, "top_camera"):
                env_cfg.scene.top_camera.width = int(args_cli.top_camera_width)
                env_cfg.scene.top_camera.height = int(args_cli.top_camera_height)
    except Exception:
        pass
    env_cfg.sim.device = args_cli.device

    def _set_cuda_linalg_backend(backend: str) -> None:
        b = (backend or "").strip().lower()
        if b not in ("cusolver", "magma"):
            return
        try:
            torch.backends.cuda.preferred_linalg_library(b)
            print(f"[Info] torch CUDA linalg backend set to: {b}")
        except Exception as e:
            print(f"[Warn] Failed to set torch CUDA linalg backend={b}: {type(e).__name__}: {e}")

    if args_cli.cuda_linalg_backend in ("cusolver", "magma"):
        _set_cuda_linalg_backend(args_cli.cuda_linalg_backend)

    def _try_create_env() -> "VLAIsaacEnv":
        # First: handle cuSOLVER internal error by switching backend once.
        try:
            return VLAIsaacEnv(cfg=env_cfg)
        except RuntimeError as e:
            msg = str(e)
            if (
                args_cli.cuda_linalg_backend == "auto"
                and "CUSOLVER_STATUS_INTERNAL_ERROR" in msg
                and "cusolverDnCreate" in msg
                and torch.cuda.is_available()
            ):
                print("[Warn] Detected cuSOLVER handle creation failure; switching torch linalg backend to 'magma' and retrying once.")
                _set_cuda_linalg_backend("magma")
                return VLAIsaacEnv(cfg=env_cfg)
            raise

    def _looks_like_cuda_oom(err: BaseException) -> bool:
        if isinstance(err, torch.OutOfMemoryError):
            return True
        s = str(err)
        return ("CUDA out of memory" in s) or ("torch.OutOfMemoryError" in s)

    try:
        env = _try_create_env()
    except Exception as e:
        # Handle CUDA OOM during environment creation (common when IsaacSim/RTX fills VRAM).
        if _looks_like_cuda_oom(e) and str(args_cli.device).startswith("cuda") and args_cli.cuda_oom_fallback != "none":
            print(f"[Warn] CUDA OOM during env creation on device={args_cli.device}. Applying fallback={args_cli.cuda_oom_fallback}.")

            def _reduce_top_camera():
                try:
                    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "top_camera"):
                        w = int(getattr(env_cfg.scene.top_camera, "width", 0) or 0)
                        h = int(getattr(env_cfg.scene.top_camera, "height", 0) or 0)
                        if w > 0 and h > 0:
                            new_w = max(160, w // 2)
                            new_h = max(160, h // 2)
                            env_cfg.scene.top_camera.width = int(new_w)
                            env_cfg.scene.top_camera.height = int(new_h)
                            print(f"[Info] Reduced top_camera resolution: {w}x{h} -> {new_w}x{new_h}")
                except Exception as ee:
                    print(f"[Warn] Failed to reduce top_camera resolution: {type(ee).__name__}: {ee}")

            def _switch_to_cpu():
                try:
                    args_cli.device = "cpu"
                    env_cfg.sim.device = "cpu"
                    print("[Info] Switched env_cfg.sim.device to cpu for retry.")
                except Exception as ee:
                    print(f"[Warn] Failed to switch to CPU: {type(ee).__name__}: {ee}")

            # Apply strategy
            if args_cli.cuda_oom_fallback in ("reduce_camera", "reduce_then_cpu"):
                _reduce_top_camera()
                try:
                    env = _try_create_env()
                except Exception as e2:
                    if args_cli.cuda_oom_fallback == "reduce_then_cpu" and _looks_like_cuda_oom(e2):
                        _switch_to_cpu()
                        env = _try_create_env()
                    else:
                        raise
            elif args_cli.cuda_oom_fallback == "cpu":
                _switch_to_cpu()
                env = _try_create_env()
            else:
                raise
        else:
            raise
    env.reset()

    # 推进一帧，让相机输出可用
    adim = env.unwrapped.action_manager.total_action_dim
    env.step(torch.zeros((env.num_envs, adim), device=env.device))

    tools = ToolManager()
    from isaaclab_logistics_vla.agent.tools import (
        build_gripper_close_tool,
        build_gripper_open_tool,
        build_gripper_pick_nearest,
    )

    def _append_cb():
        _append_topview_video_frame(topview_video_writer)

    tools.register(build_get_topview_image_tool(env))
    tools.register(build_step_action_tool(env))
    tools.register(build_gripper_close_tool(env))
    tools.register(build_gripper_open_tool(env))
    tools.register(build_gripper_pick_nearest(env, append_video_frame=_append_cb))
    tools.register(build_move_to_point_tool(env, append_video_frame=_append_cb))
    if args_cli.enable_task_system:
        from isaaclab_logistics_vla.agent.tools import build_task_system_tools

        for t in build_task_system_tools(args_cli.tasks_dir):
            tools.register(t)

    if args_cli.enable_task_system:
        system_prompt = (
            "你是具身智能体评测助手。每一步只输出一个 JSON 对象，不要 Markdown、不要代码围栏。\n"
            "允许的 JSON：\n"
            '1) {"tool_name":"get_topview_image","parameters":{}} — 获取最新顶视 RGB；\n'
            '2) {"tool_name":"step_action","parameters":{"action":[...],"repeat":10}} — 执行动作让机器人移动（action 长度必须等于 action_dim；也可用 joints 字典按关节名赋值）；\n'
            '3) {"tool_name":"move_to_point","parameters":{"point_id":1}} — 平滑移动机器人到底座点位 1-6（1-3 在起始 s_box 后方，4-6 在目标 t_box 前方）；\n'
            '4) {"tool_name":"close_gripper","parameters":{"left_gripper":0.0,"right_gripper":0.0,"repeat":10}} — 闭合双侧夹爪抓取物体（0.0=闭合）；\n'
            '5) {"tool_name":"open_gripper","parameters":{"left_gripper":0.04,"right_gripper":0.04,"repeat":10}} — 张开双侧夹爪释放物体（0.04=张开）；\n'
            '6) {"tool_name":"gripper_pick_nearest","parameters":{"max_distance":0.3,"grasp_down_z":0.03}} — 快速抓取：固定右手抓取；如需仅抓罐头可加 only_can=true；\n'
            '7) {"tool_name":"task_create","parameters":{"subject":"...","description":"...","blockedBy":[...]}} — 创建任务；\n'
            '8) {"tool_name":"task_update","parameters":{"task_id":1,"status":"in_progress|completed|cancelled"}} — 更新任务；\n'
            '9) {"tool_name":"task_list","parameters":{"view":"all|ready|blocked"}} — 查看任务；\n'
            '10) {"tool_name":"task_get","parameters":{"task_id":1}} — 查看单个任务；\n'
            '11) {"scene_description":"..."} — 在对话中已经出现 tool 返回后，用中文详细描述画面内容；\n'
            '12) {"done":true,"summary":"...","scene_description":"可选"} — 结束本轮。\n'
        )
    else:
        system_prompt = (
            "你是具身智能体评测助手。每一步只输出一个 JSON 对象，不要 Markdown、不要代码围栏。\n"
            "允许的 JSON：\n"
            '1) {"tool_name":"get_topview_image","parameters":{}} — 获取最新顶视 RGB；\n'
            '2) {"tool_name":"step_action","parameters":{"action":[...],"repeat":10}} — 执行动作让机器人移动（action 长度必须等于 action_dim；也可用 joints 字典按关节名赋值）；\n'
            '3) {"tool_name":"move_to_point","parameters":{"point_id":1}} — 平滑移动机器人到底座点位 1-6（1-3 在起始 s_box 后方，4-6 在目标 t_box 前方）；\n'
            '4) {"tool_name":"close_gripper","parameters":{"left_gripper":0.0,"right_gripper":0.0,"repeat":10}} — 闭合双侧夹爪抓取物体（0.0=闭合）；\n'
            '5) {"tool_name":"open_gripper","parameters":{"left_gripper":0.04,"right_gripper":0.04,"repeat":10}} — 张开双侧夹爪释放物体（0.04=张开）；\n'
            '6) {"tool_name":"gripper_pick_nearest","parameters":{"max_distance":2.0,"grasp_down_z":0.03}} — 快速抓取：固定右手抓取；如需仅抓罐头可加 only_can=true；\n'
            '7) {"scene_description":"..."} — 在对话中已经出现 tool 返回后，用中文详细描述画面内容；\n'
            '8) {"done":true,"summary":"...","scene_description":"可选"} — 结束本轮。\n'
        )

    backend_kw = {}
    if args_cli.volc_api_key.strip():
        backend_kw["api_key"] = args_cli.volc_api_key.strip()
    if args_cli.anthropic_base_url.strip():
        backend_kw["base_url"] = args_cli.anthropic_base_url.strip()
    if args_cli.vlm_model.strip():
        backend_kw["model"] = args_cli.vlm_model.strip()
    if args_cli.backend == "openai_compat":
        if args_cli.openai_api_key.strip():
            backend_kw["api_key"] = args_cli.openai_api_key.strip()
        if args_cli.openai_base_url.strip():
            backend_kw["base_url"] = args_cli.openai_base_url.strip()

    backend = create_vlm_backend(args_cli.backend, test_mode="", **backend_kw)
    runner = VlmToolUseRunner(
        backend=backend,
        tool_manager=tools,
        system_prompt=system_prompt,
        max_steps=args_cli.max_steps,
        allow_scene_description_only=True,
        require_first_tool_name="",
        require_task_dag_before_done=False,
        tasks_dir_for_validation="",
        trace_jsonl_path=args_cli.trace_jsonl,
    )
    instruction_text = (args_cli.instruction or "").strip()
    if not instruction_text:
        instruction_text = "你是一个VLM机器人助手。请按照步骤完成搬运任务：\n1. 先观察场景，明白自身位置和每个箱子有什么东西\n2. 移动到任意一个放有物品的起始箱（点位 1-3 选一个）站定\n3. 使用gripper_pick_nearest工具抓取最近的物品\n4. 移动到任意一个目标箱（点位 4-6 选一个）站定\n5. 使用open_gripper工具放下物品\n6. 确认完成，输出done结束\n每做完一步都要输出scene_description描述当前位置和完成情况，让我看明白进度。"

    runner.reset(instruction_text)

    # 初始图像：直接先取一张，作为 VLM 的当前观测
    first = tools.execute("get_topview_image")
    if not first.get("success", False):
        raise RuntimeError(first)
    current_img = first["image_rgb_uint8"]
    if args_cli.save_observation_images:
        tid, subj = ("no_task", "no_task")
        if args_cli.enable_task_system:
            tid, subj = _pick_current_task_label(args_cli.tasks_dir)
        fname = f"step_{0:04d}__tool_get_topview_image__task_{_sanitize_filename(tid)}__{_sanitize_filename(subj)}.png"
        out_dir = Path(args_cli.observation_images_dir)
        out_path = out_dir / fname
        _save_rgb_png(current_img, out_path)
        print(f"[ObsImage] saved: {out_path}")

    # 外层顶视录制：从 agent 主循环开始，每完成一轮循环采一帧，直到循环结束
    topview_video_writer = None
    topview_video_path_resolved = None
    if args_cli.record_topview_video:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:
            import imageio  # type: ignore

        raw_vp = (args_cli.topview_video_path or "").strip()
        if raw_vp:
            vp = Path(raw_vp)
        else:
            vp = Path("traces/videos") / f"agent_topview_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        vp.parent.mkdir(parents=True, exist_ok=True)
        topview_video_path_resolved = vp.resolve()
        topview_video_writer = imageio.get_writer(
            str(topview_video_path_resolved),
            fps=float(args_cli.topview_video_fps),
        )
        print(
            f"[TopviewVideo] recording -> {topview_video_path_resolved} "
            f"(fps={args_cli.topview_video_fps}, one frame per agent iteration)"
        )

    def _append_topview_video_frame(writer) -> None:
        if writer is None:
            return
        fr = read_topview_rgb_uint8(env)
        if fr is not None:
            writer.append_data(fr)

    try:
        while True:
            out = runner.step(current_img)
            # 把模型的文字输出直接打印出来，便于人观察
            if out.get("description_only") and isinstance(out.get("scene_description"), str):
                print("\n" + "-" * 60)
                print("[VLM] scene_description:")
                print(out["scene_description"])
                print("-" * 60 + "\n")
            if out.get("done"):
                if isinstance(out.get("summary"), str) and out.get("summary", "").strip():
                    print("\n" + "-" * 60)
                    print("[VLM] done summary:")
                    print(out["summary"])
                    print("-" * 60 + "\n")
                _append_topview_video_frame(topview_video_writer)
                break

            # Print invalid-call feedback (these are validator rejections, not tool failures)
            if out.get("invalid_call"):
                err = out.get("error", "invalid_call")
                detail = out.get("detail", "")
                print(f"[InvalidCall] {err} {detail}".rstrip())
            if args_cli.enable_task_system and args_cli.print_task_board:
                tool_called = out.get("tool_called")
                tool_result = out.get("tool_result") or {}
                if tool_called in ("task_list", "task_create", "task_update", "task_get"):
                    try:
                        from isaaclab_logistics_vla.agent.task_system import TaskManager, format_task_board

                        print("\n" + "=" * 60)
                        print(f"[TaskBoard] after {tool_called}")
                        if isinstance(tool_result, dict) and isinstance(tool_result.get("tasks"), list):
                            print(format_task_board(tool_result["tasks"]))
                        else:
                            tm = TaskManager(Path(args_cli.tasks_dir))
                            print(format_task_board(tm.list_all()))
                        print("=" * 60 + "\n")
                    except Exception as e:
                        print(f"[TaskBoard] render failed: {type(e).__name__}: {e}")
            # 如果 tool 返回了新图像，则更新当前观测
            tr = out.get("tool_result") or {}
            if out.get("tool_called") and isinstance(tr, dict) and not tr.get("success", True):
                # 把失败原因打印出来，便于定位（否则只看到 tool_failures 计数）
                err = tr.get("error", "unknown_error")
                detail = tr.get("detail", "")
                print(f"[ToolError] {out.get('tool_called')}: {err} {detail}".rstrip())
            if tr.get("success") and "image_rgb_uint8" in tr:
                current_img = tr["image_rgb_uint8"]
                if out.get("tool_called") == "get_topview_image" and args_cli.save_observation_images:
                    # name includes current executing task (best-effort)
                    tid, subj = ("no_task", "no_task")
                    if args_cli.enable_task_system:
                        tid, subj = _pick_current_task_label(args_cli.tasks_dir)
                    step = int(getattr(runner, "stats", None).steps) if getattr(runner, "stats", None) is not None else 0
                    fname = f"step_{step:04d}__tool_get_topview_image__task_{_sanitize_filename(tid)}__{_sanitize_filename(subj)}.png"
                    out_dir = Path(args_cli.observation_images_dir)
                    out_path = out_dir / fname
                    _save_rgb_png(current_img, out_path)
                    print(f"[ObsImage] saved: {out_path}")
            # 物理推进一帧（保持静止动作即可）
            # 注意：如果 tool 已经做过 step（如 step_action），这里不要重复 step，避免动作放大。
            if out.get("tool_called") not in ("step_action", "move_to_point"):
                env.step(build_hold_action(env.unwrapped, env.num_envs, adim, env.device))
            _append_topview_video_frame(topview_video_writer)
    finally:
        if topview_video_writer is not None:
            try:
                topview_video_writer.close()
            except Exception as e:
                print(f"[TopviewVideo] writer.close failed: {type(e).__name__}: {e}")
            else:
                print(f"[TopviewVideo] closed: {topview_video_path_resolved}")

    result = {
        "backend": backend.model_name,
        "trace_jsonl": args_cli.trace_jsonl or None,
        "stats": runner.stats.__dict__,
        "done_reason": runner.stats.done_reason,
        "instruction": instruction_text,
        "topview_video": str(topview_video_path_resolved) if topview_video_path_resolved is not None else None,
    }
    print(result)
    if args_cli.out_json:
        out_path = Path(args_cli.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    runner.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
