import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
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
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument(
        "--backend",
        type=str,
        default="dummy",
        choices=["dummy", "volc_ark"],
        help="dummy=本地假模型；volc_ark=火山 Ark Coding（Anthropic 兼容，需 pip install anthropic 与 API Key）",
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
        help="模型名，默认 ark-code-latest",
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
        default=640,
        help="顶视相机宽度（默认 640，降低显存占用；设为 0 表示不覆盖场景配置）。",
    )
    parser.add_argument(
        "--top_camera_height",
        type=int,
        default=480,
        help="顶视相机高度（默认 480，降低显存占用；设为 0 表示不覆盖场景配置）。",
    )
    parser.add_argument(
        "--test_describe",
        action="store_true",
        default=False,
        help="测试模式：允许在调用 get_topview_image 后输出 {\"scene_description\":\"...\"}，并写入 stats.scene_descriptions。",
    )
    parser.add_argument(
        "--test_plan",
        action="store_true",
        default=False,
        help="测试模式：必须先 get_topview_image，再通过 task_create 编排一个 DAG（>=2 tasks 且有 blockedBy），最后 done 结束。会覆盖 test_describe。",
    )
    parser.add_argument(
        "--use_virtual_instruction",
        action="store_true",
        default=False,
        help="使用虚拟指令提供器（当前无真实订单系统时用于占位）。",
    )
    parser.add_argument(
        "--virtual_task_type",
        type=str,
        default="move_all_between_targets",
        help="虚拟任务类型：move_all_between_targets（默认）。",
    )
    parser.add_argument("--virtual_src_target", type=int, default=1, help="虚拟任务：源订单箱编号（从 1 开始）。")
    parser.add_argument("--virtual_dst_target", type=int, default=2, help="虚拟任务：目标订单箱编号（从 1 开始）。")
    parser.add_argument(
        "--enable_task_system",
        action="store_true",
        default=False,
        help="启用 harness 级持久化任务系统 tools（task_create/update/list/get）。默认关闭以保持最小动作空间。",
    )
    parser.add_argument(
        "--tasks_dir",
        type=str,
        default="/home/junzhe/isaaclab_logistics_vla/.tasks",
        help="任务系统持久化目录（每任务一个 JSON 文件）。",
    )
    parser.add_argument(
        "--print_task_board",
        action="store_true",
        default=False,
        help="控制台打印任务看板（READY/BLOCKED/DONE）。仅在 enable_task_system 时生效。",
    )
    args_cli, _ = parser.parse_known_args()
    if args_cli.test_plan:
        args_cli.test_describe = False
        args_cli.enable_task_system = True

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
    from isaaclab_logistics_vla.agent.tools import ToolManager, build_get_topview_image_tool
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

    env = VLAIsaacEnv(cfg=env_cfg)
    env.reset()

    # 推进一帧，让相机输出可用
    adim = env.unwrapped.action_manager.total_action_dim
    env.step(torch.zeros((env.num_envs, adim), device=env.device))

    tools = ToolManager()
    tools.register(build_get_topview_image_tool(env))
    if args_cli.enable_task_system:
        from isaaclab_logistics_vla.agent.tools import build_task_system_tools

        for t in build_task_system_tools(args_cli.tasks_dir):
            tools.register(t)

    if args_cli.test_plan:
        system_prompt = (
            "你是具身智能体评测助手（任务编排测试）。每一步只输出一个 JSON 对象，不要 Markdown、不要代码围栏。\n"
            "目标：先调用一次 get_topview_image，然后用 task_create 创建一个带依赖的任务图（DAG），最后 done。\n"
            "允许的 JSON：\n"
            '1) {"tool_name":"get_topview_image","parameters":{}} — 必须先调用；\n'
            '2) {"tool_name":"task_create","parameters":{"subject":"...","description":"...","blockedBy":[...]}} — 创建任务；\n'
            '3) {"tool_name":"task_update","parameters":{"task_id":1,"status":"completed"}} — 更新任务；\n'
            '4) {"tool_name":"task_list","parameters":{"view":"ready"}} — 查看可执行；\n'
            '5) {"done":true,"summary":"..."} — 仅当 DAG 已创建（>=2 tasks 且存在 blockedBy）才可结束。\n'
            "要求：DAG 至少 2 个任务，且至少 1 条 blockedBy 依赖边。"
        )
    elif args_cli.test_describe:
        system_prompt = (
            "你是具身智能体评测助手（测试阶段）。每一步只输出一个 JSON 对象，不要 Markdown、不要代码围栏。\n"
            "允许三种 JSON：\n"
            '1) {"tool_name":"get_topview_image","parameters":{}} — 获取最新顶视 RGB；\n'
            '2) {"scene_description":"..."} — 在对话中已经出现 tool 返回后，用中文详细描述画面内容；\n'
            '3) {"done":true,"summary":"...","scene_description":"可选"} — 结束本轮。\n'
            "建议顺序：先 1 再 2 再 3。只允许工具名 get_topview_image。"
        )
    else:
        system_prompt = (
            "你是一个具身智能体。你每一步必须输出 JSON："
            '{"tool_name":"get_topview_image","parameters":{}} 或 {"done":true,"summary":"..."}。'
            "当前只允许使用一个工具：get_topview_image。"
        )

    backend_kw = {}
    if args_cli.volc_api_key.strip():
        backend_kw["api_key"] = args_cli.volc_api_key.strip()
    if args_cli.anthropic_base_url.strip():
        backend_kw["base_url"] = args_cli.anthropic_base_url.strip()
    if args_cli.vlm_model.strip():
        backend_kw["model"] = args_cli.vlm_model.strip()

    backend = create_vlm_backend(
        args_cli.backend, test_mode=("plan" if args_cli.test_plan else ("describe" if args_cli.test_describe else "")), **backend_kw
    )
    runner = VlmToolUseRunner(
        backend=backend,
        tool_manager=tools,
        system_prompt=system_prompt,
        max_steps=args_cli.max_steps,
        allow_scene_description_only=args_cli.test_describe,
        require_first_tool_name=("get_topview_image" if args_cli.test_plan else ""),
        require_task_dag_before_done=args_cli.test_plan,
        tasks_dir_for_validation=(args_cli.tasks_dir if args_cli.test_plan else ""),
        trace_jsonl_path=args_cli.trace_jsonl,
    )
    instruction_text = (args_cli.instruction or "").strip()
    instruction_meta = None
    if not instruction_text and args_cli.use_virtual_instruction:
        from isaaclab_logistics_vla.agent.instruction_provider import VirtualInstructionProvider

        provider = VirtualInstructionProvider(
            task_type=args_cli.virtual_task_type,
            src_target_idx=args_cli.virtual_src_target,
            dst_target_idx=args_cli.virtual_dst_target,
        )
        spec = provider.next_instruction()
        instruction_text = spec.instruction
        instruction_meta = {"task_type": spec.task_type, "task_id": spec.task_id, "metadata": spec.metadata or {}}
    if not instruction_text:
        if args_cli.test_plan:
            instruction_text = (
                "你是一个VLM机器人助手。请严格按顺序：先调用 get_topview_image，"
                "然后使用 task_create（可多次）编排一个任务 DAG（至少 2 个任务且存在 blockedBy 依赖），最后 done 结束。"
            )
        else:
            instruction_text = "你是一个VLM机器人助手。请先调用 get_topview_image，然后输出 scene_description 描述你看到的场景，最后 done 结束。"

    runner.reset(instruction_text)

    # 初始图像：直接先取一张，作为 VLM 的当前观测
    first = tools.execute("get_topview_image")
    if not first.get("success", False):
        raise RuntimeError(first)
    current_img = first["image_rgb_uint8"]

    while True:
        out = runner.step(current_img)
        if out.get("done"):
            break
        # 任务系统可视化（不影响模型动作空间，只是把 tool 返回画成看板给人看）
        if args_cli.enable_task_system and args_cli.print_task_board:
            tool_called = out.get("tool_called")
            tool_result = out.get("tool_result") or {}
            if tool_called in ("task_list", "task_create", "task_update", "task_get"):
                try:
                    from isaaclab_logistics_vla.agent.task_system import format_task_board, TaskManager

                    print("\n" + "=" * 60)
                    print(f"[TaskBoard] after {tool_called}")
                    # 若本次返回里带 tasks，优先用它；否则从磁盘读全量任务
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
        if tr.get("success") and "image_rgb_uint8" in tr:
            current_img = tr["image_rgb_uint8"]
        # 物理推进一帧（保持静止动作即可）
        env.step(torch.zeros((env.num_envs, adim), device=env.device))

    result = {
        "backend": backend.model_name,
        "test_describe": args_cli.test_describe,
        "test_plan": args_cli.test_plan,
        "enable_task_system": args_cli.enable_task_system,
        "tasks_dir": args_cli.tasks_dir if args_cli.enable_task_system else None,
        "print_task_board": args_cli.print_task_board if args_cli.enable_task_system else False,
        "trace_jsonl": args_cli.trace_jsonl or None,
        "stats": runner.stats.__dict__,
        "done_reason": runner.stats.done_reason,
        "instruction": instruction_text,
        "instruction_meta": instruction_meta,
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

