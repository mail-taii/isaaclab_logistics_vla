from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from isaaclab_logistics_vla.agent.tool_spec import ToolSpec


def build_move_to_point_tool(env, append_video_frame = None) -> ToolSpec:
    """Move robot base smoothly to one of 6 preset points near boxes.

    point_id: 1..6 → (1-3) near s_box_1..s_box_3, (4-6) near t_box_1..t_box_3.
    offset_xy: [dx, dy] optional fine-tune offset relative to box front.
    speed: linear movement speed m/s (default ~0.5 m/s).
    append_video_frame: optional callback to append a frame after each simulation step (for topview video recording).
    """

    def _handler(
        point_id: int,
        offset_xy: Optional[list[float]] = None,
        speed: float = 0.5,
    ) -> Dict[str, Any]:
        import torch
        from isaaclab_logistics_vla.agent.tools.gripper_control import sync_attached_object_to_gripper, build_hold_action

        from isaaclab_logistics_vla.utils.object_position import set_asset_relative_position

        pid = int(point_id)
        if pid < 1 or pid > 6:
            return {"success": False, "error": "invalid_point_id", "detail": "point_id must be 1..6"}

        box_name = f"s_box_{pid}" if pid <= 3 else f"t_box_{pid - 3}"

        # For source boxes (s_box): stand behind it (+Y)
        # For target boxes (t_box): stand in front (-Y)
        if pid <= 3:
            ox, oy = (0.0, +0.8)
        else:
            ox, oy = (0.0, -0.8)
        if offset_xy is not None:
            try:
                arr = np.asarray(offset_xy, dtype=np.float32).reshape(-1)
                if int(arr.shape[0]) != 2:
                    raise ValueError("offset_xy must have 2 floats")
                ox, oy = float(arr[0]), float(arr[1])
            except Exception as e:
                return {"success": False, "error": "invalid_offset_xy", "detail": f"{type(e).__name__}: {e}"}

        isaac_env = env.unwrapped
        device = getattr(isaac_env, "device", "cpu")
        num_envs = int(getattr(isaac_env, "num_envs", 1))
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

        scene = getattr(isaac_env, "scene", None)
        if scene is None:
            return {"success": False, "error": "scene_missing"}

        scene_keys = list(getattr(scene, "keys", lambda: [])())
        if "robot" not in scene_keys:
            return {
                "success": False,
                "error": "robot_not_found",
                "detail": f"scene keys={scene_keys[:50]}",
            }
        if box_name not in scene_keys:
            return {
                "success": False,
                "error": "box_not_found",
                "detail": {"box": box_name, "scene_keys": scene_keys[:50]},
            }

        try:
            robot = scene["robot"]
            box = scene[box_name]
        except KeyError as e:
            return {
                "success": False,
                "error": "scene_key_error",
                "detail": str(e),
                "scene_keys": scene_keys[:50],
            }

        # Compute target position (relative to box)
        try:
            box_pos_w = box.data.root_pos_w[env_ids]
            box_z = box_pos_w[..., 2:3]
            rel_z = robot.data.root_pos_w[env_ids, 2:3] - box_z
        except Exception:
            rel_z = torch.zeros((len(env_ids), 1), device=device)

        rel_xy = torch.tensor([[ox, oy]], device=device).repeat(len(env_ids), 1)
        rel_pos_target = torch.cat([rel_xy, rel_z], dim=-1)
        target_pos_w = box_pos_w + rel_pos_target

        # Get start position from current
        start_pos_w = robot.data.root_pos_w[env_ids].clone()

        # Straight-line linear interpolation (constant speed)
        # Estimate physics step dt ~ 1/60
        dt = 1.0 / 60.0
        distance = torch.norm(target_pos_w - start_pos_w, dim=-1).max().item()
        if distance < 1e-3:
            # already at target
            distance = 0.0
            n_steps = 0
        else:
            # Clamp min/max steps to avoid extremely long moves
            min_steps = 2
            max_steps = 500
            n_steps = int(max(min_steps, min(max_steps, round(distance / (speed * dt)))))

        adim = int(isaac_env.action_manager.total_action_dim)

        # Predefine standard yaw rotations (heading in world frame):
        # - s_box (start, 1-3): robot stands BEHIND box → face SOUTH (-Y) → yaw 0 deg = wxyz [1, 0, 0, 0] in Isaac Sim convention
        # - t_box (target, 4-6): robot stands IN FRONT box → face NORTH (+Y) → yaw 180 deg = wxyz [0, 0, 0, 1]
        quat_face_south = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        quat_face_north = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        if pid <= 3:
            target_quat_w = quat_face_south.repeat(len(env_ids), 1)
        else:
            target_quat_w = quat_face_north.repeat(len(env_ids), 1)

        # Start from current orientation and slerp to target during movement
        start_quat_w = robot.data.root_quat_w[env_ids].clone()

        # Smooth interpolation steps (linear position + spherical quaternion interpolation)
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            interp_pos = start_pos_w * (1 - alpha) + target_pos_w * alpha
            # Simple SLERP for orientation: linear interpolation then renormalize
            interp_quat = (1 - alpha) * start_quat_w + alpha * target_quat_w
            interp_quat = interp_quat / torch.norm(interp_quat, dim=-1, keepdim=True)
            # Write to simulation correctly
            robot.write_root_pose_to_sim(torch.cat([interp_pos, interp_quat], dim=-1), env_ids)
            # If a test object is attached by gripper_pick_nearest, keep it following.
            sync_attached_object_to_gripper(env)
            # Step simulation with zero action (base doesn't move itself)
            isaac_env.step(build_hold_action(isaac_env, num_envs, adim, device))
            # If video recording is active, append this intermediate frame
            if append_video_frame is not None:
                append_video_frame()

        # Final snap to exactly target position/quaternion at end for accuracy
        robot.write_root_pose_to_sim(torch.cat([target_pos_w, target_quat_w], dim=-1), env_ids)
        sync_attached_object_to_gripper(env)
        isaac_env.step(build_hold_action(isaac_env, num_envs, adim, device))
        # Append final frame too
        if append_video_frame is not None:
            append_video_frame()

        return {
            "success": True,
            "point_id": pid,
            "box_name": box_name,
            "offset_xy": [ox, oy],
            "distance_m": distance,
            "steps_taken": n_steps,
            "speed_m_s": speed,
        }

    return ToolSpec(
        name="move_to_point",
        description="Move robot smoothly (linear interpolation) to one of 6 preset standing points next to boxes: point_id=1..6 (1-3 behind s_box 1-3, 4-6 in front of t_box 1-3). Optional speed (m/s, default 0.5).",
        handler=_handler,
    )
