from __future__ import annotations

from typing import Any, Dict

import torch

from isaaclab_logistics_vla.agent.tool_spec import ToolSpec


def _to_binary_gripper_cmd(v: float, default_close: bool) -> float:
    """Map legacy 0.0/0.04 style and binary style to BinaryJointPositionAction cmd.

    Returns:
    -1.0 => close
    +1.0 => open
    """
    x = float(v)
    # Backward-compatible mapping used in this repo prompts.
    if 0.0 <= x <= 0.04:
        return -1.0 if x <= 0.02 else 1.0
    if x < 0.0:
        return -1.0
    if x > 0.0:
        return 1.0
    return -1.0 if default_close else 1.0


def build_hold_action(isaac_env, num_envs: int, adim: int, device):
    """Build zero action while holding latest gripper command on indices 14/15."""
    a = torch.zeros((num_envs, adim), device=device)
    left = float(getattr(isaac_env, "_tool_gripper_left_cmd", 0.0))
    right = float(getattr(isaac_env, "_tool_gripper_right_cmd", 0.0))
    if 14 < adim:
        a[..., 14] = left
    if 15 < adim:
        a[..., 15] = right
    return a


def _get_ee_frame_pos_map(isaac_env, env_ids):
    scene = getattr(isaac_env, "scene", None)
    if scene is None:
        return {}
    scene_keys = list(getattr(scene, "keys", lambda: [])())
    if "ee_frame" not in scene_keys:
        return {}
    out = {}
    try:
        ee = scene["ee_frame"]
        pos = ee.data.target_pos_w[env_ids][0].detach().cpu().numpy()  # (N,3)
        names = list(getattr(ee.data, "target_frame_names", []) or [])
        if len(names) == pos.shape[0]:
            for i, n in enumerate(names):
                out[str(n)] = pos[i]
    except Exception:
        return {}
    return out


def sync_attached_object_to_gripper(env) -> bool:
    """If object attachment is active, keep object following gripper center."""
    import numpy as np

    isaac_env = env.unwrapped
    scene = getattr(isaac_env, "scene", None)
    if scene is None:
        return False
    obj_name = str(getattr(isaac_env, "_tool_attached_object_name", "") or "")
    if not obj_name:
        return False
    scene_keys = list(getattr(scene, "keys", lambda: [])())
    if obj_name not in scene_keys or "robot" not in scene_keys:
        return False
    device = getattr(isaac_env, "device", "cpu")
    num_envs = int(getattr(isaac_env, "num_envs", 1))
    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    frame_name = str(getattr(isaac_env, "_tool_attached_frame_name", "right_ee_tcp") or "right_ee_tcp")
    ee_map = _get_ee_frame_pos_map(isaac_env, env_ids)
    grasp_down_z = float(getattr(isaac_env, "_tool_grasp_down_z", 0.03))
    if frame_name in ee_map:
        gripper_center_pos = ee_map[frame_name].copy()
        gripper_center_pos[2] = float(gripper_center_pos[2] - grasp_down_z)
    else:
        robot = scene["robot"]
        base_pos = robot.data.root_pos_w[env_ids][0].detach().cpu().numpy()
        gripper_center_pos = base_pos + np.array([0.0, 0.0, 1.4 - grasp_down_z], dtype=np.float32)
    asset = scene[obj_name]
    pos = torch.tensor(gripper_center_pos[None, :], device=device, dtype=torch.float32)
    if bool(getattr(isaac_env, "_tool_strong_attach_lock_quat", True)):
        q_saved = getattr(isaac_env, "_tool_attached_quat_w", None)
        if q_saved is not None:
            quat = torch.tensor(q_saved, device=device, dtype=torch.float32).view(1, 4)
        else:
            quat = asset.data.root_quat_w[env_ids][0][None, :].clone().to(torch.float32)
    else:
        quat = asset.data.root_quat_w[env_ids][0][None, :].clone().to(torch.float32)
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids)
    if bool(getattr(isaac_env, "_tool_strong_attach_zero_vel", True)) and hasattr(asset, "write_root_velocity_to_sim"):
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device, dtype=torch.float32), env_ids=env_ids)
    return True


def read_gripper_debug_state(env) -> Dict[str, Any]:
    """Read hold commands + finger joint positions for debugging."""
    isaac_env = env.unwrapped
    out: Dict[str, Any] = {
        "hold_left_cmd": float(getattr(isaac_env, "_tool_gripper_left_cmd", 0.0)),
        "hold_right_cmd": float(getattr(isaac_env, "_tool_gripper_right_cmd", 0.0)),
        "attached_object": str(getattr(isaac_env, "_tool_attached_object_name", "") or ""),
        "attached_frame": str(getattr(isaac_env, "_tool_attached_frame_name", "") or ""),
    }
    scene = getattr(isaac_env, "scene", None)
    if scene is None:
        return out
    try:
        robot = scene["robot"]
        names = list(getattr(robot.data, "joint_names", []) or [])
        q = robot.data.joint_pos[0].detach().cpu()
        finger = {}
        for name in ("left_left_joint", "left_right_joint", "right_left_joint", "right_right_joint"):
            if name in names:
                finger[name] = float(q[names.index(name)].item())
        out["finger_joint_pos"] = finger
    except Exception:
        pass
    return out


def build_gripper_close_tool(env) -> ToolSpec:
    """Close both left/right gripper to grasp an object."""

    def _handler(
        left_gripper: float = 0.0,
        right_gripper: float = 0.0,
        repeat: int = 10,
    ) -> Dict[str, Any]:
        """Close gripper fingers to given joint position (default 0.04 is fully closed).
        - repeat: number of simulation steps to wait after closing.
        """
        isaac_env = env.unwrapped
        adim = int(isaac_env.action_manager.total_action_dim)
        num_envs = int(getattr(isaac_env, "num_envs", 1))
        device = getattr(isaac_env, "device", "cpu")

        # build zero action
        a = torch.zeros((num_envs, adim), device=device)
        # set gripper joints positions
        # NOTE: we assume the standard joint mapping: left_gripper=14, right_gripper=15 (same as we have in run_vlm script)
        if 14 < adim:
            a[..., 14] = _to_binary_gripper_cmd(left_gripper, default_close=True)
        if 15 < adim:
            a[..., 15] = _to_binary_gripper_cmd(right_gripper, default_close=True)
        isaac_env._tool_gripper_left_cmd = float(a[0, 14].item()) if 14 < adim else 0.0
        isaac_env._tool_gripper_right_cmd = float(a[0, 15].item()) if 15 < adim else 0.0
        # step to execute
        for _ in range(int(repeat)):
            _, _, _, _, info = isaac_env.step(a)

        return {
            "success": True,
            "left_gripper": float(left_gripper),
            "right_gripper": float(right_gripper),
            "steps": int(repeat),
        }

    return ToolSpec(
        name="close_gripper",
        description="Close gripper (Binary action). Supports legacy values: 0.0=close, 0.04=open; or explicit binary: -1 close, +1 open.",
        handler=_handler,
    )


def build_gripper_open_tool(env) -> ToolSpec:
    """Open both left/right gripper to release an object."""

    def _handler(
        left_gripper: float = 0.04,
        right_gripper: float = 0.04,
        repeat: int = 10,
    ) -> Dict[str, Any]:
        """Open gripper fingers to given joint position (default 0.0 is fully open).
        - repeat: number of simulation steps to wait after opening.
        """
        isaac_env = env.unwrapped
        adim = int(isaac_env.action_manager.total_action_dim)
        num_envs = int(getattr(isaac_env, "num_envs", 1))
        device = getattr(isaac_env, "device", "cpu")

        # build zero action
        a = torch.zeros((num_envs, adim), device=device)
        # set gripper joints positions
        if 14 < adim:
            a[..., 14] = _to_binary_gripper_cmd(left_gripper, default_close=False)
        if 15 < adim:
            a[..., 15] = _to_binary_gripper_cmd(right_gripper, default_close=False)
        isaac_env._tool_gripper_left_cmd = float(a[0, 14].item()) if 14 < adim else 0.0
        isaac_env._tool_gripper_right_cmd = float(a[0, 15].item()) if 15 < adim else 0.0
        isaac_env._tool_attached_object_name = ""
        isaac_env._tool_attached_frame_name = ""
        isaac_env._tool_attached_quat_w = None
        # step to execute
        for _ in range(int(repeat)):
            _, _, _, _, info = isaac_env.step(a)

        return {
            "success": True,
            "left_gripper": float(left_gripper),
            "right_gripper": float(right_gripper),
            "steps": int(repeat),
        }

    return ToolSpec(
        name="open_gripper",
        description="Open gripper (Binary action). Supports legacy values: 0.04=open, 0.0=close; or explicit binary: +1 open, -1 close.",
        handler=_handler,
    )


def build_gripper_pick_nearest_tool(env, append_video_frame=None) -> ToolSpec:
    """Teleport the nearest rigid body object into gripper center then close gripper.
    This is a helper for testing: skip motion planning, just "grab it now" to test moving with object.
    """

    def _handler(
        max_distance: float = 2.0,
        close_repeat: int = 20,
        grasp_down_z: float = 0.03,
        only_can: bool = False,
        strong_attach: bool = True,
        lock_quat: bool = True,
        zero_velocity: bool = True,
    ) -> Dict[str, Any]:
        import torch
        import numpy as np

        isaac_env = env.unwrapped
        device = getattr(isaac_env, "device", "cpu")
        num_envs = int(getattr(isaac_env, "num_envs", 1))
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

        # Get robot and gripper info from scene (all assets already registered in IsaacLab scene)
        scene = getattr(isaac_env, "scene", None)
        if scene is None:
            return {"success": False, "error": "scene_missing"}

        scene_keys = list(getattr(scene, "keys", lambda: [])())
        if "robot" not in scene_keys:
            return {"success": False, "error": "robot_not_found", "scene_keys": scene_keys[:50]}

        # Force right-hand grasp anchor when available.
        ee_map = _get_ee_frame_pos_map(isaac_env, env_ids)
        if "right_ee_tcp" in ee_map:
            gripper_center_pos = ee_map["right_ee_tcp"].copy()
            gripper_center_pos[2] = float(gripper_center_pos[2] - grasp_down_z)
            attached_frame = "right_ee_tcp"
        else:
            robot = scene["robot"]
            base_pos = robot.data.root_pos_w[env_ids][0].detach().cpu().numpy()
            gripper_center_pos = base_pos + np.array([0.0, 0.0, 1.4 - grasp_down_z])
            attached_frame = "base_offset"

        # Now find all objects in scene that are not robot/ground/camera/boxes
        # We pick the closest object within max_distance
        picked: list[str] = []
        distances: list[float] = []
        positions: list[np.ndarray] = []
        for name in scene_keys:
            if name in ("robot", "ground", "top_camera", "s_box_1", "s_box_2", "s_box_3", "t_box_1", "t_box_2", "t_box_3"):
                continue
            if bool(only_can) and ("can" not in str(name).lower()):
                continue
            asset = scene[name]
            if asset is None:
                continue
            # get object root position from IsaacLab data
            try:
                obj_pos_w = asset.data.root_pos_w[env_ids][0].detach().cpu().numpy()
            except Exception:
                continue
            dist = np.linalg.norm(obj_pos_w - gripper_center_pos)
            if dist <= max_distance:
                distances.append(dist)
                positions.append(obj_pos_w)
                picked.append(name)

        if not picked:
            return {
                "success": False,
                "error": "no_object_in_range",
                "max_distance": max_distance,
                "only_can": bool(only_can),
                "attached_frame": attached_frame,
                "gripper_center": gripper_center_pos.tolist(),
            }

        # pick closest object
        min_i = int(np.argmin(distances))
        picked_name = picked[min_i]
        picked_asset = scene[picked_name]
        start_obj_pos = positions[min_i]

        # Move object to gripper center (keep original orientation)
        target_obj_pos = gripper_center_pos.copy()
        target_pos_tensor = torch.tensor(target_obj_pos[None, :], device=device, dtype=torch.float32)
        target_quat_tensor = picked_asset.data.root_quat_w[env_ids][0][None, :].clone().to(torch.float32)

        # Write new pose to simulation using Isaac API
        picked_asset.write_root_pose_to_sim(torch.cat([target_pos_tensor, target_quat_tensor], dim=-1), env_ids)
        if hasattr(picked_asset, "write_root_velocity_to_sim"):
            picked_asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device, dtype=torch.float32), env_ids=env_ids)
        isaac_env._tool_attached_object_name = str(picked_name)
        isaac_env._tool_attached_frame_name = attached_frame
        isaac_env._tool_grasp_down_z = float(grasp_down_z)
        isaac_env._tool_attached_quat_w = target_quat_tensor[0].detach().cpu().tolist()
        isaac_env._tool_strong_attach_lock_quat = bool(lock_quat)
        isaac_env._tool_strong_attach_zero_vel = bool(zero_velocity)

        # Auto close gripper and wait a few steps to let physics lock in
        adim = int(isaac_env.action_manager.total_action_dim)
        a = torch.zeros((num_envs, adim), device=device)
        if 14 < adim:
            a[..., 14] = 1.0   # keep left gripper open
        if 15 < adim:
            a[..., 15] = -1.0  # close right gripper for right-hand grasp
        isaac_env._tool_gripper_left_cmd = float(a[0, 14].item()) if 14 < adim else 0.0
        isaac_env._tool_gripper_right_cmd = float(a[0, 15].item()) if 15 < adim else 0.0
        for _ in range(int(close_repeat)):
            if bool(strong_attach):
                sync_attached_object_to_gripper(env)
            isaac_env.step(a)
            # append frame to video during wait steps if recording is on
            if append_video_frame is not None:
                append_video_frame()

        return {
            "success": True,
            "object_name": picked_name,
            "original_distance_m": float(distances[min_i]),
            "original_position": start_obj_pos.tolist(),
            "target_position": target_obj_pos.tolist(),
            "gripper_center": gripper_center_pos.tolist(),
            "only_can": bool(only_can),
            "attached_frame": attached_frame,
            "strong_attach": bool(strong_attach),
            "lock_quat": bool(lock_quat),
            "zero_velocity": bool(zero_velocity),
            "close_steps": int(close_repeat),
        }

    return ToolSpec(
        name="gripper_pick_nearest",
        description="Grab helper: defaults to right-hand grasp. Set only_can=true to restrict to cans. strong_attach/lock_quat/zero_velocity can make attachment look firmly grasped.",
        handler=_handler,
    )


# Alias for shorter imports (matches tool name gripper_pick_nearest)
build_gripper_pick_nearest = build_gripper_pick_nearest_tool
