# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Randomization functions for single arm sorting task.

This module provides functions to randomize object generation, including:
- Random generation of multiple objects
- Support for external resources (USD files)
- Configurable object types and properties
"""

from __future__ import annotations

import torch
import random
from typing import TYPE_CHECKING, Sequence
from pathlib import Path

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class ObjectSpawnConfig:
    """Configuration for spawning a single object."""
    
    # Object type: "cuboid", "sphere", "cylinder", or "usd" for external file
    object_type: str = "cuboid"
    
    # For USD files, specify the path
    usd_path: str | None = None
    
    # Size parameters (for primitive shapes)
    size: tuple[float, float, float] = (0.10, 0.10, 0.10)
    
    # Mass
    mass: float = 0.1
    
    # Color (RGB, 0-1)
    color: tuple[float, float, float] = (0.8, 0.2, 0.2)
    
    # Position range (relative to source area)
    pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.1, 0.1),  # x range
        (-0.1, 0.1),  # y range
        (0.0, 0.1),   # z range (height)
    )


def randomize_objects(
    env: ManagerBasedRLEnv,
    source_cfg: SceneEntityCfg = SceneEntityCfg("source_area"),
    num_objects_range: tuple[int, int] = (1, 5),
    object_configs: Sequence[ObjectSpawnConfig] | None = None,
    object_prefix: str = "object",
) -> None:
    """Randomize object generation in the scene.
    
    This function can generate multiple objects with random properties or from external resources.
    
    Args:
        env: The environment instance.
        source_cfg: Configuration for the source area (objects will be spawned near it).
        num_objects_range: Range of number of objects to generate (min, max).
        object_configs: List of object configurations. If None, uses default configurations.
        object_prefix: Prefix for object names in the scene.
    
    Note:
        This function should be called during environment reset or initialization.
        It will create objects dynamically in the scene.
    """
    num_envs = env.scene.num_envs
    device = env.device
    
    # Get source area position
    source = env.scene[source_cfg.name]
    source_pos = source.data.root_pos_w  # (num_envs, 3)
    
    # Determine number of objects per environment
    if isinstance(num_objects_range, tuple):
        num_objects = torch.randint(
            num_objects_range[0],
            num_objects_range[1] + 1,
            (num_envs,),
            device=device,
        )
    else:
        num_objects = torch.full((num_envs,), num_objects_range, device=device)
    
    # Default object configurations if not provided
    if object_configs is None:
        object_configs = [
            ObjectSpawnConfig(
                object_type="cuboid",
                size=(0.10, 0.10, 0.10),
                mass=0.1,
                color=(0.8, 0.2, 0.2),
            )
        ]
    
    # Generate objects for each environment
    max_objects = num_objects.max().item()
    
    for obj_idx in range(max_objects):
        # Select random config for this object
        config = random.choice(object_configs)
        
        # Create object name
        obj_name = f"{object_prefix}_{obj_idx}"
        
        # Check if object already exists in scene
        if obj_name in env.scene:
            obj: RigidObject = env.scene[obj_name]
        else:
            # Object will be created by the scene configuration
            # For now, we'll just randomize positions of existing objects
            continue
        
        # Randomize position for each environment
        mask = num_objects > obj_idx  # Which environments should have this object
        
        if mask.any():
            # Random positions relative to source area
            if config.object_type == "cuboid":
                pos_x = torch.rand(num_envs, device=device) * (config.pos_range[0][1] - config.pos_range[0][0]) + config.pos_range[0][0]
                pos_y = torch.rand(num_envs, device=device) * (config.pos_range[1][1] - config.pos_range[1][0]) + config.pos_range[1][0]
                pos_z = torch.rand(num_envs, device=device) * (config.pos_range[2][1] - config.pos_range[2][0]) + config.pos_range[2][0]
            else:
                # Default position
                pos_x = torch.zeros(num_envs, device=device)
                pos_y = torch.zeros(num_envs, device=device)
                pos_z = torch.full((num_envs,), 0.05, device=device)
            
            # Add to source position
            obj_pos = source_pos.clone()
            obj_pos[:, 0] += pos_x
            obj_pos[:, 1] += pos_y
            obj_pos[:, 2] += pos_z
            
            # Only set position for environments that should have this object
            obj_pos[~mask] = obj.data.root_pos_w[~mask]  # Keep original position for others
            
            # Set object position
            obj.set_root_pose_w(obj_pos, torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(num_envs, 1))
            
            # Randomize orientation
            random_quat = _random_quaternion(num_envs, device)
            obj.set_root_pose_w(obj_pos, random_quat)


def randomize_object_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    source_cfg: SceneEntityCfg = SceneEntityCfg("source_area"),
    pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.1, 0.1),
        (-0.1, 0.1),
        (0.0, 0.1),
    ),
) -> None:
    """Randomize the position of a single object relative to the source area.
    
    This function is compatible with the EventManager convention:
    ``func(env, env_ids, **params)``. If ``env_ids`` is None, all environments
    are randomized.
    
    Args:
        env: The environment instance.
        env_ids: Tensor of environment indices to randomize. If None, all envs.
        object_cfg: Configuration for the object to randomize.
        source_cfg: Configuration for the source area.
        pos_range: Range of positions relative to source area (x_range, y_range, z_range).
    """
    device = env.device
    num_envs = env.scene.num_envs
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device)
    
    # Get source and object
    try:
        source = env.scene[source_cfg.name]
        obj = env.scene[object_cfg.name]
    except KeyError:
        # Object or source not found, skip randomization
        return

    # 如果对象是 RigidObjectCollection（例如 Realman 中的目标物集合），
    # 则其随机化由专门的 randomize_target_objects 控制，这里不再做通用位姿抖动。
    if isinstance(obj, RigidObjectCollection):
        return

    assert isinstance(obj, RigidObject), "randomize_object_positions only supports RigidObject or skips collections."
    
    # Source position for selected envs
    source_pos = source.data.root_pos_w[env_ids]  # (N, 3)
    n = env_ids.shape[0]
    
    # Random offsets
    pos_x = torch.rand(n, device=device) * (pos_range[0][1] - pos_range[0][0]) + pos_range[0][0]
    pos_y = torch.rand(n, device=device) * (pos_range[1][1] - pos_range[1][0]) + pos_range[1][0]
    pos_z = torch.rand(n, device=device) * (pos_range[2][1] - pos_range[2][0]) + pos_range[2][0]
    
    # Compute new object positions for selected envs
    obj_pos = source_pos.clone()
    obj_pos[:, 0] += pos_x
    obj_pos[:, 1] += pos_y
    obj_pos[:, 2] += pos_z
    
    # Random orientation for selected envs
    random_quat = _random_quaternion(n, device)
    
    # Compose root pose tensor (pos + quat) for the selected envs and write to sim
    root_pose = torch.cat([obj_pos, random_quat], dim=-1)  # (N, 7)
    obj.write_root_pose_to_sim(root_pose, env_ids=env_ids)


def randomize_object_properties(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    mass_range: tuple[float, float] = (0.05, 0.2),
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize object properties like mass and scale.
    
    Args:
        env: The environment instance.
        env_ids: Tensor of environment indices to randomize (unused for now).
        object_cfg: Configuration for the object to randomize.
        mass_range: Range of mass values (min, max). Currently not implemented.
        scale_range: Range of scale factors (min, max). Currently not implemented.
    
    Note:
        Mass and scale randomization require direct API access to modify physics properties.
        This function is a placeholder for future implementation. Currently, it only
        randomizes visual properties if supported by the object.
    """
    try:
        obj = env.scene[object_cfg.name]
    except KeyError:
        # Object not found, skip randomization
        return

    # 同上：集合由 randomize_target_objects 控制，这里不改属性
    if isinstance(obj, RigidObjectCollection):
        return

    assert isinstance(obj, RigidObject), "randomize_object_properties only supports RigidObject or skips collections."
    
    # TODO: Implement mass and scale randomization
    # This requires access to the underlying physics simulation API
    # For now, this function is a placeholder that can be extended
    # when the necessary APIs are available
    pass


def randomize_multiple_objects(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    source_cfg: SceneEntityCfg = SceneEntityCfg("source_area"),
    pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.1, 0.1),
        (-0.1, 0.1),
        (0.0, 0.1),
    ),
    num_active_range: tuple[int, int] | None = None,
) -> None:
    """Randomize positions of multiple objects in the scene.
    
    This function can be used when multiple objects are defined in the scene configuration.
    It will randomize the positions of all specified objects, and optionally control
    how many objects are active in each environment.
    
    Args:
        env: The environment instance.
        object_names: List of object names in the scene to randomize.
        source_cfg: Configuration for the source area.
        pos_range: Range of positions relative to source area.
        num_active_range: Optional range for number of active objects per environment.
                         If None, all objects are active. If specified, only the first
                         N objects (randomly chosen) will be positioned, others will be
                         moved out of the way.
    """
    num_envs = env.scene.num_envs
    device = env.device
    
    # Get source area
    try:
        source = env.scene[source_cfg.name]
        source_pos = source.data.root_pos_w  # (num_envs, 3)
    except KeyError:
        return
    
    # Determine how many objects to activate per environment
    if num_active_range is not None:
        num_active = torch.randint(
            num_active_range[0],
            num_active_range[1] + 1,
            (num_envs,),
            device=device,
        )
    else:
        num_active = torch.full((num_envs,), len(object_names), device=device)
    
    # Randomize each object
    for obj_idx, obj_name in enumerate(object_names):
        try:
            obj: RigidObject = env.scene[obj_name]
        except KeyError:
            continue
        
        # Determine which environments should have this object active
        mask = num_active > obj_idx
        
        # Random offsets
        pos_x = (
            torch.rand(num_envs, device=device) * (pos_range[0][1] - pos_range[0][0])
            + pos_range[0][0]
        )
        pos_y = (
            torch.rand(num_envs, device=device) * (pos_range[1][1] - pos_range[1][0])
            + pos_range[1][0]
        )
        pos_z = (
            torch.rand(num_envs, device=device) * (pos_range[2][1] - pos_range[2][0])
            + pos_range[2][0]
        )
        
        # Compute object position
        obj_pos = source_pos.clone()
        obj_pos[:, 0] += pos_x
        obj_pos[:, 1] += pos_y
        obj_pos[:, 2] += pos_z
        
        # For inactive objects, move them far away
        obj_pos[~mask] = torch.tensor([-10.0, -10.0, -10.0], device=device).unsqueeze(0).repeat((~mask).sum(), 1)
        
        # Random orientation
        random_quat = _random_quaternion(num_envs, device)
        
        # Set object pose
        obj.set_root_pose_w(obj_pos, random_quat)


def _random_quaternion(num_envs: int, device: torch.device) -> torch.Tensor:
    """Generate random quaternions.
    
    Args:
        num_envs: Number of environments.
        device: Device to create tensors on.
        
    Returns:
        Random quaternions of shape (num_envs, 4) in (w, x, y, z) format.
    """
    # Generate random rotation using axis-angle representation
    # Random axis
    axis = torch.randn(num_envs, 3, device=device)
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    
    # Random angle
    angle = torch.rand(num_envs, device=device) * 2 * torch.pi
    
    # Convert to quaternion
    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle).unsqueeze(-1)
    
    quat = torch.cat([w.unsqueeze(-1), xyz], dim=1)
    return quat

