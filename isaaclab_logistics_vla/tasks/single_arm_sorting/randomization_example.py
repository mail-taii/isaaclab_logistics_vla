# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example of using object randomization in single arm sorting task.

This example demonstrates how to:
1. Configure multiple objects with different types
2. Use external USD resources
3. Randomize object generation
"""

from isaaclab_logistics_vla.tasks.single_arm_sorting.config.realman import RealmanSingleArmSortingEnvCfg
from isaaclab_logistics_vla.tasks.single_arm_sorting.mdp.randomization import ObjectSpawnConfig, EventCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_logistics_vla.tasks.single_arm_sorting import mdp


@configclass
class RandomizedRealmanSingleArmSortingEnvCfg(RealmanSingleArmSortingEnvCfg):
    """Extended configuration with object randomization."""
    
    def __post_init__(self):
        """Post initialization with randomization."""
        # Call parent initialization first
        super().__post_init__()
        
        # Add randomization events
        # Example 1: Randomize single object position
        self.events.randomize_object_position = EventTerm(
            func=mdp.randomize_object_positions,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("object"),
                "source_cfg": SceneEntityCfg("source_area"),
                "pos_range": ((-0.15, 0.15), (-0.15, 0.15), (0.0, 0.15)),
            },
        )
        
        # Example 2: Randomize object properties
        self.events.randomize_object_properties = EventTerm(
            func=mdp.randomize_object_properties,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("object"),
                "mass_range": (0.05, 0.3),
                "scale_range": (0.7, 1.3),
            },
        )


# Example usage with multiple object configurations
def create_object_configs():
    """Create example object configurations.
    
    Returns:
        List of ObjectSpawnConfig instances.
    """
    configs = []
    
    # Configuration 1: Small red cuboid
    configs.append(
        ObjectSpawnConfig(
            object_type="cuboid",
            size=(0.08, 0.08, 0.08),
            mass=0.1,
            color=(0.8, 0.2, 0.2),
            pos_range=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)),
        )
    )
    
    # Configuration 2: Medium blue cuboid
    configs.append(
        ObjectSpawnConfig(
            object_type="cuboid",
            size=(0.12, 0.12, 0.12),
            mass=0.15,
            color=(0.2, 0.2, 0.8),
            pos_range=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)),
        )
    )
    
    # Configuration 3: Large green cuboid
    configs.append(
        ObjectSpawnConfig(
            object_type="cuboid",
            size=(0.15, 0.15, 0.15),
            mass=0.2,
            color=(0.2, 0.8, 0.2),
            pos_range=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)),
        )
    )
    
    # Configuration 4: External USD file (if available)
    # Uncomment and set path if you have external USD files
    # configs.append(
    #     ObjectSpawnConfig(
    #         object_type="usd",
    #         usd_path="/path/to/your/object.usd",
    #         mass=0.1,
    #         pos_range=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)),
    #     )
    # )
    
    return configs


if __name__ == "__main__":
    # Example: Create environment with randomization
    cfg = RandomizedRealmanSingleArmSortingEnvCfg()
    
    # You can also customize object configs
    object_configs = create_object_configs()
    
    print("Object randomization configured!")
    print(f"Number of object types: {len(object_configs)}")
    print("\nObject configurations:")
    for i, config in enumerate(object_configs):
        print(f"  {i+1}. Type: {config.object_type}, Size: {config.size}, Color: {config.color}")

