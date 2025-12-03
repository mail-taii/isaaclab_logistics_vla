# IsaacLab Logistics VLA Benchmark

## Overview

This extension provides Vision-Language-Action (VLA) benchmark tasks for logistics scenarios, focusing on cargo grasping and transportation tasks.

## Features

- **Multiple Task Categories**: Push-pull sorting, mobile transportation, single/dual-arm coordination, random SKU sorting, and empty container handling
- **VLA Support**: Natural language instruction input with vision-language multimodal learning
- **Multiple Robot Platforms**: Support for Franka, UR, Kuka, and other manipulators
- **Complex Scenarios**: Warehouse and sorting center environments

## Installation

```bash
cd /path/to/IsaacLab-2.2.1
./isaaclab.sh -p -m pip install -e source/isaaclab_logistics_vla
```

## Quick Start

```python
import isaaclab_logistics_vla
import gymnasium as gym

# Create environment
env = gym.make("Isaac-Logistics-SingleArmSorting-v0")
```

## Task Categories

1. **Push-Pull Assisted Sorting** (推拉辅助分拣)
2. **Mobile Transportation Sorting** (本体移动搬运分拣)
3. **Single/Dual-Arm Coordinated Sorting** (单/双臂协同分拣)
4. **Random SKU Specialized Sorting** (随机 SKU 专项分拣)
5. **Empty Container Specialized Sorting** (空货箱专项分拣)

## Documentation

See the [API documentation](api.md) for detailed information.

## License

BSD-3-Clause

