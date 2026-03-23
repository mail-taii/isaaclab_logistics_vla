# 环境侧挂相机说明

本文档说明如何在任务场景里挂载相机，供 VLA 观测或多路 RGB 使用。流程分为三步：**① 配置相机**（位置/朝向/分辨率）、**② 在场景里挂上**、**③ 在观测里暴露**。

---

## 1. 相机配置（configs/camera_configs/）

每个机器人一套相机配置，用**类属性**定义若干 `CameraCfg`，例如头、手、顶视。

### 1.1 现有示例：Realman（realman.py）

```python
# configs/camera_configs/realman.py
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab_logistics_vla.utils.util import euler_to_quat_isaac

class RealmanCameraConfig:
    # 挂在机器人 link 下：prim_path 必须是 {ENV_REGEX_NS}/Robot/.../相机名
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link2/camera_link/head_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),           # 相对父 link 的平移 (m)
            rot=euler_to_quat_isaac(r=45, p=180, y=0, return_tensor=False),  # 欧拉 r,p,y 度，wxyz 四元数
        ),
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        width=640,
        height=480,
        data_types=["rgb"],
    )

    ee_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_left_hand/ee_camera",
        offset=CameraCfg.OffsetCfg(pos=(0, 0, 0.08), rot=...),
        ...
    )

    # 顶视：不挂在 Robot 下，挂到 env 命名空间下，用 offset 的 pos 表示世界系位置
    top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        offset=CameraCfg.OffsetCfg(pos=(1.0, 2.0, 4.0), rot=...),
        ...
    )
```

要点：

- **挂在机器人身上**：`prim_path` 写成 `{ENV_REGEX_NS}/Robot/<机器人在 USD 里的 link 路径>/<相机名>`，这样相机会随机器人动。`offset` 为相对该 link 的位姿（pos 米，rot 四元数 wxyz）。
- **顶视/固定相机**：`prim_path` 用 `{ENV_REGEX_NS}/top_camera` 等，父节点由场景挂到 env 下；`offset.pos` 一般为世界系下的安装位置。
- **欧拉转四元数**：用 `euler_to_quat_isaac(r, p, y, return_tensor=False)`，角度制，顺序 XYZ。

### 1.2 新机器人：新增文件并注册

1. 在 `configs/camera_configs/` 下新建 `你的机器人.py`，仿照 `realman.py` 写一个类，提供至少 `head_camera`、`ee_camera`、`top_camera` 三个类属性（名字可约定一致，便于统一用 `get_camera_config`）。
2. 在 `configs/camera_configs/__init__.py` 里：
   - `from . import 你的机器人`
   - 在 `CAMERA_CONFIG_REGISTRY` 里加一项，例如 `"your_robot": 你的机器人.YourRobotCameraConfig`。

之后即可用 `get_camera_config("your_robot")` 拿到这套配置。

---

## 2. 在场景里挂上相机（scene_cfg.py）

在任务的 **SceneCfg**（继承自 BaseOrderSceneCfg 或 InteractiveSceneCfg）里，把上一步的相机配置**挂到场景**上，这样 Isaac 才会真正 spawn 这些相机。

```python
# tasks/ss_st_series/sparse_scene/scene_cfg.py
from isaaclab_logistics_vla.configs.camera_configs import get_camera_config
from typing import ClassVar

@configclass
class Spawn_ss_st_sparse_SceneCfg(BaseOrderSceneCfg):
    robot: ArticulationCfg = ...

    # ① 按机器人取相机配置（类，提供 .head_camera / .ee_camera / .top_camera）
    _cameras: ClassVar = get_camera_config("realman")
    # ② 挂到场景上：属性名与 prim_path 末尾/观测里 SceneEntityCfg 用的名字一致
    head_camera = _cameras.head_camera
    ee_camera = _cameras.ee_camera
    top_camera = _cameras.top_camera
```

说明：

- `get_camera_config("realman")` 返回的是**配置类**（或实例），上面有 `head_camera`、`ee_camera`、`top_camera` 等 `CameraCfg`。
- 把这三个赋给 SceneCfg 的**同名属性**，Isaac Lab 会把这些 `CameraCfg` 当作场景中的传感器 spawn；`prim_path` 里的 `{ENV_REGEX_NS}` 会按环境 id 替换。
- 若你只想要其中一两路，只写需要的即可，例如只写 `head_camera = _cameras.head_camera`；观测里不用到的可以不挂。

---

## 3. 在观测里暴露（observation_cfg.py）

挂上相机后，还要在 **ObservationCfg** 里用 `mdp.image` + `SceneEntityCfg("相机名")` 把图像放进观测，评估侧或策略才能读到。

```python
# tasks/ss_st_series/sparse_scene/observation_cfg.py
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_logistics_vla.tasks import mdp

@configclass
class ObservationsCfg:
    @configclass
    class CamerasCfg(ObsGroup):
        head_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False},
        )
        ee_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "rgb", "normalize": False},
        )
        top_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "data_type": "rgb", "normalize": False},
        )
```

- `SceneEntityCfg("head_camera")` 里的 **"head_camera"** 必须和 scene_cfg 里挂的**属性名**一致（即 `head_camera = _cameras.head_camera` 里的 `head_camera`）。
- 若需要 depth/segmentation，在对应 `CameraCfg` 里把 `data_types` 改为 `["rgb", "depth"]` 等，并在这里增加 `data_type="depth"` 的 ObsTerm。

---

## 4. 小结：环境侧加相机的三步

| 步骤 | 位置 | 操作 |
|------|------|------|
| 1. 配置 | `configs/camera_configs/你的机器人.py` + `__init__.py` 注册 | 定义 `CameraCfg`（prim_path、offset、分辨率、data_types），并 `get_camera_config("key")` 可用 |
| 2. 挂载 | 任务 `scene_cfg.py` | `_cameras = get_camera_config("key")`，然后 `head_camera = _cameras.head_camera` 等挂到场景 |
| 3. 观测 | 任务 `observation_cfg.py` | 用 `ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("head_camera"), ...})` 暴露图像 |

**prim_path 约定**：挂机器人身上用 `{ENV_REGEX_NS}/Robot/<link 路径>/<相机名>`；固定顶视用 `{ENV_REGEX_NS}/top_camera`。观测里的 `SceneEntityCfg("相机名")` 与 scene 里属性名一致即可。
