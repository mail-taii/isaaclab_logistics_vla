# 用 Isaac XR Teleop Sample Client (AVP) 对接 isaaclab_logistics_vla Benchmark

你已在 AVP 上安装好 **Isaac XR Teleop Sample Client**，下面是在 **workstation（bench）** 上还需要完成的步骤，以及如何与你的 benchmark 对接。

---

## 一、Workstation 端必须完成的步骤

### 1. 环境与依赖

- **Isaac Lab**：需要你当前跑 benchmark 用的那一份 Isaac Lab（例如 `IsaacLab-2.2.1` 或 `IsaacLab-Arena` 里的 submodule），且版本要与 AVP 客户端对应：
  - Isaac Lab **2.3.x** → 客户端 tag **v2.3.0**
  - Isaac Lab **2.2.x** → 客户端 tag **v2.2.0**
  - Isaac Lab **2.1.x** → 客户端 tag **v1.0.0**
- **Docker**：Docker 26.0+、Docker Compose 2.25+、[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)。
- **网络**：AVP 与 workstation 在同一局域网、互相能 ping 通（建议用独立 WiFi 6 路由器，避免机构网络隔离导致连不上）。

### 2. 放行 CloudXR 端口

在 workstation 上执行：

```bash
sudo ufw allow 47998:48000,48005,48008,48012/udp
sudo ufw allow 48010/tcp
```

### 3. 启动 CloudXR Runtime + Isaac Lab（二选一）

#### 方式 A：Docker Compose（推荐）

在 **Isaac Lab 仓库根目录**（即你的 `ISAACLAB_ROOT`，例如 `IsaacLab-2.2.1` 或 `IsaacLab-Arena/submodules/IsaacLab`）执行：

```bash
./docker/container.py start \
    --files docker-compose.cloudxr-runtime.patch.yaml \
    --env-file .env.cloudxr-runtime
```

按提示选择 X11 转发（需要看到 Isaac Sim 界面）。然后进入容器：

```bash
./docker/container.py enter base
```

在容器内再运行带 XR 的遥操作脚本（见下文「四、跑哪个任务」）。

#### 方式 B：本地跑 Isaac Lab + 单独起 CloudXR 容器

1. 在 Isaac Lab 仓库根目录创建共享目录并启动 CloudXR 容器：

```bash
mkdir -p $(pwd)/openxr
docker run -it --rm --name cloudxr-runtime \
    --user $(id -u):$(id -g) \
    --gpus=all \
    -e "ACCEPT_EULA=Y" \
    --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
    -p 48010:48010 -p 47998:47998/udp -p 47999:47999/udp -p 48000:48000/udp \
    -p 48005:48005/udp -p 48008:48008/udp -p 48012:48012/udp \
    nvcr.io/nvidia/cloudxr-runtime:5.0.1
```

2. **另开一个终端**，在要跑 Isaac Lab 的目录下设置环境变量后再启动 Isaac Lab：

```bash
export XDG_RUNTIME_DIR=/path/to/isaaclab/openxr/run
export XR_RUNTIME_JSON=/path/to/isaaclab/openxr/share/openxr/1/openxr_cloudxr.json
# 然后运行 isaaclab.sh -p ...（见第四节）
```

注意：方式 B 下 `openxr` 的路径要和 Docker 挂载的是同一目录。

### 4. 在 Isaac Sim 里开启 AR/OpenXR

- 在 Isaac Sim 界面找到 **Panel → AR**。
- **Selected Output Plugin** 选 **OpenXR**，**OpenXR Runtime** 选 **System OpenXR Runtime**。
- 点击 **Start AR**，看到双眼渲染且状态为 “AR profile is active” 即表示就绪，AVP 客户端可以连上来。

---

## 二、AVP 端操作（你已完成安装，仅需连接）

1. 打开 **Isaac XR Teleop Sample Client**。
2. 输入 **workstation 的 IP 地址**（与 AVP 同网段的那个）。
3. 点击 **Connect**，首次可能需允许本地网络、手部追踪等权限。
4. 连接成功后点击 **Play** 即可用手部控制仿真中的机器人；**Stop** / **Reset** 可暂停或重置。

---

## 三、先验证「官方任务」是否通

在 workstation（或 Docker 容器内）用 **Isaac Lab 自带的** 遥操作脚本和任务，确认 AVP ↔ CloudXR ↔ Isaac Lab 整条链路正常：

```bash
# 在 Isaac Lab 根目录下，激活对应 conda 后再执行
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --teleop_device handtracking \
    --enable_pinocchio \
    --xr
```

- 若这里能正常用手控制 GR1，说明 AVP 端和 CloudXR 配置都没问题。
- 然后再考虑把你自己的 benchmark 场景接进来。

---

## 四、在你的 Benchmark 场景里用 AVP 遥操作（需要开发）

你当前的 benchmark（isaaclab_logistics_vla）情况是：

- **已有**：Bunny 遥操作（AVP → Bunny → ROS2 qpos → `run_bunny_teleop.py`），驱动的是 **realman + 你的任务场景**（如 `Spawn_ds_st_sparse_EnvCfg`）。
- **Isaac XR 官方**：走的是 **CloudXR + OpenXR**，手部数据在 Isaac Lab 里通过 `OpenXRDevice` + Retargeter 直接驱动机器人，**不经过 ROS2**。

因此，若要在 **同一套 benchmark 场景**（realman、Spawn_ds_st 等）上使用 **Isaac XR Teleop Sample Client**，需要多做一步「桥接」：

1. **在 Isaac Lab 侧**（或 isaaclab_logistics_vla 扩展里）增加一个 **带 XR 的遥操作入口脚本**：
   - 启动时加上 `--xr`，并设置 `XDG_RUNTIME_DIR` / `XR_RUNTIME_JSON`（若用方式 B）。
   - 用你现有的 **register** 加载场景，例如：  
     `register.load_env_configs("Spawn_ds_st_sparse_EnvCfg")()`，而不是 `parse_env_cfg("Isaac-PickPlace-GR1T2-Abs-v0", ...)`。

2. **在环境配置里接上 OpenXR 手部遥操作**：
   - 官方 `teleop_se3_agent.py` 依赖 env_cfg 里的 **teleop_devices**，其中 `handtracking` 对应 `OpenXRDevice` + 手部 Retargeter（如 Se3RelRetargeter / 双臂 Retargeter）。
   - 你的场景目前是 **realman 双臂 + 夹爪**（17 维动作），和官方 GR1 单臂/人形不同，需要：
     - 要么在 env_cfg 中为你的任务增加 `teleop_devices`，配置 `OpenXRDevice` 和适合 **双臂 realman** 的 Retargeter；  
     - 要么在 Isaac Lab 里查是否有现成的双臂/多臂手部 Retargeter 可复用，否则需要自己实现或适配一套「手部位姿 → realman 17 维动作」的映射。

3. **和现有 Bunny 遥操作的关系**：
   - Bunny：AVP → Bunny 客户端 → bunny_teleop_server → ROS2 → `run_bunny_teleop.py`（不经过 CloudXR）。
   - Isaac XR：AVP 上的 App → CloudXR → Isaac Lab OpenXR → 同一套仿真场景。
   - 两套可以并存：同一台 workstation 上，要么跑「Bunny + run_bunny_teleop」，要么跑「CloudXR + 新的 XR 遥操作脚本」，选一种方式驱动你的 benchmark 场景。

**建议顺序**：  
先按第三节用官方任务确认 AVP 与 CloudXR 正常；再在 isaaclab_logistics_vla 或 Isaac Lab 里新增一个「XR + 你的场景」的脚本，并逐步把 `teleop_devices` + Retargeter 接到 realman 上。

### 你现在可以直接用的实现（已写进仓库）

我已经在 `/home/junzhe/isaaclab_logistics_vla` 里补齐了一个可跑的“XR teleop 变体”，不改动你原本的评估/训练动作空间：

- **新的 EnvCfg**：`Spawn_ds_st_sparse_XRTeleop_EnvCfg`  
  - 复用同一场景/奖励/观测，但把 `actions` 换成 **双臂 Differential IK + 双夹爪**（更适配 OpenXR 手部遥操作）  
  - 并在 `teleop_devices.handtracking` 里配置了双手的 `Se3RelRetargeter` + `GripperRetargeter`
  - 文件：`isaaclab_logistics_vla/tasks/ds_st_series/sparse_scene/xr_teleop_env_cfg.py`

- **新的启动脚本**：`scripts/run_xr_teleop.py`  
  - 启动 Isaac App、加载 `*_XRTeleop_EnvCfg`、创建 OpenXR teleop device  
  - 监听 AVP UI 发来的 `START/STOP/RESET`（Play/Stop/Reset）事件

启动命令（在 `isaaclab_logistics_vla` 仓库根目录执行，确保使用 `isaaclab.sh` 的 Python 环境）：

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/run_xr_teleop.py \
  --task_scene_name Spawn_ds_st_sparse_XRTeleop_EnvCfg \
  --num_envs 1 \
  --control_hz 45 \
  --asset_root_path /home/junzhe/Benchmark \
  --device cuda:0 \
  --xr
```

说明：
- `--xr` 会确保加载 XR experience，并在 Isaac Sim UI 里出现 AR Panel。
- 首次建议 `--num_envs 1`，避免 XR 渲染/编码负载过高。

---

## 五、简要检查清单

| 步骤 | 内容 |
|------|------|
| 1 | Workstation 已装 Docker + NVIDIA Container Toolkit，并放行 CloudXR 端口 |
| 2 | 用 Docker Compose 或方式 B 启动 CloudXR Runtime，并让 Isaac Lab 使用同一 `openxr` 目录（环境变量） |
| 3 | 在 Isaac Lab 里运行带 `--xr` 的脚本，并在 Isaac Sim 中 AR Panel 里点 Start AR |
| 4 | AVP 与 workstation 同网，AVP 上打开 App，输入 IP，Connect → Play |
| 5 | 先用官方任务 `Isaac-PickPlace-GR1T2-Abs-v0` 验证整条链路 |
| 6 | 再在 benchmark 中增加「XR + 你的场景 + realman Retargeter」的脚本与配置 |

官方文档入口：[Setting Up CloudXR Teleoperation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html)。
