## 在物流评估框架中接入 openpi（π0.5‑DROID）

本文档记录当前 `/home/junzhe/isaaclab_logistics_vla` 项目如何通过 `openpi` 的 WebSocket 服务做远程 VLA 推理，重点是和评估框架的对接细节。

---

### 一、整体架构

- **GPU 机 / bench**（本机）：
  - 安装并运行 `openpi`，在 **单张 GPU** 上启动 π₀.₅‑DROID Policy Server（WebSocket）。
- **评估进程（env_isaaclab 环境）**：
  - 使用 `ObservationBuilder` 构造 `ObservationDict`。
  - 用 `OpenPIRemotePolicy` 将 `ObservationDict` 封装成 openpi 期望的字典，通过 `openpi-client` 发送给远程 server。
  - 接收 openpi 返回的动作，并映射为当前机器人的动作空间，再交给 IsaacLab 环境执行。

数据流（单步）大致如下：

```text
ObservationBuilder (IsaacLab env)
    → ObservationDict
    → OpenPIRemotePolicy.predict()
        → openpi_client.WebsocketClientPolicy.infer()
            → openpi serve_policy.py (π0.5-DROID)
        ← {"actions": (horizon, 8)}
    → 映射为当前机器人 action_dim (例如 17 维 realman)
    → env.step(action)
```

---

### 二、openpi 侧部署（GPU 机）

#### 1. 安装 openpi 环境

```bash
cd /home/junzhe/openpi
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv run python -c "from openpi.training import config; from openpi.policies import policy_config; print('OK')"
```

#### 2. 启动 π₀.₅‑DROID Policy Server

只用一张 GPU（例如 0 号）：

```bash
cd /home/junzhe/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --env DROID
```

- `--env DROID` 等价于：
  - `config="pi05_droid"`
  - `dir="gs://openpi-assets/checkpoints/pi05_droid"`
- 第一次会自动下载 checkpoint 到 `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid`。
- 日志出现：

```text
server listening on 0.0.0.0:8000
```

即表示 WebSocket server 已在 `0.0.0.0:8000` 监听。

> 若 server 在其他机器上，只需将 client 端的 `OPENPI_HOST` / `OPENPI_PORT` 改为对应 IP/端口。

---

### 三、评估环境侧：安装 openpi-client

评估脚本运行在 Conda 环境 `env_isaaclab` 中，需要在该环境安装 `openpi-client`：

```bash
conda activate env_isaaclab
cd /home/junzhe/openpi/packages/openpi-client
pip install -e .

python -c "from openpi_client import websocket_client_policy; print('OK')"
```

输出 `OK` 即说明 `openpi-client` 安装成功。

---

### 四、在评估脚本中使用 openpi 策略

构造策略的入口在 `evaluation/evaluator/vla_evaluator.py`：

```python
if policy_name in ("openpi", "pi0", "openpi_remote"):
    from isaaclab_logistics_vla.evaluation.models.policy.openpi_remote_policy import OpenPIRemotePolicy
    return OpenPIRemotePolicy(action_dim=action_dim, device=device)
```

使用方式（示意）：

- 启动 IsaacLab 评估脚本时，指定：
  - `--policy openpi_remote`
  - `--robot_id realman_dual_left_arm`（或后续的 `ur5e`）
- 确保在同一台机上 openpi server 已在 `0.0.0.0:8000` 运行：
  - client 默认使用 `host="localhost"`、`port=8000`；
  - 也可以通过环境变量覆盖：
    - `OPENPI_HOST=xxx.xxx.xxx.xxx`
    - `OPENPI_PORT=8000`

---

### 五、数据对接细节（realman_dual_left_arm）

#### 1. ObservationDict 结构

`ObservationBuilder`（`evaluation/observation/builder.py`）构造的 `ObservationDict` 主要字段：

- `obs["vision"]["cameras"]`：`List[str]`，相机名列表（例如 `["head_camera", "ee_camera", "top_camera"]`）。
- `obs["vision"]["rgb"]`：`(C, num_envs, H, W, 3)`，C 为相机个数。
- `obs["robot_state"]["qpos"]` / `qvel`：`(num_envs, n_dof)`。
  - 对 `realman_dual_left_arm`，action_dim=17，对应字典打印为：

    ```python
    {
      'l_joint1': 0, 'r_joint1': 1,
      'l_joint2': 2, 'r_joint2': 3,
      'l_joint3': 4, 'r_joint3': 5,
      'l_joint4': 6, 'r_joint4': 7,
      'l_joint5': 8, 'r_joint5': 9,
      'l_joint6': 10, 'r_joint6': 11,
      'l_joint7': 12, 'r_joint7': 13,
      'left_gripper': 14, 'right_gripper': 15,
      'platform_joint': 16
    }
    ```

- `obs["instruction"]["text"]`：任务指令。

#### 2. OpenPIRemotePolicy：构造发往 openpi 的请求

文件：`evaluation/models/policy/openpi_remote_policy.py`。

每个 env 的单步预测：

1. **选两路相机作为 DROID 的 image / wrist_image**

   ```python
   vision = obs.get("vision") or {}
   cams = vision.get("cameras") or []

   # base camera：prefer_camera（默认 "head_camera"）或列表第一个
   base_cam_idx = 0
   if self.prefer_camera and cams and self.prefer_camera in cams:
       base_cam_idx = cams.index(self.prefer_camera)
   base_cam_name = cams[base_cam_idx] if cams else None

   # wrist camera：列表中的下一路（若只有一路则复用 base）
   if len(cams) >= 2:
       wrist_cam_idx = (base_cam_idx + 1) % len(cams)
   else:
       wrist_cam_idx = base_cam_idx
   wrist_cam_name = cams[wrist_cam_idx] if cams else None
   ```

   使用 `_pick_rgb_np` 从 `rgb` 里取单 env 的 `(H, W, 3)` 图像，并 `convert_to_uint8 + resize_with_pad` 到 224x224：

   ```python
   rgb_base = _pick_rgb_np(obs, env_id, prefer_camera=base_cam_name,  resize_hw=(image_size, image_size))
   rgb_wrist = _pick_rgb_np(obs, env_id, prefer_camera=wrist_cam_name, resize_hw=(image_size, image_size))
   ```

2. **构造 DROID 风格的状态**

   ```python
   state_np = _build_state_np(obs, env_id)  # qpos[env_id] + qvel[env_id]

   if state_np.shape[0] >= 8:
       joint_pos = state_np[:7]    # 近似左臂 7 关节
       gripper_pos = state_np[7:8] # 近似左手 gripper
   else:
       joint_pos = np.zeros((7,), dtype=np.float32)
       gripper_pos = np.zeros((1,), dtype=np.float32)
   ```

3. **按 π0.5‑DROID 的 DROIDInputs 约定构造 request**

   ```python
   prompt = _pick_prompt(obs, default_instruction)

   request = {
       "observation/exterior_image_1_left": rgb_base,
       "observation/wrist_image_left":      rgb_wrist,
       "observation/joint_position":        joint_pos,
       "observation/gripper_position":      gripper_pos,
       "prompt":                            prompt,
   }
   ```

4. **通过 WebSocket 调用 openpi server**

   ```python
   response = self._client.infer(request)
   act = np.asarray(response["actions"])  # 形状 (H, 8) 或 (8,)
   act0 = act[0] if act.ndim == 2 else act
   ```

5. **将 openpi 的 8 维动作映射到 realman 的 17 维动作空间**

   openpi π₀.₅‑DROID 输出：`[7 关节, 1 gripper]`，我们目前采用的简单映射是：

   ```python
   env_action = np.zeros((self.action_dim,), dtype=np.float32)  # 17 维

   # 左臂 7 关节：直接用前 7 维
   env_action[:7] = act0[:7].astype(np.float32)

   # 右臂 7 关节：保持 0（不动）
   # env_action[7:14] 默认为 0

   # 左手夹爪：用第 8 维的符号映射为二值开/闭命令
   g_raw = float(act0[7])
   env_action[14] = 1.0 if g_raw > 0.0 else -1.0

   # 右手夹爪 / 平台：保持 0（不动）
   # env_action[15] = 0.0
   # env_action[16] = 0.0

   actions_out[env_id] = torch.from_numpy(env_action).to(self.device)
   ```

   > 这使得 openpi 只控制左臂和左夹爪，右臂与平台保持静止（或由其他策略控制）。

---

### 六、后续扩展建议

- **更精确的关节映射**：当前的 joint/gripper 切分是「简单近似」，若要提高表现，可以根据 `qpos` 的真实布局（各关节索引）来精确定义映射。
- **单臂 / UR5e 适配**：
  - 对 FRANKA/realman 左臂，π0.5‑DROID 是拓扑接近的，相对容易迁移。
  - 对 UR5e，openpi 提供了 `examples/ur5` 的 fine-tune 模板，建议基于 `pi0_base` / `pi05_base` 自行训一个 `pi0_ur5` / `pi05_ur5`，再用类似的 remote policy 对接。
- **双臂控制**：若后续希望 openpi 同时控制双臂，可以考虑在 env 侧将动作拆为「左臂 + 右臂」两个 policy 调用，或在映射中加入对右臂的策略性利用（例如镜像左臂动作）。

当前这一版文档对应的实现包括：

- `openpi_remote_policy.py` 中的请求构造与动作映射逻辑；
- `vla_evaluator.py` 中对 `OpenPIRemotePolicy` 的调用；
- `robot_registry.py` 中 realman_dual_left_arm 的注册信息；
- openpi 侧使用 `serve_policy.py --env DROID` 的 π₀.₅‑DROID server。

