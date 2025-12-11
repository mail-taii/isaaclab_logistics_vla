# Benchmark 方法接入目录

这个目录用于存放接入 benchmark 的外部方法/算法。

## 目录结构

```
method/
├── README.md          # 本文件
├── test_interface.py  # 接口测试脚本（随机动作）
└── your_method.py     # 你的方法实现（示例）
```

## 接口规范

### 输入

- **动作 (Action)**: `torch.Tensor` 或 `numpy.ndarray`
  - 形状: `(num_envs, action_dim)` 或 `(action_dim,)`
  - 内容: 机器人的关节位置和夹爪动作

### 输出

环境返回的观测、奖励、终止标志和额外信息：

#### 1. 观测 (Observation)

包含以下信息：
- **场景画面 (RGB Image)**: 场景相机拍摄的图像
- **机器人当前位姿 (Robot Pose)**:
  - 关节位置 (joint positions)
  - 关节速度 (joint velocities)
  - 末端执行器位置 (end-effector position)
  - 物体位置 (object position)
  - 目标位置 (target position)

#### 2. 指标 (Metrics)

在 `info["log"]` 中：

- **抓取成功率 (Grasping Success Rate)**: 物体是否被成功抓取
- **意图正确率 (Intent Accuracy)**: 机器人是否按照正确意图执行
- **任务成功率 (Task Success Rate)**: 整个任务是否成功完成

指标在 episode 结束时记录到 `info["log"]` 中。

## 使用方法

### 1. 测试接口

运行测试脚本验证接口是否正常：

```bash
cd /home/junzhe/IsaacLab-2.2.1
./isaaclab.sh -p source/isaaclab_logistics_vla/method/test_interface.py
```

### 2. 实现你的方法

创建一个新的方法文件，例如 `your_method.py`：

```python
import torch
import gymnasium as gym

class YourMethod:
    def __init__(self):
        # 初始化你的模型/策略
        pass
    
    def predict(self, observation, deterministic=True):
        """根据观测预测动作"""
        # 你的策略逻辑
        # observation 包含图像和状态信息
        action = your_model(observation)
        return action
```

### 3. 运行你的方法

```python
import gymnasium as gym
from your_method import YourMethod

env = gym.make("Isaac-Logistics-SingleArmSorting-Realman-v0")
method = YourMethod()

obs, info = env.reset()
for step in range(1000):
    action = method.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 从 info 中获取指标
    if "log" in info and "metrics" in info["log"]:
        metrics = info["log"]["metrics"]
        print(f"指标: {metrics}")
    
    if terminated.any() or truncated.any():
        obs, info = env.reset()
```

## 环境信息

- **环境名称**: `Isaac-Logistics-SingleArmSorting-Realman-v0`
- **场景**: 操作台 + 两个箱子（源箱子和目标箱子）+ Realman 机器人
- **任务**: 从源箱子抓取物体并放置到目标箱子

## 注意事项

1. 环境需要 `--enable_cameras` 标志才能使用相机
2. 观测格式取决于环境的观测配置（可能是一个向量或字典）
3. 指标在 episode 结束时才会记录到 `info["log"]` 中

