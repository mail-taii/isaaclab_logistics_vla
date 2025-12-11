# 新任务开发模板

这是创建新 benchmark 任务的模板。复制此目录并重命名为你的任务名称。

## 目录结构

```
your_task_name/
├── README.md                    # 任务说明文档（本文件）
├── __init__.py                  # 环境注册
├── your_task_name_env.py        # 环境实现
├── your_task_name_env_cfg.py    # 环境配置
├── config/                      # 机器人配置
│   └── realman/
│       └── __init__.py
└── mdp/                         # MDP定义
    ├── __init__.py
    ├── observations.py          # 观测定义
    ├── rewards.py               # 奖励函数
    ├── terminations.py          # 终止条件
    └── metrics.py               # 评估指标
```

## 开发步骤

### 1. 复制模板

```bash
cd /home/junzhe/IsaacLab-2.2.1/source/isaaclab_logistics_vla/isaaclab_logistics_vla/tasks
cp -r TASK_TEMPLATE your_task_name
cd your_task_name
```

### 2. 重命名文件

将所有 `your_task_name` 替换为你的实际任务名称。

### 3. 实现任务

按照以下顺序实现：

1. **场景配置** (`your_task_name_env_cfg.py`)
   - 定义场景中的对象（机器人、物体、环境）
   - 配置观测空间
   - 配置动作空间

2. **MDP定义** (`mdp/`)
   - `observations.py`: 定义观测项
   - `rewards.py`: 定义奖励函数
   - `terminations.py`: 定义终止条件
   - `metrics.py`: 定义评估指标

3. **环境实现** (`your_task_name_env.py`)
   - 继承 `ManagerBasedRLEnv`
   - 实现指标追踪（如果需要）

4. **环境注册** (`__init__.py`)
   - 注册 Gym 环境

### 4. 测试

创建测试脚本验证环境是否正常工作。

## 任务要求

### 必须实现的接口

1. **观测接口**
   - 必须包含场景图像（如果任务需要视觉）
   - 必须包含机器人状态（关节位置、速度等）
   - 必须包含任务相关状态（物体位置、目标位置等）

2. **动作接口**
   - 动作空间必须明确定义
   - 动作维度必须固定

3. **指标接口**
   - 必须实现三个核心指标：
     - `grasping_success_rate`: 抓取成功率
     - `intent_accuracy`: 意图正确率
     - `task_success_rate`: 任务成功率
   - 指标在 episode 结束时记录到 `info["log"]` 中

### 指标计算规范

参考 `single_arm_sorting` 任务的实现：
- 指标在 `mdp/metrics.py` 中定义
- 在环境的 `step()` 方法中更新
- 在 episode 结束时记录

## 参考示例

查看 `single_arm_sorting` 任务作为参考：
- 场景配置：`single_arm_sorting_env_cfg.py`
- MDP定义：`single_arm_sorting/mdp/`
- 环境实现：`single_arm_sorting_env.py`
- 指标计算：`mdp/metrics.py`

## 测试要求

1. 环境可以正常创建和重置
2. 可以执行动作并获得观测
3. 指标可以正确计算和记录
4. 通过 `test_interface.py` 测试

## 提交要求

1. 完整的任务代码
2. README 文档（任务描述、使用方法）
3. 测试脚本
4. 示例运行结果
