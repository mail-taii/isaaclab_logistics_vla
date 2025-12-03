# IsaacLab Logistics VLA 扩展包安装指南

## 安装步骤

### 1. 确保在项目根目录

```bash
cd /home/junzhe/IsaacLab-2.2.1
```

### 2. 安装扩展包

**推荐方式（使用 IsaacLab 的 Python 环境）：**

```bash
./isaaclab.sh -p -m pip install -e source/isaaclab_logistics_vla
```

**或者如果已激活 conda 环境：**

```bash
conda activate env_isaaclab  # 或你的环境名
pip install -e source/isaaclab_logistics_vla
```

### 3. 验证安装

```bash
# 使用 isaaclab.sh 测试
./isaaclab.sh -p source/isaaclab_logistics_vla/test_installation.py

# 或者在 Python 中
./isaaclab.sh -p
>>> import isaaclab_logistics_vla
>>> print(isaaclab_logistics_vla.__version__)
```

## 使用示例

```python
import isaaclab_logistics_vla
import gymnasium as gym

# 创建环境
env = gym.make("Isaac-Logistics-SingleArmSorting-Franka-v0")

# 使用环境
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## 目录结构

```
source/isaaclab_logistics_vla/
├── config/
│   └── extension.toml          # 扩展包配置
├── docs/
│   └── README.md                # 文档
├── isaaclab_logistics_vla/
│   ├── __init__.py              # 主包初始化
│   ├── tasks/                   # 任务定义
│   │   └── single_arm_sorting/ # 单臂分拣任务
│   ├── assets/                  # 资产配置
│   ├── mdp/                     # MDP 组件
│   ├── sensors/                 # 传感器扩展
│   └── utils/                   # 工具函数
└── setup.py                     # 安装脚本
```

## 故障排除

### 问题 1: 导入失败

**错误：** `ModuleNotFoundError: No module named 'isaaclab_logistics_vla'`

**解决：**
- 确保已安装扩展包：`./isaaclab.sh -p -m pip install -e source/isaaclab_logistics_vla`
- 确保使用正确的 Python 环境

### 问题 2: 依赖缺失

**错误：** `ImportError: cannot import name 'xxx'`

**解决：**
- 检查 `setup.py` 中的依赖是否已安装
- 运行：`./isaaclab.sh -p -m pip install -r source/isaaclab_logistics_vla/requirements.txt`（如果有）

### 问题 3: Gym 环境未注册

**错误：** `gym.error.UnregisteredEnv`

**解决：**
- 确保导入了 `isaaclab_logistics_vla` 包
- 检查 `config/franka/__init__.py` 中的 `gym.register()` 是否正确执行

## 下一步

1. 完善第一个任务（单臂分拣）
2. 添加视觉传感器
3. 添加语言指令支持
4. 扩展到其他任务

