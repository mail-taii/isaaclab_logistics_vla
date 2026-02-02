# VLABench Evaluation 架构设计文档

本文档详细介绍 VLABench 的评估系统整体设计，包括 VLA（Vision-Language-Action）和 VLM（Vision-Language Model）的接入方式，帮助你理解并复用这套设计到自己的 benchmark 上。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [VLA 评估系统](#2-vla-评估系统)
3. [VLM 评估系统](#3-vlm-评估系统)
4. [核心设计原则](#4-核心设计原则)
5. [如何在自己的 Benchmark 上复用](#5-如何在自己的-benchmark-上复用)

---

## 1. 整体架构概览

VLABench 的评估系统分为两个相对独立的子系统：

```
┌─────────────────────────────────────────────────────────────┐
│                    VLABench Evaluation System                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   VLA Evaluation     │      │   VLM Evaluation     │    │
│  │  (策略执行评估)       │      │  (技能序列生成评估)   │    │
│  ├──────────────────────┤      ├──────────────────────┤    │
│  │ Evaluator (base.py)   │      │ VLMEvaluator         │    │
│  │   ↓                   │      │   ↓                  │    │
│  │ Policy Interface      │      │ BaseVLM Interface    │    │
│  │   ↓                   │      │   ↓                  │    │
│  │ OpenVLA / Gr00t / ... │      │ LLaVA / GPT-4V / ...│    │
│  └──────────────────────┘      └──────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 两个评估系统的区别

| 维度 | VLA 评估 | VLM 评估 |
|------|---------|---------|
| **评估对象** | 机器人策略（Policy） | 视觉语言模型（VLM） |
| **输入** | 环境观测（图像+状态+指令） | 静态图像+指令（可能带 few-shot） |
| **输出** | 动作（EE 或 joint） | 技能序列（skill sequence）JSON |
| **评估方式** | 在仿真环境中执行，统计成功率 | 对比输出格式和内容，无需执行 |
| **适用场景** | 需要实际控制机器人的任务 | 序列独立任务（planning / reasoning） |
| **Evaluator** | `Evaluator` (base.py) | `VLMEvaluator` (vlm.py) |
| **模型接口** | `Policy` | `BaseVLM` |

---

## 2. VLA 评估系统

### 2.1 核心组件

#### 2.1.1 Evaluator（评估器）

**位置**：`VLABench/evaluation/evaluator/base.py`

**职责**：
- 管理评估流程（任务循环、episode 循环、step 循环）
- 与环境交互（获取观测、执行动作）
- 调用策略接口（`policy.predict`）
- 统计指标（success_rate、progress_score、intention_score）

**关键方法**：

```python
class Evaluator:
    def __init__(self, tasks, n_episodes, episode_config=None, 
                 max_substeps=1, metrics=["success_rate"], save_dir=None, ...):
        # 初始化评估配置
        
    def evaluate(self, agent):
        """
        主评估循环
        - 遍历所有任务
        - 每个任务运行 n_episodes 次
        - 调用 evaluate_single_episode
        - 计算并保存指标
        """
        
    def evaluate_single_episode(self, agent, task_name, episode_id, 
                                 episode_config, seed=42, max_episode_length=200, **kwargs):
        """
        单个 episode 的执行循环
        - 创建/重置环境
        - 每步：获取观测 → 调用 agent.predict → 执行动作
        - 直到任务成功或达到最大步数
        - 返回 episode 信息（success, progress, intention_score 等）
        """
```

**Evaluator 的核心循环**：

```python
# 伪代码
for task in tasks:
    for episode in range(n_episodes):
        env = load_env(task)
        env.reset()
        
        while not done and step < max_episode_length:
            # 1. 获取环境观测
            observation = env.get_observation()
            observation["instruction"] = env.task.get_instruction()
            observation["last_action"] = last_action
            
            # 2. 调用策略（不关心策略内部实现）
            if agent.control_mode == "ee":
                pos, euler, gripper = agent.predict(observation, unnorm_key=...)
                # 通过 IK 转成关节动作
            elif agent.control_mode == "joint":
                qpos, gripper = agent.predict(observation, ...)
                # 直接使用关节动作
            
            # 3. 执行动作
            timestep = env.step(action)
            
            # 4. 检查是否完成
            if timestep.last():
                success = True
                break
        
        # 5. 记录结果
        task_infos.append({
            "success": success,
            "progress_score": env.get_task_progress(),
            "intention_score": env.get_intention_score(),
            ...
        })
    
    # 6. 计算任务指标
    metrics[task] = compute_metric(task_infos)
```

#### 2.1.2 Policy 接口（策略接口）

**位置**：`VLABench/evaluation/model/policy/base.py`

**统一接口定义**：

```python
class Policy:
    def __init__(self, model):
        self.model = model
    
    def reset(self):
        """每个 episode 开始前重置内部状态"""
        pass
    
    def predict(self, obs, **kwargs):
        """
        根据观测预测动作
        
        Args:
            obs: 环境观测字典，包含：
                - obs["rgb"]: 多视角图像列表
                - obs["ee_state"]: 末端执行器状态
                - obs["instruction"]: 自然语言指令
                - obs["last_action"]: 上一步动作（可选）
                - ...
            **kwargs: 其他参数（如 unnorm_key, max_episode_length）
        
        Returns:
            - 如果 control_mode == "ee":
                (target_pos, target_euler, gripper_state)
            - 如果 control_mode == "joint":
                (qpos, gripper_state)
        """
        pass
    
    @property
    def name(self):
        """策略名称，用于日志"""
        return "Policy"
    
    @property
    def control_mode(self):
        """控制模式：'ee' 或 'joint'"""
        return "ee"
```

**关键设计点**：

1. **接口最小化**：Evaluator 只依赖 `reset()` 和 `predict()`，不关心内部实现
2. **控制模式分离**：通过 `control_mode` 属性区分末端控制和关节控制
3. **观测格式统一**：所有策略接收相同的 `observation` 字典格式
4. **动作格式统一**：所有策略返回相同格式的动作（EE 或 joint）

#### 2.1.3 已实现的 Policy 示例

**OpenVLA**（本地 HF 模型）：
- 加载本地 HuggingFace 格式的 OpenVLA 模型
- 在 `predict` 中：图像+指令 → Processor → 模型 → Δaction → 叠加到当前状态

**Gr00t**（远程 ZMQ 服务）：
- 通过 ZeroMQ 与远程 Gr00t 服务器通信
- 在 `predict` 中：观测 → 打包成 Gr00t 格式 → 发送请求 → 接收 action chunk → 展开为单步动作

**OpenPi**（远程 WebSocket 服务）：
- 通过 WebSocket + msgpack 与远程 OpenPi 服务器通信
- 类似 Gr00t，但通信协议不同

**RandomPolicy**（随机基准）：
- 纯本地实现，用于 sanity check

### 2.2 数据流

```
┌─────────────┐
│  Environment │ (MuJoCo)
│  (VLABench)  │
└──────┬───────┘
       │ get_observation()
       │ (图像 + 状态 + 指令)
       ↓
┌─────────────┐
│  Evaluator  │
└──────┬───────┘
       │ policy.predict(observation)
       ↓
┌─────────────┐
│   Policy    │ (OpenVLA / Gr00t / ...)
│  Interface  │
└──────┬───────┘
       │ 内部处理：
       │ - 观测对齐（VLABench → 模型输入格式）
       │ - 调用模型/服务
       │ - 输出对齐（模型输出 → 统一动作格式）
       ↓
┌─────────────┐
│   Action    │ (pos, euler, gripper) 或 (qpos, gripper)
└──────┬───────┘
       │
       ↓
┌─────────────┐
│  Evaluator  │
│  (IK 转换)   │
└──────┬───────┘
       │ env.step(action)
       ↓
┌─────────────┐
│  Environment │
└─────────────┘
```

### 2.3 如何接入新的 VLA

**步骤 1：实现 Policy 接口**

在 `VLABench/evaluation/model/policy/` 下创建新文件，例如 `myvla.py`：

```python
from VLABench.evaluation.model.policy.base import Policy

class MyVLAPolicy(Policy):
    def __init__(self, model_path, device="cuda", **kwargs):
        # 加载你的模型
        self.model = load_your_model(model_path)
        self.device = device
        super().__init__(self.model)
    
    @property
    def name(self):
        return "MyVLA"
    
    @property
    def control_mode(self):
        return "ee"  # 或 "joint"
    
    def reset(self):
        """重置内部状态（如 action buffer）"""
        self.timestep = 0
        # ...
    
    def predict(self, obs, **kwargs):
        """
        核心预测逻辑
        
        1. 从 obs 提取所需信息（图像、状态、指令）
        2. 转换成你的模型输入格式
        3. 调用模型
        4. 转换成统一输出格式
        """
        # 提取观测
        rgb = obs["rgb"][cam_index]  # 选择视角
        instruction = obs["instruction"]
        ee_state = obs["ee_state"]
        
        # 转换成模型输入
        model_input = your_preprocessing(rgb, instruction, ee_state)
        
        # 调用模型
        model_output = self.model(model_input)
        
        # 转换成统一格式
        target_pos = model_output["pos"]
        target_euler = model_output["euler"]
        gripper_state = model_output["gripper"]
        
        return target_pos, target_euler, gripper_state
```

**步骤 2：在评估脚本中注册**

在 `scripts/evaluate_policy.py` 中添加：

```python
elif args.policy.lower() == "myvla":
    from VLABench.evaluation.model.policy.myvla import MyVLAPolicy
    policy = MyVLAPolicy(
        model_path=args.model_path,
        device=args.device,
        ...
    )
```

**步骤 3：运行评估**

```bash
python scripts/evaluate_policy.py \
    --tasks select_toy \
    --n-episode 5 \
    --policy myvla \
    --model_path /path/to/your/model \
    ...
```

**关键点**：
- 你只需要实现 `predict` 方法，把 VLABench 的观测转成你的模型输入，再把模型输出转成统一动作格式
- Evaluator 会自动处理环境交互、指标统计、结果保存等

---

## 3. VLM 评估系统

### 3.1 核心组件

#### 3.1.1 VLMEvaluator（VLM 评估器）

**位置**：`VLABench/evaluation/evaluator/vlm.py`

**职责**：
- 管理 VLM 评估流程（任务循环、example 循环）
- 构建输入（图像+指令，支持 few-shot）
- 调用 VLM 接口（`vlm.evaluate`）
- 解析和验证输出（技能序列 JSON）
- 统计指标（格式正确率、内容准确率等）

**关键方法**：

```python
class VLMEvaluator(Evaluator):
    def __init__(self, tasks, n_episodes, data_path="./dataset", 
                 save_path="./output", language="en"):
        # 初始化数据路径、prompt 模板等
        
    def evaluate(self, vlm, task_list=None, save_interval=1, 
                 few_shot_num=0, with_CoT=False, eval_dim="default"):
        """
        主评估循环
        - 遍历所有任务和 examples
        - 构建输入（可能包含 few-shot examples）
        - 调用 vlm.evaluate
        - 解析输出并保存
        """
        
    def build_input(self, task_name, example_num, few_shot_num=0):
        """
        构建 VLM 输入
        - 加载输入图像、带标签图像、指令
        - 如果 few_shot_num > 0，随机采样 few-shot examples
        - 返回 prepared_input 字典
        """
        
    def get_single_answer(self, task_name, example_num, vlm, 
                          few_shot_num=0, with_CoT=False):
        """
        获取单个 example 的 VLM 输出
        - 构建输入
        - 调用 vlm.evaluate
        - 返回解析后的输出（skill_sequence 或 format_error）
        """
```

**VLMEvaluator 的核心循环**：

```python
# 伪代码
for task_name in task_list:
    for example_num in range(num_examples):
        # 1. 构建输入
        prepared_input = self.build_input(task_name, example_num, few_shot_num)
        # prepared_input 包含：
        # - pre_prompt: 系统提示
        # - shot_input_pic / shot_input_pic_gt / shot_input_instruction / shot_output (few-shot)
        # - input_pic / input_pic_gt / input_instruction (当前任务)
        
        # 2. 调用 VLM
        answer = vlm.evaluate(prepared_input, language, with_CoT)
        # answer 应该是：
        # - {"skill_sequence": [...]} 或
        # - {"format_error": "..."}
        
        # 3. 保存结果
        model_output[task_name][example_num] = answer
        
        # 4. 定期保存（避免中断丢失）
        if step % save_interval == 0:
            save_to_json(model_output)
```

#### 3.1.2 BaseVLM 接口（VLM 接口）

**位置**：`VLABench/evaluation/model/vlm/base.py`

**统一接口定义**：

```python
class BaseVLM:
    def __init__(self):
        self.name = self.get_name()
    
    def evaluate(self, input_dict, language, with_CoT=False):
        """
        根据输入生成技能序列
        
        Args:
            input_dict: 包含以下键的字典：
                - "pre_prompt": 系统提示文本
                - "shot_input_pic": {str: path} few-shot 输入图像路径
                - "shot_input_pic_gt": {str: path} few-shot 带标签图像路径
                - "shot_input_instruction": {str: str} few-shot 指令
                - "shot_output": {str: dict} few-shot 输出（技能序列）
                - "input_pic": 当前任务输入图像路径
                - "input_pic_gt": 当前任务带标签图像路径
                - "input_instruction": 当前任务指令
            language: "en" 或 "zh"
            with_CoT: 是否使用 Chain-of-Thought
        
        Returns:
            dict: 包含以下键之一：
                - "skill_sequence": 技能序列列表（格式正确的输出）
                - "format_error": 错误信息（格式错误）
        """
        raise NotImplementedError
    
    def get_name(self):
        """返回模型名称"""
        return "BaseVLM"
```

**关键设计点**：

1. **输入格式统一**：所有 VLM 接收相同的 `input_dict` 结构
2. **输出格式统一**：所有 VLM 返回相同格式（`skill_sequence` 或 `format_error`）
3. **支持多语言**：通过 `language` 参数支持中英文
4. **支持 Few-shot 和 CoT**：通过 `few_shot_num` 和 `with_CoT` 参数控制

#### 3.1.3 已实现的 VLM 示例

**LLaVA**（本地 HF 模型）：
- 加载本地 LLaVA 模型和 Processor
- 在 `evaluate` 中：构建多模态 prompt → 调用模型 → 解析 JSON 输出

**GPT-4V / Claude / Gemini**（API 调用）：
- 通过 API 调用云端模型
- 在 `evaluate` 中：构建 prompt → 发送 API 请求 → 解析响应

**Qwen-VL / InternVL / MiniCPM**（本地或 API）：
- 类似实现，但模型不同

### 3.2 数据流

```
┌─────────────┐
│  Dataset     │ (静态图像 + 指令)
│  (本地文件)  │
└──────┬───────┘
       │ load_single_input()
       │ (图像路径 + 指令文本)
       ↓
┌─────────────┐
│VLMEvaluator │
│ build_input │
└──────┬───────┘
       │ prepared_input (可能包含 few-shot)
       │ vlm.evaluate(input_dict, language, with_CoT)
       ↓
┌─────────────┐
│  BaseVLM    │ (LLaVA / GPT-4V / ...)
│  Interface  │
└──────┬───────┘
       │ 内部处理：
       │ - 构建多模态 prompt（文本+图像）
       │ - 调用模型/API
       │ - 解析输出（提取 JSON）
       ↓
┌─────────────┐
│   Output    │ {"skill_sequence": [...]} 或 {"format_error": "..."}
└──────┬───────┘
       │
       ↓
┌─────────────┐
│VLMEvaluator │
│ (验证+保存)  │
└─────────────┘
```

### 3.3 如何接入新的 VLM

**步骤 1：实现 BaseVLM 接口**

在 `VLABench/evaluation/model/vlm/` 下创建新文件，例如 `myvlm.py`：

```python
from VLABench.evaluation.model.vlm.base import BaseVLM, get_ti_list

class MyVLM(BaseVLM):
    def __init__(self):
        # 加载你的模型
        self.model = load_your_vlm_model()
        self.processor = load_your_processor()
        super().__init__()
    
    def get_name(self):
        return "MyVLM"
    
    def evaluate(self, input_dict, language, with_CoT=False):
        """
        核心评估逻辑
        
        1. 使用 get_ti_list 构建文本+图像交替列表
        2. 转换成你的模型输入格式
        3. 调用模型
        4. 解析输出为 skill_sequence 或 format_error
        """
        # 使用工具函数构建文本+图像列表
        ti_list = get_ti_list(input_dict, language, with_CoT)
        
        # 转换成你的模型输入
        model_input = your_preprocessing(ti_list)
        
        # 调用模型
        raw_output = self.model.generate(model_input)
        
        # 解析输出（尝试提取 JSON）
        try:
            json_data = extract_json_from_output(raw_output)
            skill_sequence = json.loads(json_data)
            return {"skill_sequence": skill_sequence}
        except:
            return {"format_error": "format_error"}
```

**步骤 2：在评估脚本中注册**

在 `scripts/evaluate_vlm.py`（如果存在）或创建新脚本：

```python
from VLABench.evaluation.model.vlm.myvlm import MyVLM
from VLABench.evaluation.evaluator.vlm import VLMEvaluator

evaluator = VLMEvaluator(tasks=..., data_path=..., save_path=...)
vlm = MyVLM()
evaluator.evaluate(vlm, task_list=..., few_shot_num=0, with_CoT=False)
```

**步骤 3：运行评估**

```bash
python scripts/evaluate_vlm.py \
    --tasks task1 task2 \
    --data_path ./dataset \
    --save_path ./output \
    --vlm myvlm \
    --few_shot_num 2 \
    --with_cot
```

**关键点**：
- 使用 `get_ti_list` 工具函数可以自动处理 few-shot 和 CoT 的 prompt 构建
- 输出必须是 `{"skill_sequence": [...]}` 或 `{"format_error": "..."}` 格式
- VLMEvaluator 会自动处理数据加载、结果保存、指标统计等

---

## 4. 核心设计原则

### 4.1 接口与实现分离

- **Evaluator / VLMEvaluator** 只依赖统一的接口（`Policy` / `BaseVLM`）
- **模型实现**（OpenVLA / Gr00t / LLaVA / GPT-4V）完全独立，可以自由替换
- **评估流程**（环境交互、指标统计、结果保存）与模型解耦

### 4.2 统一的数据格式

- **VLA 输入**：统一的 `observation` 字典格式
- **VLA 输出**：统一的动作格式（EE 或 joint）
- **VLM 输入**：统一的 `input_dict` 格式
- **VLM 输出**：统一的输出格式（`skill_sequence` 或 `format_error`）

### 4.3 可扩展性

- **添加新模型**：只需实现接口，无需修改 Evaluator
- **添加新指标**：在 Evaluator 的 `compute_metric` 中添加
- **添加新任务**：在环境层面添加，Evaluator 自动支持

### 4.4 容错与恢复

- **VLA 评估**：单个 episode 失败不影响其他 episode
- **VLM 评估**：支持断点续传（`save_interval`），避免中断丢失进度

---

## 5. 如何在自己的 Benchmark 上复用

### 5.1 复用 VLA 评估系统

**步骤 1：定义你的 Policy 接口**

```python
# my_benchmark/evaluation/policy/base.py
class Policy:
    def reset(self):
        pass
    
    def predict(self, obs, **kwargs):
        """
        返回 (target_pos, target_euler, gripper_state) 或 (qpos, gripper_state)
        """
        pass
    
    @property
    def control_mode(self):
        return "ee"  # 或 "joint"
```

**步骤 2：实现你的 Evaluator**

```python
# my_benchmark/evaluation/evaluator.py
class Evaluator:
    def __init__(self, tasks, n_episodes, ...):
        self.tasks = tasks
        self.n_episodes = n_episodes
        # ...
    
    def evaluate(self, policy):
        for task in self.tasks:
            for episode in range(self.n_episodes):
                # 创建你的环境
                env = your_env_factory(task)
                env.reset()
                
                while not done:
                    # 获取观测（你的环境格式）
                    obs = env.get_observation()
                    
                    # 调用策略（统一接口）
                    if policy.control_mode == "ee":
                        pos, euler, gripper = policy.predict(obs)
                        action = your_ik_solver(pos, euler, gripper)
                    else:
                        qpos, gripper = policy.predict(obs)
                        action = np.concatenate([qpos, gripper])
                    
                    # 执行动作
                    obs, reward, done, info = env.step(action)
                
                # 记录结果
                results.append(info)
        
        # 计算指标
        return self.compute_metrics(results)
```

**步骤 3：接入你的模型**

```python
# my_benchmark/evaluation/policy/my_model.py
from my_benchmark.evaluation.policy.base import Policy

class MyModelPolicy(Policy):
    def __init__(self, model_path):
        self.model = load_your_model(model_path)
    
    def predict(self, obs, **kwargs):
        # 你的观测格式 → 模型输入
        model_input = your_preprocessing(obs)
        
        # 调用模型
        model_output = self.model(model_input)
        
        # 模型输出 → 统一动作格式
        return your_postprocessing(model_output)
```

**关键点**：
- 保持 `Policy` 接口简单（`reset` + `predict`）
- 在 `predict` 内部处理格式转换（你的观测 → 模型输入，模型输出 → 统一动作）
- Evaluator 只负责环境交互和指标统计，不关心模型细节

### 5.2 复用 VLM 评估系统

**步骤 1：定义你的 BaseVLM 接口**

```python
# my_benchmark/evaluation/vlm/base.py
class BaseVLM:
    def evaluate(self, input_dict, **kwargs):
        """
        返回 {"skill_sequence": [...]} 或 {"format_error": "..."}
        """
        raise NotImplementedError
    
    def get_name(self):
        return "BaseVLM"
```

**步骤 2：实现你的 VLMEvaluator**

```python
# my_benchmark/evaluation/vlm_evaluator.py
class VLMEvaluator:
    def __init__(self, data_path, save_path, ...):
        self.data_path = data_path
        self.save_path = save_path
        # ...
    
    def evaluate(self, vlm, task_list, **kwargs):
        for task in task_list:
            for example in load_examples(task):
                # 构建输入（你的格式）
                input_dict = self.build_input(task, example, **kwargs)
                
                # 调用 VLM（统一接口）
                output = vlm.evaluate(input_dict, **kwargs)
                
                # 验证和保存
                if "skill_sequence" in output:
                    # 格式正确，可以进一步验证内容
                    pass
                else:
                    # 格式错误
                    pass
                
                results[task][example] = output
        
        return results
```

**步骤 3：接入你的 VLM**

```python
# my_benchmark/evaluation/vlm/my_vlm.py
from my_benchmark.evaluation.vlm.base import BaseVLM

class MyVLM(BaseVLM):
    def __init__(self):
        self.model = load_your_vlm()
        super().__init__()
    
    def evaluate(self, input_dict, **kwargs):
        # 你的输入格式 → 模型输入
        model_input = your_preprocessing(input_dict)
        
        # 调用模型
        raw_output = self.model.generate(model_input)
        
        # 解析输出
        try:
            skill_sequence = parse_output(raw_output)
            return {"skill_sequence": skill_sequence}
        except:
            return {"format_error": "format_error"}
```

**关键点**：
- 保持 `BaseVLM` 接口简单（`evaluate` + `get_name`）
- 在 `evaluate` 内部处理格式转换
- VLMEvaluator 只负责数据加载、结果保存、指标统计

### 5.3 设计检查清单

在实现你自己的评估系统时，确保：

- [ ] **接口最小化**：Evaluator 只依赖最少的接口方法
- [ ] **格式统一**：所有模型接收/返回统一的数据格式
- [ ] **实现独立**：模型实现可以自由替换，不影响评估流程
- [ ] **容错处理**：单个样本/episode 失败不影响整体评估
- [ ] **结果可复现**：支持固定 seed、保存详细结果
- [ ] **指标可扩展**：容易添加新指标
- [ ] **文档清晰**：接口文档、使用示例、接入指南

---

## 6. 总结

VLABench 的评估系统通过**接口与实现分离**的设计，实现了：

1. **VLA 评估**：统一的 `Policy` 接口 + `Evaluator` 流程，支持本地模型和远程服务
2. **VLM 评估**：统一的 `BaseVLM` 接口 + `VLMEvaluator` 流程，支持多种 VLM 模型
3. **易于扩展**：添加新模型只需实现接口，无需修改评估流程
4. **易于复用**：核心设计原则可以应用到任何 benchmark

**核心思想**：**评估流程与模型实现解耦，通过统一接口实现灵活扩展**。

你可以参考这个设计，在自己的 benchmark 上实现类似的评估系统。
