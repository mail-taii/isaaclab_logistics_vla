#!/bin/bash
# 统一任务启动脚本
# 使用方法: ./run_task.sh --task Isaac-Realman-lift --num_envs 2

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ISAACLAB_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "[警告]: 未检测到conda环境，尝试激活 env_isaaclab..."
    eval "$(conda shell.bash hook)"
    conda activate env_isaaclab
fi

# 切换到项目目录
cd "$SCRIPT_DIR"

# 使用isaaclab.sh运行random_agent.py
"$ISAACLAB_ROOT/isaaclab.sh" -p random_agent.py "$@"

