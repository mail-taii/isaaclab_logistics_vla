#!/bin/bash
# 启动 order_series 任务的脚本
# 使用方法: 
#   GUI模式: ./run_order_task.sh --num_envs 4
#   无头模式: CUDA_VISIBLE_DEVICES=2 HEADLESS=1 STREAM=1 ./run_order_task.sh --num_envs 4

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 尝试找到 IsaacLab 根目录
# 如果已设置环境变量，直接使用
if [ -n "$ISAACLAB_ROOT" ] && [ -f "$ISAACLAB_ROOT/isaaclab.sh" ]; then
    # 使用环境变量指定的路径
    :
elif [ -f "$SCRIPT_DIR/../IsaacLab-2.2.1/isaaclab.sh" ]; then
    # 项目目录的兄弟目录
    ISAACLAB_ROOT="$( cd "$SCRIPT_DIR/../IsaacLab-2.2.1" && pwd )"
elif [ -f "$SCRIPT_DIR/../../IsaacLab-2.2.1/isaaclab.sh" ]; then
    # 向上两级目录
    ISAACLAB_ROOT="$( cd "$SCRIPT_DIR/../../IsaacLab-2.2.1" && pwd )"
elif [ -f "/home/junzhe/IsaacLab-2.2.1/isaaclab.sh" ]; then
    # 默认路径
    ISAACLAB_ROOT="/home/junzhe/IsaacLab-2.2.1"
elif [ -f "$SCRIPT_DIR/../../isaaclab.sh" ]; then
    # 向上两级查找 isaaclab.sh
    ISAACLAB_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
elif [ -f "$SCRIPT_DIR/../../../isaaclab.sh" ]; then
    # 向上三级查找 isaaclab.sh
    ISAACLAB_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
else
    echo "[错误]: 未找到 isaaclab.sh"
    echo "[提示]: 请设置 ISAACLAB_ROOT 环境变量指向 IsaacLab 根目录，例如："
    echo "        export ISAACLAB_ROOT=/home/junzhe/IsaacLab-2.2.1"
    exit 1
fi

echo "[INFO]: 使用 IsaacLab: $ISAACLAB_ROOT"

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "[警告]: 未检测到conda环境，尝试激活 env_isaaclab..."
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate env_isaaclab
    else
        echo "[警告]: 未找到 conda，请手动激活环境"
    fi
fi

# 切换到项目目录
cd "$SCRIPT_DIR"

# 使用 isaaclab.sh 运行脚本，传递所有参数
"$ISAACLAB_ROOT/isaaclab.sh" -p run_order_task.py "$@"
