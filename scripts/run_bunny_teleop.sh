#!/bin/bash
# Bunny 遥操作启动脚本：在 Python 进程启动前设置 LD_LIBRARY_PATH，
# 确保 rclpy 能加载 librcl_action.so 等 Isaac Sim 内置 ROS2 库。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 获取 humble/lib 路径（优先从当前 conda env）
if [ -n "$CONDA_PREFIX" ]; then
    HUMBLE_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"
elif [ -n "$VIRTUAL_ENV" ]; then
    HUMBLE_LIB="$VIRTUAL_ENV/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"
else
    HUMBLE_LIB="$(python3 -c "
try:
    import isaacsim
    import os
    p = os.path.join(os.path.dirname(isaacsim.__file__), 'exts/isaacsim.ros2.bridge/humble/lib')
    print(p)
except Exception:
    exit(1)
" 2>/dev/null)"
fi

if [ -d "$HUMBLE_LIB" ]; then
    export ROS_DISTRO="${ROS_DISTRO:-humble}"
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
    export LD_LIBRARY_PATH="$HUMBLE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python scripts/run_bunny_teleop.py "$@"
