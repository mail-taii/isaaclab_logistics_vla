from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def get_logs_path():
    path = Path(__file__).resolve().parent.parent.parent.joinpath('logs')
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root():
    # 假设 path_utils.py 在项目根目录下的某个子文件夹中
    # 请根据实际层级调整 .parent 的数量，确保指向项目根目录
    return Path(__file__).resolve().parent.parent.parent

def get_env_order_info_path():
    """获取项目根目录下的 env_order_info 文件夹路径"""
    path = get_project_root().joinpath('env_order_info')
    path.mkdir(parents=True, exist_ok=True)
    return path