from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def get_logs_path():
    path = Path(__file__).resolve().parent.parent.parent.joinpath('logs')
    path.mkdir(parents=True, exist_ok=True)
    
    return path