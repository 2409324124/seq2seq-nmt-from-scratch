import os
import sys
import torch

def get_conda_env():
    """获取当前 Conda 环境名称"""
    return os.environ.get('CONDA_DEFAULT_ENV', 'Base/System')

def get_cuda_status():
    """获取 GPU 硬件状态"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"CUDA Ready: {device_name} ({vram:.1f}GB)"
    return "CPU Mode (No GPU detected)"

def get_python_version():
    """获取简短的 Python 版本号"""
    import sys
    return f"Python {sys.version.split()[0]}"

def get_system_summary():
    """返回用于 GUI 状态栏的一行摘要"""
    env = get_conda_env()
    ver = get_python_version()
    hw = get_cuda_status()
    return f"📟 Current: {env} | {ver} | {hw}"

if __name__ == "__main__":
    print(get_system_summary())
