# File utilities
# 文件工具模块

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Union, Optional


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    
    Returns:
        目录路径对象
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Union[str, Path], 
              pattern: str = "*", 
              recursive: bool = False) -> List[Path]:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        recursive: 是否递归搜索子目录
    
    Returns:
        文件路径列表
    """
    path = Path(directory)
    if recursive:
        return list(path.glob(f"**/{pattern}"))
    else:
        return list(path.glob(pattern))


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        JSON数据，如果加载失败返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_json(data: Dict[str, Any], 
             file_path: Union[str, Path], 
             indent: int = 4) -> bool:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    
    Returns:
        是否保存成功
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_yaml(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    加载YAML文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        YAML数据，如果加载失败返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def save_yaml(data: Dict[str, Any], 
             file_path: Union[str, Path]) -> bool:
    """
    保存数据到YAML文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    
    Returns:
        是否保存成功
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception:
        return False


def copy_file(src: Union[str, Path], 
             dst: Union[str, Path], 
             overwrite: bool = False) -> bool:
    """
    复制文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        overwrite: 是否覆盖已存在的文件
    
    Returns:
        是否复制成功
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            return False
        
        if dst_path.exists() and not overwrite:
            return False
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    except Exception:
        return False


def get_file_size(file_path: Union[str, Path], 
                 human_readable: bool = False) -> Union[int, str]:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        human_readable: 是否返回人类可读的大小字符串
    
    Returns:
        文件大小（字节）或人类可读的大小字符串
    """
    size = Path(file_path).stat().st_size
    
    if not human_readable:
        return size
    
    # 转换为人类可读格式
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024 or unit == 'TB':
            return f"{size:.2f} {unit}"
        size /= 1024
    
    # 默认返回（不应该到达这里）
    return f"{size:.2f} TB"


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件扩展名（不包含点）
    """
    return Path(file_path).suffix.lstrip('.')


def is_file_newer(file1: Union[str, Path], 
                 file2: Union[str, Path]) -> bool:
    """
    判断file1是否比file2更新
    
    Args:
        file1: 文件1路径
        file2: 文件2路径
    
    Returns:
        file1是否比file2更新
    """
    try:
        mtime1 = Path(file1).stat().st_mtime
        mtime2 = Path(file2).stat().st_mtime
        return mtime1 > mtime2
    except Exception:
        return False


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    # 假设当前文件在 src/utils 目录下
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_data_dir() -> Path:
    """
    获取数据目录
    
    Returns:
        数据目录路径
    """
    root = get_project_root()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_models_dir() -> Path:
    """
    获取模型目录
    
    Returns:
        模型目录路径
    """
    root = get_project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_logs_dir() -> Path:
    """
    获取日志目录
    
    Returns:
        日志目录路径
    """
    root = get_project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_config_dir() -> Path:
    """
    获取配置目录
    
    Returns:
        配置目录路径
    """
    root = get_project_root()
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir