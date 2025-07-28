# Configuration management
# 配置管理模块

__all__ = [
    'Settings',
    'ModelConfig', 
    'CameraConfig',
    'load_config',
    'save_config'
]

def __getattr__(name):
    if name == 'Settings':
        from .settings import Settings
        return Settings
    elif name == 'ModelConfig':
        from .model_config import ModelConfig
        return ModelConfig
    elif name == 'CameraConfig':
        from .camera_config import CameraConfig
        return CameraConfig
    elif name == 'load_config':
        from .settings import load_config
        return load_config
    elif name == 'save_config':
        from .settings import save_config
        return save_config
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")