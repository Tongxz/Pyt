# Utility functions
# 工具函数模块

__all__ = [
    'logger',
    'image_utils',
    'video_utils',
    'math_utils',
    'file_utils'
]

def __getattr__(name):
    if name == 'logger':
        from .logger import get_logger
        return get_logger
    elif name == 'image_utils':
        from . import image_utils
        return image_utils
    elif name == 'video_utils':
        from . import video_utils
        return video_utils
    elif name == 'math_utils':
        from . import math_utils
        return math_utils
    elif name == 'file_utils':
        from . import file_utils
        return file_utils
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")