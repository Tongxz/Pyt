# Core modules for human behavior detection
# 人体行为检测核心模块

# 延迟导入以避免循环依赖
__all__ = [
    'HumanDetector',
    'MultiObjectTracker', 
    'BehaviorRecognizer',
    'RegionManager'
]

def __getattr__(name):
    if name == 'HumanDetector':
        from .detector import HumanDetector
        return HumanDetector
    elif name == 'MultiObjectTracker':
        from .tracker import MultiObjectTracker
        return MultiObjectTracker
    elif name == 'BehaviorRecognizer':
        from .behavior import BehaviorRecognizer
        return BehaviorRecognizer
    elif name == 'RegionManager':
        from .region import RegionManager
        return RegionManager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")