import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """检测配置"""
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    input_size: tuple = (640, 640)
    device: str = 'auto'

@dataclass
class TrackingConfig:
    """追踪配置"""
    max_disappeared: int = 10
    iou_threshold: float = 0.3
    max_tracks: int = 50
    track_buffer: int = 30

@dataclass
class BehaviorConfig:
    """行为识别配置"""
    confidence_threshold: float = 0.6
    min_duration: float = 1.0
    stability_frames: int = 5
    enabled_behaviors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.enabled_behaviors is None:
            self.enabled_behaviors = ['hairnet', 'handwashing', 'sanitizing']

@dataclass
class SystemConfig:
    """系统配置"""
    log_level: str = 'INFO'
    max_fps: int = 30
    save_video: bool = False
    video_output_dir: str = './output/videos'
    log_output_dir: str = './output/logs'
    model_cache_dir: str = './models'
    
@dataclass
class APIConfig:
    """API配置"""
    host: str = '0.0.0.0'
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    max_request_size: int = 16 * 1024 * 1024  # 16MB

class Settings:
    """系统设置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化设置
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file or 'config.json'
        
        # 默认配置
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.behavior = BehaviorConfig()
        self.system = SystemConfig()
        self.api = APIConfig()
        
        # 加载配置文件
        if os.path.exists(self.config_file):
            self.load_from_file(self.config_file)
        else:
            logger.info(f"Config file {self.config_file} not found, using defaults")
    
    def load_from_file(self, file_path: str) -> bool:
        """
        从文件加载配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            True if successfully loaded
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            if 'detection' in config_data:
                self.detection = DetectionConfig(**config_data['detection'])
            
            if 'tracking' in config_data:
                self.tracking = TrackingConfig(**config_data['tracking'])
            
            if 'behavior' in config_data:
                self.behavior = BehaviorConfig(**config_data['behavior'])
            
            if 'system' in config_data:
                self.system = SystemConfig(**config_data['system'])
            
            if 'api' in config_data:
                self.api = APIConfig(**config_data['api'])
            
            logger.info(f"Configuration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return False
    
    def save_to_file(self, file_path: Optional[str] = None) -> bool:
        """
        保存配置到文件
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            True if successfully saved
        """
        file_path = file_path or self.config_file
        
        try:
            config_data = {
                'detection': asdict(self.detection),
                'tracking': asdict(self.tracking),
                'behavior': asdict(self.behavior),
                'system': asdict(self.system),
                'api': asdict(self.api)
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False
    
    def update_detection_config(self, **kwargs):
        """更新检测配置"""
        for key, value in kwargs.items():
            if hasattr(self.detection, key):
                setattr(self.detection, key, value)
                logger.info(f"Detection config updated: {key} = {value}")
    
    def update_tracking_config(self, **kwargs):
        """更新追踪配置"""
        for key, value in kwargs.items():
            if hasattr(self.tracking, key):
                setattr(self.tracking, key, value)
                logger.info(f"Tracking config updated: {key} = {value}")
    
    def update_behavior_config(self, **kwargs):
        """更新行为配置"""
        for key, value in kwargs.items():
            if hasattr(self.behavior, key):
                setattr(self.behavior, key, value)
                logger.info(f"Behavior config updated: {key} = {value}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            'detection': asdict(self.detection),
            'tracking': asdict(self.tracking),
            'behavior': asdict(self.behavior),
            'system': asdict(self.system),
            'api': asdict(self.api)
        }
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.behavior = BehaviorConfig()
        self.system = SystemConfig()
        self.api = APIConfig()
        
        logger.info("Configuration reset to defaults")

# 全局设置实例
_settings = None

def get_settings() -> Settings:
    """获取全局设置实例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def load_config(config_file: str) -> Settings:
    """加载配置文件"""
    global _settings
    _settings = Settings(config_file)
    return _settings

def save_config(config_file: Optional[str] = None) -> bool:
    """保存当前配置"""
    settings = get_settings()
    return settings.save_to_file(config_file)