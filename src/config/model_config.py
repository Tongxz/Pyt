from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import os

@dataclass
class YOLOConfig:
    """YOLO模型配置"""
    model_path: str = 'yolov8n.pt'
    input_size: tuple = (640, 640)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: Optional[List[int]] = None  # None表示检测所有类别
    device: str = 'auto'
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = [0]  # 默认只检测人体 (COCO class 0)

@dataclass
class PoseEstimationConfig:
    """姿态估计模型配置"""
    model_type: str = 'mediapipe'  # 'mediapipe', 'openpose', 'hrnet'
    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    tracking_confidence: float = 0.5
    detection_confidence: float = 0.5
    max_num_hands: int = 2
    max_num_faces: int = 1
    max_num_poses: int = 1
    static_image_mode: bool = False
    model_complexity: int = 1  # 0, 1, 2 (for MediaPipe)

@dataclass
class HairnetDetectionConfig:
    """发网检测模型配置"""
    model_path: str = './models/hairnet_detector.pth'
    input_size: tuple = (224, 224)
    confidence_threshold: float = 0.7
    batch_size: int = 8
    device: str = 'auto'
    preprocessing: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = {
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }

@dataclass
class HandwashDetectionConfig:
    """洗手检测模型配置"""
    model_path: str = './models/handwash_detector.pth'
    sequence_length: int = 16  # 时序模型的序列长度
    input_size: tuple = (224, 224)
    confidence_threshold: float = 0.6
    min_duration: float = 15.0  # 最小洗手时间（秒）
    max_duration: float = 60.0  # 最大洗手时间（秒）
    device: str = 'auto'

@dataclass
class SanitizeDetectionConfig:
    """消毒检测模型配置"""
    model_path: str = './models/sanitize_detector.pth'
    input_size: tuple = (224, 224)
    confidence_threshold: float = 0.6
    min_duration: float = 3.0   # 最小消毒时间（秒）
    max_duration: float = 30.0  # 最大消毒时间（秒）
    hand_distance_threshold: int = 100  # 双手距离阈值（像素）
    device: str = 'auto'

@dataclass
class SelfLearningConfig:
    """自学习模型配置"""
    enabled: bool = True
    learning_rate: float = 0.001
    batch_size: int = 16
    update_frequency: int = 100  # 每N个样本更新一次模型
    confidence_threshold: float = 0.8  # 用于自动标注的置信度阈值
    max_samples_per_class: int = 1000
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    model_save_frequency: int = 500  # 每N次更新保存一次模型
    
class ModelConfig:
    """模型配置管理器"""
    
    def __init__(self):
        """初始化模型配置"""
        self.yolo = YOLOConfig()
        self.pose_estimation = PoseEstimationConfig()
        self.hairnet_detection = HairnetDetectionConfig()
        self.handwash_detection = HandwashDetectionConfig()
        self.sanitize_detection = SanitizeDetectionConfig()
        self.self_learning = SelfLearningConfig()
        
        # 模型路径映射
        self.model_paths = {
            'yolo': self.yolo.model_path,
            'hairnet': self.hairnet_detection.model_path,
            'handwash': self.handwash_detection.model_path,
            'sanitize': self.sanitize_detection.model_path
        }
    
    def get_model_config(self, model_name: str) -> Optional[Any]:
        """
        获取指定模型的配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置对象
        """
        config_map = {
            'yolo': self.yolo,
            'pose_estimation': self.pose_estimation,
            'hairnet_detection': self.hairnet_detection,
            'handwash_detection': self.handwash_detection,
            'sanitize_detection': self.sanitize_detection,
            'self_learning': self.self_learning
        }
        
        return config_map.get(model_name)
    
    def update_model_config(self, model_name: str, **kwargs) -> bool:
        """
        更新模型配置
        
        Args:
            model_name: 模型名称
            **kwargs: 配置参数
            
        Returns:
            True if successfully updated
        """
        config = self.get_model_config(model_name)
        if config is None:
            return False
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return True
    
    def validate_model_paths(self) -> Dict[str, bool]:
        """
        验证模型文件是否存在
        
        Returns:
            模型文件存在状态字典
        """
        status = {}
        
        for model_name, model_path in self.model_paths.items():
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # 对于PyTorch模型文件，检查文件是否存在
                status[model_name] = os.path.exists(model_path)
            else:
                # 对于其他类型（如MediaPipe），标记为可用
                status[model_name] = True
        
        return status
    
    def get_device_config(self) -> str:
        """
        获取设备配置

        Returns:
            设备类型 ('cpu', 'cuda', 'mps')
        """
        try:
            import torch
            
            if self.yolo.device == 'auto':
                if torch.cuda.is_available():
                    return 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                else:
                    return 'cpu'
            
            return self.yolo.device
        except ImportError:
            return 'cpu'
    
    def get_memory_requirements(self) -> Dict[str, str]:
        """
        获取模型内存需求估算
        
        Returns:
            内存需求字典
        """
        requirements = {
            'yolo': '200-500MB',
            'pose_estimation': '50-100MB',
            'hairnet_detection': '100-200MB',
            'handwash_detection': '150-300MB',
            'sanitize_detection': '100-200MB',
            'total_estimated': '600MB-1.3GB'
        }
        
        return requirements
    
    def optimize_for_device(self, device_type: str):
        """
        根据设备类型优化配置
        
        Args:
            device_type: 设备类型 ('cpu', 'cuda', 'edge')
        """
        if device_type == 'cpu':
            # CPU优化：减少批量大小，降低模型复杂度
            self.yolo.input_size = (416, 416)
            self.hairnet_detection.batch_size = 4
            self.handwash_detection.sequence_length = 8
            self.pose_estimation.model_complexity = 0
            
        elif device_type == 'cuda':
            # GPU优化：可以使用更大的批量和更高的分辨率
            self.yolo.input_size = (640, 640)
            self.hairnet_detection.batch_size = 16
            self.handwash_detection.sequence_length = 16
            self.pose_estimation.model_complexity = 2
            
        elif device_type == 'edge':
            # 边缘设备优化：最小化资源使用
            self.yolo.model_path = 'yolov8n.pt'  # 使用nano版本
            self.yolo.input_size = (320, 320)
            self.hairnet_detection.batch_size = 2
            self.handwash_detection.sequence_length = 4
            self.pose_estimation.model_complexity = 0
            self.pose_estimation.max_num_hands = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            配置字典
        """
        from dataclasses import asdict
        
        return {
            'yolo': asdict(self.yolo),
            'pose_estimation': asdict(self.pose_estimation),
            'hairnet_detection': asdict(self.hairnet_detection),
            'handwash_detection': asdict(self.handwash_detection),
            'sanitize_detection': asdict(self.sanitize_detection),
            'self_learning': asdict(self.self_learning)
        }