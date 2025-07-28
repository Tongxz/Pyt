from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class CameraSource:
    """摄像头源配置"""
    source_id: str
    source_type: str  # 'webcam', 'rtsp', 'file', 'ip_camera'
    source_path: str  # 摄像头ID、RTSP URL、文件路径等
    name: str = ""
    is_active: bool = True
    
    # 摄像头参数
    width: int = 1920
    height: int = 1080
    fps: int = 30
    
    # 连接参数
    timeout: int = 10  # 连接超时（秒）
    retry_count: int = 3
    retry_interval: int = 5  # 重试间隔（秒）
    
    # 缓冲区设置
    buffer_size: int = 1
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.source_type}_{self.source_id}"

@dataclass
class VideoProcessingConfig:
    """视频处理配置"""
    # 输入处理
    input_resize: Optional[Tuple[int, int]] = None  # (width, height)
    maintain_aspect_ratio: bool = True
    
    # 帧处理
    frame_skip: int = 0  # 跳帧数量
    max_fps: int = 30
    
    # 图像增强
    brightness: float = 0.0  # -1.0 to 1.0
    contrast: float = 1.0    # 0.0 to 2.0
    saturation: float = 1.0  # 0.0 to 2.0
    
    # 降噪
    denoise: bool = False
    denoise_strength: float = 0.5
    
    # 稳定化
    stabilization: bool = False

@dataclass
class RecordingConfig:
    """录制配置"""
    enabled: bool = False
    output_dir: str = './output/recordings'
    
    # 录制格式
    codec: str = 'mp4v'  # 'mp4v', 'XVID', 'H264'
    file_format: str = 'mp4'
    
    # 录制质量
    quality: int = 80  # 1-100
    bitrate: Optional[int] = None
    
    # 录制策略
    max_file_size_mb: int = 1000  # 单个文件最大大小（MB）
    max_duration_minutes: int = 60  # 单个文件最大时长（分钟）
    
    # 存储管理
    auto_cleanup: bool = True
    max_storage_gb: int = 10  # 最大存储空间（GB）
    retention_days: int = 7   # 保留天数

@dataclass
class StreamingConfig:
    """流媒体配置"""
    enabled: bool = False
    
    # RTMP推流
    rtmp_url: Optional[str] = None
    rtmp_key: Optional[str] = None
    
    # WebRTC
    webrtc_enabled: bool = True
    webrtc_port: int = 8080
    
    # 流质量
    stream_width: int = 1280
    stream_height: int = 720
    stream_fps: int = 25
    stream_bitrate: int = 2000  # kbps

class CameraConfig:
    """摄像头配置管理器"""
    
    def __init__(self):
        """初始化摄像头配置"""
        self.cameras: Dict[str, CameraSource] = {}
        self.video_processing = VideoProcessingConfig()
        self.recording = RecordingConfig()
        self.streaming = StreamingConfig()
        
        # 默认添加一个网络摄像头
        self.add_camera(CameraSource(
            source_id='default',
            source_type='webcam',
            source_path='0',
            name='默认摄像头'
        ))
        
        logger.info("CameraConfig initialized")
    
    def add_camera(self, camera: CameraSource) -> bool:
        """
        添加摄像头
        
        Args:
            camera: 摄像头配置
            
        Returns:
            True if successfully added
        """
        if camera.source_id in self.cameras:
            logger.warning(f"Camera {camera.source_id} already exists")
            return False
        
        self.cameras[camera.source_id] = camera
        logger.info(f"Camera {camera.name} added successfully")
        return True
    
    def remove_camera(self, source_id: str) -> bool:
        """
        移除摄像头
        
        Args:
            source_id: 摄像头ID
            
        Returns:
            True if successfully removed
        """
        if source_id not in self.cameras:
            logger.warning(f"Camera {source_id} not found")
            return False
        
        del self.cameras[source_id]
        logger.info(f"Camera {source_id} removed successfully")
        return True
    
    def get_camera(self, source_id: str) -> Optional[CameraSource]:
        """
        获取摄像头配置
        
        Args:
            source_id: 摄像头ID
            
        Returns:
            摄像头配置对象
        """
        return self.cameras.get(source_id)
    
    def get_active_cameras(self) -> List[CameraSource]:
        """
        获取所有活跃的摄像头
        
        Returns:
            活跃摄像头列表
        """
        return [camera for camera in self.cameras.values() if camera.is_active]
    
    def update_camera_config(self, source_id: str, **kwargs) -> bool:
        """
        更新摄像头配置
        
        Args:
            source_id: 摄像头ID
            **kwargs: 配置参数
            
        Returns:
            True if successfully updated
        """
        camera = self.get_camera(source_id)
        if camera is None:
            return False
        
        for key, value in kwargs.items():
            if hasattr(camera, key):
                setattr(camera, key, value)
                logger.info(f"Camera {source_id} config updated: {key} = {value}")
        
        return True
    
    def validate_camera_sources(self) -> Dict[str, bool]:
        """
        验证摄像头源是否可用
        
        Returns:
            摄像头可用状态字典
        """
        status = {}
        
        for source_id, camera in self.cameras.items():
            try:
                if camera.source_type == 'webcam':
                    # 验证网络摄像头
                    try:
                        import cv2
                        cap = cv2.VideoCapture(int(camera.source_path))
                        is_available = cap.isOpened()
                        cap.release()
                        status[source_id] = is_available
                    except (ImportError, ValueError):
                        status[source_id] = False
                    
                elif camera.source_type == 'rtsp':
                    # 验证RTSP流（简单检查URL格式）
                    status[source_id] = camera.source_path.startswith(('rtsp://', 'rtmp://'))
                    
                elif camera.source_type == 'file':
                    # 验证文件是否存在
                    import os
                    status[source_id] = os.path.exists(camera.source_path)
                    
                else:
                    status[source_id] = True  # 其他类型默认可用
                    
            except Exception as e:
                logger.error(f"Failed to validate camera {source_id}: {e}")
                status[source_id] = False
        
        return status
    
    def get_optimal_settings(self, device_type: str = 'auto') -> Dict[str, Any]:
        """
        获取设备优化设置
        
        Args:
            device_type: 设备类型 ('cpu', 'gpu', 'edge', 'auto')
            
        Returns:
            优化设置字典
        """
        if device_type == 'edge':
            # 边缘设备优化
            return {
                'input_resize': (640, 480),
                'max_fps': 15,
                'frame_skip': 1,
                'recording_quality': 60,
                'stream_bitrate': 1000
            }
        elif device_type == 'cpu':
            # CPU优化
            return {
                'input_resize': (1280, 720),
                'max_fps': 20,
                'frame_skip': 0,
                'recording_quality': 70,
                'stream_bitrate': 1500
            }
        else:
            # GPU或默认设置
            return {
                'input_resize': None,  # 保持原分辨率
                'max_fps': 30,
                'frame_skip': 0,
                'recording_quality': 80,
                'stream_bitrate': 2000
            }
    
    def apply_optimal_settings(self, device_type: str = 'auto'):
        """
        应用设备优化设置
        
        Args:
            device_type: 设备类型
        """
        settings = self.get_optimal_settings(device_type)
        
        # 更新视频处理配置
        if 'input_resize' in settings:
            self.video_processing.input_resize = settings['input_resize']
        if 'max_fps' in settings:
            self.video_processing.max_fps = settings['max_fps']
        if 'frame_skip' in settings:
            self.video_processing.frame_skip = settings['frame_skip']
        
        # 更新录制配置
        if 'recording_quality' in settings:
            self.recording.quality = settings['recording_quality']
        
        # 更新流媒体配置
        if 'stream_bitrate' in settings:
            self.streaming.stream_bitrate = settings['stream_bitrate']
        
        logger.info(f"Applied optimal settings for {device_type} device")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            配置字典
        """
        from dataclasses import asdict
        
        return {
            'cameras': {k: asdict(v) for k, v in self.cameras.items()},
            'video_processing': asdict(self.video_processing),
            'recording': asdict(self.recording),
            'streaming': asdict(self.streaming)
        }
    
    def get_camera_info_summary(self) -> List[Dict[str, Any]]:
        """
        获取摄像头信息摘要
        
        Returns:
            摄像头信息列表
        """
        summary = []
        
        for camera in self.cameras.values():
            info = {
                'source_id': camera.source_id,
                'name': camera.name,
                'type': camera.source_type,
                'resolution': f"{camera.width}x{camera.height}",
                'fps': camera.fps,
                'is_active': camera.is_active,
                'source_path': camera.source_path
            }
            summary.append(info)
        
        return summary