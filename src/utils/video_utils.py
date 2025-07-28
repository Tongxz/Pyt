# Video processing utilities
# 视频处理工具模块

import time
from typing import Optional, Tuple, Iterator, Union
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class VideoCapture:
    """
    视频捕获类，支持文件和摄像头
    """
    
    def __init__(self, source: Union[str, int]):
        """
        初始化视频捕获
        
        Args:
            source: 视频源（文件路径或摄像头索引）
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video processing")
        
        self.source = source
        self.cap = None
        self.fps = 30
        self.frame_count = 0
        self.current_frame = 0
        self.width = 0
        self.height = 0
        
    def open(self) -> bool:
        """
        打开视频源
        
        Returns:
            是否成功打开
        """
        if not CV2_AVAILABLE:
            return False
        
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        
        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return True
    
    def read(self) -> Tuple[bool, Optional[any]]:
        """
        读取一帧
        
        Returns:
            (是否成功, 帧图像)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        
        return ret, frame
    
    def seek(self, frame_number: int) -> bool:
        """
        跳转到指定帧
        
        Args:
            frame_number: 帧号
        
        Returns:
            是否成功
        """
        if self.cap is None:
            return False
        
        if CV2_AVAILABLE:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        return False
    
    def release(self):
        """
        释放资源
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __iter__(self) -> Iterator[any]:
        """
        迭代器接口
        """
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame


class VideoWriter:
    """
    视频写入类
    """
    
    def __init__(self, output_path: Union[str, Path], 
                 fps: float = 30.0,
                 frame_size: Tuple[int, int] = (640, 480),
                 fourcc: str = 'mp4v'):
        """
        初始化视频写入器
        
        Args:
            output_path: 输出文件路径
            fps: 帧率
            frame_size: 帧尺寸 (width, height)
            fourcc: 编码格式
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video writing")
        
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = fourcc
        self.writer = None
        
    def open(self) -> bool:
        """
        打开视频写入器
        
        Returns:
            是否成功打开
        """
        if not CV2_AVAILABLE:
            return False
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc_code,
            self.fps,
            self.frame_size
        )
        
        return self.writer.isOpened()
    
    def write(self, frame) -> bool:
        """
        写入一帧
        
        Args:
            frame: 帧图像
        
        Returns:
            是否成功写入
        """
        if self.writer is None:
            return False
        
        try:
            # 确保帧尺寸正确
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            
            self.writer.write(frame)
            return True
        except Exception:
            return False
    
    def release(self):
        """
        释放资源
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def get_video_info(video_path: Union[str, Path]) -> Optional[dict]:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        视频信息字典
    """
    if not CV2_AVAILABLE:
        return None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    except Exception:
        return None


def extract_frames(video_path: Union[str, Path],
                  output_dir: Union[str, Path],
                  frame_interval: int = 1,
                  start_frame: int = 0,
                  end_frame: Optional[int] = None) -> bool:
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_interval: 帧间隔
        start_frame: 开始帧
        end_frame: 结束帧（None表示到视频结束）
    
    Returns:
        是否成功
    """
    if not CV2_AVAILABLE:
        return False
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with VideoCapture(str(video_path)) as cap:
            if not cap.open():
                return False
            
            # 跳转到开始帧
            if start_frame > 0:
                cap.seek(start_frame)
            
            frame_count = 0
            saved_count = 0
            
            for frame in cap:
                current_frame_num = start_frame + frame_count
                
                # 检查是否到达结束帧
                if end_frame is not None and current_frame_num >= end_frame:
                    break
                
                # 按间隔保存帧
                if frame_count % frame_interval == 0:
                    frame_filename = output_path / f"frame_{current_frame_num:06d}.jpg"
                    cv2.imwrite(str(frame_filename), frame)
                    saved_count += 1
                
                frame_count += 1
        
        return True
    except Exception:
        return False


def create_video_from_frames(frames_dir: Union[str, Path],
                           output_path: Union[str, Path],
                           fps: float = 30.0,
                           frame_pattern: str = "*.jpg") -> bool:
    """
    从帧图像创建视频
    
    Args:
        frames_dir: 帧图像目录
        output_path: 输出视频路径
        fps: 帧率
        frame_pattern: 帧文件匹配模式
    
    Returns:
        是否成功
    """
    if not CV2_AVAILABLE:
        return False
    
    try:
        frames_path = Path(frames_dir)
        frame_files = sorted(frames_path.glob(frame_pattern))
        
        if not frame_files:
            return False
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            return False
        
        height, width = first_frame.shape[:2]
        
        with VideoWriter(output_path, fps, (width, height)) as writer:
            if not writer.open():
                return False
            
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    writer.write(frame)
        
        return True
    except Exception:
        return False


class FPSCounter:
    """
    FPS计数器
    """
    
    def __init__(self, window_size: int = 30):
        """
        初始化FPS计数器
        
        Args:
            window_size: 计算窗口大小
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        更新FPS计数
        
        Returns:
            当前FPS
        """
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        # 保持窗口大小
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # 计算平均FPS
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return 0.0
    
    def reset(self):
        """
        重置计数器
        """
        self.frame_times.clear()
        self.last_time = time.time()