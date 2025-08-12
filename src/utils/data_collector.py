import logging
import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import queue
import time

logger = logging.getLogger(__name__)


class VideoSegment:
    """视频片段类"""
    
    def __init__(self, segment_id: str, behavior_type: str, confidence: float):
        self.segment_id = segment_id
        self.behavior_type = behavior_type
        self.confidence = confidence
        self.start_time = datetime.now()
        self.frames = []
        self.metadata = {}
        self.end_time = None
    
    def add_frame(self, frame: np.ndarray, frame_metadata: Optional[Dict] = None):
        """添加帧到视频片段"""
        self.frames.append({
            'frame': frame.copy(),
            'timestamp': datetime.now(),
            'metadata': frame_metadata or {}
        })
    
    def finalize(self):
        """完成视频片段"""
        self.end_time = datetime.now()
        self.metadata['duration'] = (self.end_time - self.start_time).total_seconds()
        self.metadata['frame_count'] = len(self.frames)
    
    def get_duration(self) -> float:
        """获取片段持续时间（秒）"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


class DataCollector:
    """数据收集器
    
    自动收集和保存符合条件的视频片段，用于后续模型训练
    """
    
    def __init__(self, 
                 data_root: str = "data",
                 min_segment_duration: float = 2.0,
                 max_segment_duration: float = 30.0,
                 confidence_threshold: float = 0.5,
                 max_queue_size: int = 100):
        """
        初始化数据收集器
        
        Args:
            data_root: 数据根目录
            min_segment_duration: 最小片段持续时间（秒）
            max_segment_duration: 最大片段持续时间（秒）
            confidence_threshold: 置信度阈值
            max_queue_size: 最大队列大小
        """
        self.data_root = Path(data_root)
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.confidence_threshold = confidence_threshold
        
        # 创建目录结构
        self._setup_directories()
        
        # 当前活跃的视频片段
        self.active_segments = {}  # track_id -> {behavior_type -> VideoSegment}
        
        # 保存队列和线程
        self.save_queue = queue.Queue(maxsize=max_queue_size)
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # 统计信息
        self.stats = {
            'total_segments': 0,
            'handwash_segments': 0,
            'sanitize_segments': 0,
            'negative_segments': 0,
            'total_frames': 0
        }
        
        logger.info(f"DataCollector initialized with data_root: {self.data_root}")
    
    def _setup_directories(self):
        """设置目录结构"""
        directories = [
            self.data_root / "videos" / "handwash",
            self.data_root / "videos" / "sanitize", 
            self.data_root / "videos" / "negative",
            self.data_root / "annotations",
            self.data_root / "keypoints",
            self.data_root / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data directories created")
    
    def update_detection(self, 
                        track_id: int,
                        frame: np.ndarray,
                        behavior_results: Dict[str, float],
                        frame_metadata: Optional[Dict] = None):
        """
        更新检测结果并收集数据
        
        Args:
            track_id: 追踪目标ID
            frame: 当前帧
            behavior_results: 行为检测结果 {behavior_type: confidence}
            frame_metadata: 帧元数据
        """
        current_time = datetime.now()
        
        if track_id not in self.active_segments:
            self.active_segments[track_id] = {}
        
        # 处理每种行为类型
        for behavior_type, confidence in behavior_results.items():
            if behavior_type in ['handwashing', 'sanitizing']:
                self._handle_positive_behavior(track_id, frame, behavior_type, 
                                             confidence, frame_metadata)
            else:
                self._handle_other_behavior(track_id, frame, behavior_type, 
                                          confidence, frame_metadata)
        
        # 检查并完成超时的片段
        self._check_timeout_segments(track_id)
    
    def _handle_positive_behavior(self, 
                                 track_id: int,
                                 frame: np.ndarray,
                                 behavior_type: str,
                                 confidence: float,
                                 frame_metadata: Optional[Dict]):
        """处理正向行为（洗手、消毒）"""
        if confidence >= self.confidence_threshold:
            # 开始或继续记录片段
            if behavior_type not in self.active_segments[track_id]:
                # 创建新片段
                segment_id = self._generate_segment_id(track_id, behavior_type)
                segment = VideoSegment(segment_id, behavior_type, confidence)
                self.active_segments[track_id][behavior_type] = segment
                logger.debug(f"Started recording {behavior_type} segment for track {track_id}")
            
            # 添加帧到现有片段
            segment = self.active_segments[track_id][behavior_type]
            segment.add_frame(frame, frame_metadata)
            segment.confidence = max(segment.confidence, confidence)  # 更新最高置信度
            
            # 检查是否达到最大持续时间
            if segment.get_duration() >= self.max_segment_duration:
                self._finalize_segment(track_id, behavior_type)
        
        else:
            # 置信度不足，结束当前片段（如果存在）
            if behavior_type in self.active_segments[track_id]:
                self._finalize_segment(track_id, behavior_type)
    
    def _handle_other_behavior(self, 
                              track_id: int,
                              frame: np.ndarray,
                              behavior_type: str,
                              confidence: float,
                              frame_metadata: Optional[Dict]):
        """处理其他行为（负样本）"""
        # 随机收集负样本
        import random
        if random.random() < 0.1:  # 10%的概率收集负样本
            if 'negative' not in self.active_segments[track_id]:
                segment_id = self._generate_segment_id(track_id, 'negative')
                segment = VideoSegment(segment_id, 'negative', confidence)
                self.active_segments[track_id]['negative'] = segment
            
            segment = self.active_segments[track_id]['negative']
            segment.add_frame(frame, frame_metadata)
            
            # 负样本片段较短
            if segment.get_duration() >= 5.0:  # 5秒
                self._finalize_segment(track_id, 'negative')
    
    def _finalize_segment(self, track_id: int, behavior_type: str):
        """完成并保存视频片段"""
        if (track_id not in self.active_segments or 
            behavior_type not in self.active_segments[track_id]):
            return
        
        segment = self.active_segments[track_id][behavior_type]
        segment.finalize()
        
        # 检查最小持续时间
        if segment.get_duration() >= self.min_segment_duration:
            # 添加到保存队列
            try:
                self.save_queue.put_nowait(segment)
                logger.info(f"Queued {behavior_type} segment {segment.segment_id} "
                           f"for saving (duration: {segment.get_duration():.2f}s, "
                           f"frames: {len(segment.frames)})")
            except queue.Full:
                logger.warning("Save queue is full, dropping segment")
        else:
            logger.debug(f"Segment {segment.segment_id} too short, discarding")
        
        # 从活跃片段中移除
        del self.active_segments[track_id][behavior_type]
    
    def _check_timeout_segments(self, track_id: int):
        """检查并处理超时的片段"""
        if track_id not in self.active_segments:
            return
        
        to_finalize = []
        for behavior_type, segment in self.active_segments[track_id].items():
            if segment.get_duration() >= self.max_segment_duration:
                to_finalize.append(behavior_type)
        
        for behavior_type in to_finalize:
            self._finalize_segment(track_id, behavior_type)
    
    def _generate_segment_id(self, track_id: int, behavior_type: str) -> str:
        """生成片段ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{behavior_type}_track{track_id}_{timestamp}"
    
    def _save_worker(self):
        """保存工作线程"""
        while True:
            try:
                segment = self.save_queue.get(timeout=1.0)
                self._save_segment(segment)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in save worker: {e}")
    
    def _save_segment(self, segment: VideoSegment):
        """保存视频片段"""
        try:
            # 确定保存路径
            if segment.behavior_type in ['handwashing', 'handwash']:
                video_dir = self.data_root / "videos" / "handwash"
                self.stats['handwash_segments'] += 1
            elif segment.behavior_type in ['sanitizing', 'sanitize']:
                video_dir = self.data_root / "videos" / "sanitize"
                self.stats['sanitize_segments'] += 1
            else:
                video_dir = self.data_root / "videos" / "negative"
                self.stats['negative_segments'] += 1
            
            # 保存视频文件
            video_path = video_dir / f"{segment.segment_id}.mp4"
            self._save_video_file(segment, video_path)
            
            # 保存标注信息
            annotation_path = self.data_root / "annotations" / f"{segment.segment_id}.json"
            self._save_annotation(segment, annotation_path)
            
            # 保存关键点数据（如果有）
            keypoints_path = self.data_root / "keypoints" / f"{segment.segment_id}.json"
            self._save_keypoints(segment, keypoints_path)
            
            # 更新统计
            self.stats['total_segments'] += 1
            self.stats['total_frames'] += len(segment.frames)
            
            logger.info(f"Saved segment {segment.segment_id} to {video_path}")
            
        except Exception as e:
            logger.error(f"Failed to save segment {segment.segment_id}: {e}")
    
    def _save_video_file(self, segment: VideoSegment, video_path: Path):
        """保存视频文件"""
        if not segment.frames:
            return
        
        # 获取第一帧的尺寸
        first_frame = segment.frames[0]['frame']
        height, width = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0  # 假设30fps
        
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        try:
            for frame_data in segment.frames:
                frame = frame_data['frame']
                out.write(frame)
        finally:
            out.release()
    
    def _save_annotation(self, segment: VideoSegment, annotation_path: Path):
        """保存标注信息"""
        annotation = {
            'segment_id': segment.segment_id,
            'behavior_type': segment.behavior_type,
            'confidence': segment.confidence,
            'start_time': segment.start_time.isoformat(),
            'end_time': segment.end_time.isoformat() if segment.end_time else None,
            'duration': segment.get_duration(),
            'frame_count': len(segment.frames),
            'metadata': segment.metadata
        }
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
    
    def _save_keypoints(self, segment: VideoSegment, keypoints_path: Path):
        """保存关键点数据"""
        keypoints_data = {
            'segment_id': segment.segment_id,
            'frames': []
        }
        
        for i, frame_data in enumerate(segment.frames):
            frame_keypoints = {
                'frame_index': i,
                'timestamp': frame_data['timestamp'].isoformat(),
                'keypoints': frame_data['metadata'].get('keypoints', {}),
                'pose_data': frame_data['metadata'].get('pose_data', {}),
                'hand_data': frame_data['metadata'].get('hand_data', [])
            }
            keypoints_data['frames'].append(frame_keypoints)
        
        with open(keypoints_path, 'w', encoding='utf-8') as f:
            json.dump(keypoints_data, f, indent=2, ensure_ascii=False)
    
    def force_finalize_track(self, track_id: int):
        """强制完成指定追踪目标的所有片段"""
        if track_id not in self.active_segments:
            return
        
        behavior_types = list(self.active_segments[track_id].keys())
        for behavior_type in behavior_types:
            self._finalize_segment(track_id, behavior_type)
        
        logger.info(f"Force finalized all segments for track {track_id}")
    
    def get_stats(self) -> Dict:
        """获取收集统计信息"""
        stats = self.stats.copy()
        stats['active_segments'] = sum(len(segments) for segments in self.active_segments.values())
        stats['queue_size'] = self.save_queue.qsize()
        return stats
    
    def save_stats(self):
        """保存统计信息到文件"""
        stats_path = self.data_root / "metadata" / "collection_stats.json"
        stats = self.get_stats()
        stats['last_updated'] = datetime.now().isoformat()
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def create_video_clip(self, behavior_type: str, track_id: int, confidence: float) -> str:
        """
        创建新的视频片段
        
        Args:
            behavior_type: 行为类型
            track_id: 追踪ID
            confidence: 置信度
            
        Returns:
            片段ID
        """
        if track_id not in self.active_segments:
            self.active_segments[track_id] = {}
        
        # 生成片段ID
        segment_id = self._generate_segment_id(track_id, behavior_type)
        
        # 创建新片段
        segment = VideoSegment(segment_id, behavior_type, confidence)
        self.active_segments[track_id][behavior_type] = segment
        
        logger.debug(f"Created {behavior_type} segment {segment_id} for track {track_id}")
        return segment_id
    
    def add_frame_to_clip(self, clip_id: str, frame: any) -> bool:
        """
        向视频片段添加帧（别名方法）
        
        Args:
            clip_id: 片段ID
            frame: 视频帧
            
        Returns:
            是否成功添加
        """
        return self.add_frame(clip_id, frame)
    
    def finalize_clip(self, clip_id: str) -> bool:
        """
        完成视频片段（别名方法）
        
        Args:
            clip_id: 片段ID
            
        Returns:
            是否成功完成
        """
        return self.end_clip(clip_id)
     
    def cleanup(self):
        """清理资源"""
        # 完成所有活跃片段
        for track_id in list(self.active_segments.keys()):
            self.force_finalize_track(track_id)
        
        # 等待保存队列清空
        self.save_queue.join()
        
        # 保存最终统计
        self.save_stats()
        
        logger.info("DataCollector cleaned up")