import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class Track:
    """单个追踪目标"""
    
    def __init__(self, track_id: int, bbox: List[int], confidence: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = deque(maxlen=30)  # 保存历史位置
        self.history.append(bbox)
        self.state = 'active'  # active, lost, deleted
        
    def update(self, bbox: List[int], confidence: float):
        """更新追踪目标"""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        self.state = 'active'
        
    def predict(self):
        """预测下一帧位置（简单线性预测）"""
        if len(self.history) < 2:
            return self.bbox
            
        # 计算速度
        prev_bbox = self.history[-2]
        curr_bbox = self.history[-1]
        
        dx = curr_bbox[0] - prev_bbox[0]
        dy = curr_bbox[1] - prev_bbox[1]
        
        # 预测下一帧位置
        predicted_bbox = [
            curr_bbox[0] + dx,
            curr_bbox[1] + dy,
            curr_bbox[2] + dx,
            curr_bbox[3] + dy
        ]
        
        return predicted_bbox
        
    def get_center(self) -> Tuple[float, float]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
        
    def get_area(self) -> float:
        """获取边界框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class MultiObjectTracker:
    """多目标追踪器
    
    基于IoU匹配的简单追踪算法
    """
    
    def __init__(self, max_disappeared: int = 10, iou_threshold: float = 0.3):
        """
        初始化追踪器
        
        Args:
            max_disappeared: 目标消失的最大帧数
            iou_threshold: IoU匹配阈值
        """
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        
        logger.info(f"MultiObjectTracker initialized with IoU threshold: {iou_threshold}")
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新追踪器
        
        Args:
            detections: 当前帧的检测结果
            
        Returns:
            追踪结果列表
        """
        # 如果没有检测结果，更新所有追踪目标的状态
        if not detections:
            for track in self.tracks.values():
                track.time_since_update += 1
                track.age += 1
                if track.time_since_update > self.max_disappeared:
                    track.state = 'lost'
            return self._get_active_tracks()
        
        # 计算IoU矩阵
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted_bbox = track.predict()
            
            for j, detection in enumerate(detections):
                iou = self.calculate_iou(predicted_bbox, detection['bbox'])
                iou_matrix[i, j] = iou
        
        # 匹配追踪目标和检测结果
        matched_tracks, matched_detections = self._match_tracks_detections(
            iou_matrix, track_ids, detections
        )
        
        # 更新匹配的追踪目标
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            track_id = track_ids[track_idx]
            detection = detections[det_idx]
            self.tracks[track_id].update(detection['bbox'], detection['confidence'])
        
        # 创建新的追踪目标
        unmatched_detections = set(range(len(detections))) - set(matched_detections)
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            new_track = Track(self.next_id, detection['bbox'], detection['confidence'])
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # 更新未匹配的追踪目标
        unmatched_tracks = set(range(len(track_ids))) - set(matched_tracks)
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            track.time_since_update += 1
            track.age += 1
            if track.time_since_update > self.max_disappeared:
                track.state = 'lost'
        
        # 删除长时间丢失的追踪目标
        self._cleanup_tracks()
        
        return self._get_active_tracks()
    
    def _match_tracks_detections(self, iou_matrix: np.ndarray, 
                                track_ids: List[int], 
                                detections: List[Dict]) -> Tuple[List[int], List[int]]:
        """匹配追踪目标和检测结果"""
        matched_tracks = []
        matched_detections = []
        
        # 贪心匹配：选择IoU最大的配对
        while True:
            if iou_matrix.size == 0:
                break
                
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            if max_iou < self.iou_threshold:
                break
                
            track_idx, det_idx = max_iou_idx
            matched_tracks.append(track_idx)
            matched_detections.append(det_idx)
            
            # 移除已匹配的行和列
            iou_matrix = np.delete(iou_matrix, track_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, det_idx, axis=1)
            
            # 更新索引
            for i in range(len(matched_tracks) - 1):
                if matched_tracks[i] >= track_idx:
                    matched_tracks[i] += 1
            for i in range(len(matched_detections) - 1):
                if matched_detections[i] >= det_idx:
                    matched_detections[i] += 1
        
        return matched_tracks, matched_detections
    
    def _cleanup_tracks(self):
        """清理长时间丢失的追踪目标"""
        to_delete = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared * 2:
                to_delete.append(track_id)
        
        for track_id in to_delete:
            del self.tracks[track_id]
    
    def _get_active_tracks(self) -> List[Dict]:
        """获取活跃的追踪目标"""
        active_tracks = []
        for track in self.tracks.values():
            if track.state == 'active':
                track_info = {
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.hits
                }
                active_tracks.append(track_info)
        
        return active_tracks
    
    def get_track_history(self, track_id: int) -> List[List[int]]:
        """获取指定追踪目标的历史轨迹"""
        if track_id in self.tracks:
            return list(self.tracks[track_id].history)
        return []
    
    def reset(self):
        """重置追踪器"""
        self.tracks.clear()
        self.next_id = 1
        logger.info("MultiObjectTracker reset")