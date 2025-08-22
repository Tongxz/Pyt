"""手部检测质量评估模块

该模块提供对MediaPipe手部检测结果的质量评估功能，
包括关键点完整性、稳定性、形状合理性和运动连续性评估。

Author: Trae AI Assistant
Date: 2024
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import cv2
import mediapipe as mp
from dataclasses import dataclass
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """质量评估指标数据类"""
    completeness: float  # 关键点完整性 [0.0, 1.0]
    stability: float     # 关键点稳定性 [0.0, 1.0]
    shape_validity: float  # 手部形状合理性 [0.0, 1.0]
    motion_continuity: float  # 运动连续性 [0.0, 1.0]
    overall_quality: float   # 总体质量分数 [0.0, 1.0]
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'completeness': self.completeness,
            'stability': self.stability,
            'shape_validity': self.shape_validity,
            'motion_continuity': self.motion_continuity,
            'overall_quality': self.overall_quality
        }


class HandDetectionQualityAssessor:
    """手部检测质量评估器
    
    评估MediaPipe手部检测结果的质量，提供多维度的质量指标。
    """
    
    def __init__(self, 
                 history_size: int = 10,
                 min_confidence: float = 0.5,
                 stability_threshold: float = 0.02):
        """
        初始化质量评估器
        
        Args:
            history_size: 历史帧数量，用于运动连续性评估
            min_confidence: 最小置信度阈值
            stability_threshold: 稳定性阈值
        """
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.stability_threshold = stability_threshold
        
        # 历史数据存储
        self.landmark_history: deque = deque(maxlen=history_size)
        self.quality_history: deque = deque(maxlen=history_size)
        
        # 手部关键点连接关系（MediaPipe标准）
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]
        
        # 质量权重配置
        self.quality_weights = {
            'completeness': 0.3,
            'stability': 0.25,
            'shape_validity': 0.25,
            'motion_continuity': 0.2
        }
        
        logger.info("HandDetectionQualityAssessor initialized")
    
    def assess_quality(self, 
                      results: Any, 
                      image: np.ndarray) -> List[QualityMetrics]:
        """
        评估手部检测质量
        
        Args:
            results: MediaPipe检测结果
            image: 输入图像
            
        Returns:
            List[QualityMetrics]: 每只手的质量评估结果
        """
        if not results or not results.multi_hand_landmarks:
            return []
        
        quality_metrics = []
        
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            try:
                # 提取关键点坐标
                landmark_points = self._extract_landmark_points(landmarks, image.shape)
                
                # 计算各项质量指标
                completeness = self._calculate_completeness(landmarks)
                stability = self._calculate_stability(landmark_points)
                shape_validity = self._calculate_shape_validity(landmark_points)
                motion_continuity = self._calculate_motion_continuity(landmark_points)
                
                # 计算总体质量分数
                overall_quality = (
                    completeness * self.quality_weights['completeness'] +
                    stability * self.quality_weights['stability'] +
                    shape_validity * self.quality_weights['shape_validity'] +
                    motion_continuity * self.quality_weights['motion_continuity']
                )
                
                metrics = QualityMetrics(
                    completeness=completeness,
                    stability=stability,
                    shape_validity=shape_validity,
                    motion_continuity=motion_continuity,
                    overall_quality=overall_quality
                )
                
                quality_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error assessing hand {i} quality: {e}")
                # 返回低质量分数
                quality_metrics.append(QualityMetrics(
                    completeness=0.0,
                    stability=0.0,
                    shape_validity=0.0,
                    motion_continuity=0.0,
                    overall_quality=0.0
                ))
        
        # 更新历史记录
        if quality_metrics:
            self.quality_history.append(quality_metrics)
        
        return quality_metrics
    
    def _extract_landmark_points(self, 
                               landmarks: Any, 
                               image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        提取关键点坐标
        
        Args:
            landmarks: MediaPipe关键点数据
            image_shape: 图像形状 (H, W, C)
            
        Returns:
            np.ndarray: 关键点坐标数组 (21, 3) - (x, y, z)
        """
        h, w = image_shape[:2]
        points = np.zeros((21, 3))
        
        for i, landmark in enumerate(landmarks.landmark):
            points[i] = [
                landmark.x * w,
                landmark.y * h,
                landmark.z  # 相对深度
            ]
        
        return points
    
    def _calculate_completeness(self, landmarks: Any) -> float:
        """
        计算关键点完整性
        
        Args:
            landmarks: MediaPipe关键点数据
            
        Returns:
            float: 完整性分数 [0.0, 1.0]
        """
        if not landmarks or not landmarks.landmark:
            return 0.0
        
        # 检查关键点数量
        num_landmarks = len(landmarks.landmark)
        expected_landmarks = 21
        
        if num_landmarks < expected_landmarks:
            return num_landmarks / expected_landmarks
        
        # 检查关键点可见性（基于坐标范围）
        visible_count = 0
        for landmark in landmarks.landmark:
            if (0.0 <= landmark.x <= 1.0 and 
                0.0 <= landmark.y <= 1.0):
                visible_count += 1
        
        return visible_count / expected_landmarks
    
    def _calculate_stability(self, landmark_points: np.ndarray) -> float:
        """
        计算关键点稳定性
        
        Args:
            landmark_points: 当前帧关键点坐标
            
        Returns:
            float: 稳定性分数 [0.0, 1.0]
        """
        if len(self.landmark_history) < 2:
            # 历史数据不足，返回中等分数
            self.landmark_history.append(landmark_points)
            return 0.7
        
        # 计算与历史帧的差异
        prev_points = self.landmark_history[-1]
        
        # 计算欧氏距离
        distances = np.linalg.norm(landmark_points[:, :2] - prev_points[:, :2], axis=1)
        avg_distance = np.mean(distances)
        
        # 更新历史记录
        self.landmark_history.append(landmark_points)
        
        # 稳定性评分（距离越小，稳定性越高）
        stability = max(0.0, 1.0 - (avg_distance / self.stability_threshold))
        
        return min(1.0, stability)
    
    def _calculate_shape_validity(self, landmark_points: np.ndarray) -> float:
        """
        计算手部形状合理性
        
        Args:
            landmark_points: 关键点坐标
            
        Returns:
            float: 形状合理性分数 [0.0, 1.0]
        """
        try:
            validity_scores = []
            
            # 1. 检查手指长度比例
            finger_validity = self._check_finger_proportions(landmark_points)
            validity_scores.append(finger_validity)
            
            # 2. 检查关键点连接合理性
            connection_validity = self._check_connection_validity(landmark_points)
            validity_scores.append(connection_validity)
            
            # 3. 检查手掌区域合理性
            palm_validity = self._check_palm_validity(landmark_points)
            validity_scores.append(palm_validity)
            
            return np.mean(validity_scores)
            
        except Exception as e:
            logger.warning(f"Error calculating shape validity: {e}")
            return 0.5  # 返回中等分数
    
    def _check_finger_proportions(self, landmark_points: np.ndarray) -> float:
        """
        检查手指长度比例的合理性
        
        Args:
            landmark_points: 关键点坐标
            
        Returns:
            float: 手指比例合理性分数
        """
        try:
            # 定义手指关键点索引
            fingers = {
                'thumb': [1, 2, 3, 4],
                'index': [5, 6, 7, 8],
                'middle': [9, 10, 11, 12],
                'ring': [13, 14, 15, 16],
                'pinky': [17, 18, 19, 20]
            }
            
            finger_lengths = []
            
            for finger_name, indices in fingers.items():
                # 计算手指总长度
                total_length = 0
                for i in range(len(indices) - 1):
                    p1 = landmark_points[indices[i]][:2]
                    p2 = landmark_points[indices[i + 1]][:2]
                    total_length += np.linalg.norm(p2 - p1)
                
                finger_lengths.append(total_length)
            
            # 检查长度比例的合理性
            # 中指通常最长，小指最短
            if len(finger_lengths) >= 5:
                middle_length = finger_lengths[2]  # 中指
                pinky_length = finger_lengths[4]   # 小指
                
                if middle_length > 0 and pinky_length > 0:
                    ratio = pinky_length / middle_length
                    # 合理的比例范围：0.6-0.9
                    if 0.6 <= ratio <= 0.9:
                        return 1.0
                    else:
                        return max(0.0, 1.0 - abs(ratio - 0.75) / 0.25)
            
            return 0.7  # 默认分数
            
        except Exception:
            return 0.5
    
    def _check_connection_validity(self, landmark_points: np.ndarray) -> float:
        """
        检查关键点连接的合理性
        
        Args:
            landmark_points: 关键点坐标
            
        Returns:
            float: 连接合理性分数
        """
        try:
            valid_connections = 0
            total_connections = len(self.hand_connections)
            
            for start_idx, end_idx in self.hand_connections:
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    # 计算连接长度
                    p1 = landmark_points[start_idx][:2]
                    p2 = landmark_points[end_idx][:2]
                    distance = np.linalg.norm(p2 - p1)
                    
                    # 检查距离是否在合理范围内
                    if 5 <= distance <= 100:  # 像素距离范围
                        valid_connections += 1
            
            return valid_connections / total_connections if total_connections > 0 else 0.0
            
        except Exception:
            return 0.5
    
    def _check_palm_validity(self, landmark_points: np.ndarray) -> float:
        """
        检查手掌区域的合理性
        
        Args:
            landmark_points: 关键点坐标
            
        Returns:
            float: 手掌合理性分数
        """
        try:
            # 手掌关键点：0(手腕), 5, 9, 13, 17(各手指根部)
            palm_indices = [0, 5, 9, 13, 17]
            
            if all(i < len(landmark_points) for i in palm_indices):
                palm_points = landmark_points[palm_indices][:, :2]
                
                # 计算手掌区域面积
                area = self._calculate_polygon_area(palm_points)
                
                # 检查面积是否在合理范围内
                if 100 <= area <= 10000:  # 像素面积范围
                    return 1.0
                else:
                    # 基于面积偏差计算分数
                    optimal_area = 2000
                    deviation = abs(area - optimal_area) / optimal_area
                    return max(0.0, 1.0 - deviation)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """
        计算多边形面积（使用鞋带公式）
        
        Args:
            points: 多边形顶点坐标
            
        Returns:
            float: 多边形面积
        """
        if len(points) < 3:
            return 0.0
        
        x = points[:, 0]
        y = points[:, 1]
        
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] 
                            for i in range(-1, len(x) - 1)))
    
    def _calculate_motion_continuity(self, landmark_points: np.ndarray) -> float:
        """
        计算运动连续性
        
        Args:
            landmark_points: 当前帧关键点坐标
            
        Returns:
            float: 运动连续性分数 [0.0, 1.0]
        """
        if len(self.landmark_history) < 3:
            return 0.8  # 历史数据不足，返回较高分数
        
        try:
            # 计算运动向量的一致性
            prev_points = self.landmark_history[-1]
            prev_prev_points = self.landmark_history[-2]
            
            # 当前运动向量
            current_motion = landmark_points[:, :2] - prev_points[:, :2]
            # 前一帧运动向量
            prev_motion = prev_points[:, :2] - prev_prev_points[:, :2]
            
            # 计算运动向量的相似性
            similarities = []
            for i in range(len(current_motion)):
                if (np.linalg.norm(current_motion[i]) > 0.1 and 
                    np.linalg.norm(prev_motion[i]) > 0.1):
                    # 计算余弦相似度
                    cos_sim = np.dot(current_motion[i], prev_motion[i]) / (
                        np.linalg.norm(current_motion[i]) * np.linalg.norm(prev_motion[i])
                    )
                    similarities.append(max(0.0, cos_sim))
            
            if similarities:
                return np.mean(similarities)
            else:
                return 0.8  # 运动幅度太小，认为是稳定状态
                
        except Exception as e:
            logger.warning(f"Error calculating motion continuity: {e}")
            return 0.5
    
    def get_quality_summary(self) -> Dict[str, float]:
        """
        获取质量评估摘要
        
        Returns:
            Dict[str, float]: 质量摘要统计
        """
        if not self.quality_history:
            return {}
        
        # 计算历史质量的统计信息
        all_qualities = []
        for frame_qualities in self.quality_history:
            for quality in frame_qualities:
                all_qualities.append(quality.overall_quality)
        
        if not all_qualities:
            return {}
        
        return {
            'avg_quality': np.mean(all_qualities),
            'min_quality': np.min(all_qualities),
            'max_quality': np.max(all_qualities),
            'std_quality': np.std(all_qualities),
            'recent_quality': all_qualities[-1] if all_qualities else 0.0
        }
    
    def reset_history(self) -> None:
        """
        重置历史记录
        """
        self.landmark_history.clear()
        self.quality_history.clear()
        logger.info("Quality assessor history reset")
    
    def is_quality_acceptable(self, 
                            quality_metrics: List[QualityMetrics],
                            threshold: float = 0.7) -> bool:
        """
        判断检测质量是否可接受
        
        Args:
            quality_metrics: 质量评估结果
            threshold: 质量阈值
            
        Returns:
            bool: 质量是否可接受
        """
        if not quality_metrics:
            return False
        
        # 检查是否有任何手部质量达到阈值
        return any(metrics.overall_quality >= threshold for metrics in quality_metrics)