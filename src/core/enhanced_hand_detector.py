"""增强的手部检测器模块

该模块提供增强的手部检测功能，集成了MediaPipe检测、质量评估、
备用检测方案和多模型融合策略。

Author: Trae AI Assistant
Date: 2024
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum

from .quality_assessor import HandDetectionQualityAssessor, QualityMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DetectionMode(Enum):
    """检测模式枚举"""
    PRIMARY_ONLY = "primary_only"  # 仅使用主检测器
    WITH_FALLBACK = "with_fallback"  # 使用备用检测器
    FUSION = "fusion"  # 多模型融合


@dataclass
class HandDetectionResult:
    """手部检测结果数据类"""
    landmarks: Optional[Any]  # MediaPipe关键点数据
    bbox: Optional[List[int]]  # 边界框 [x1, y1, x2, y2]
    confidence: float  # 检测置信度
    quality_metrics: Optional[QualityMetrics]  # 质量评估结果
    detection_source: str  # 检测来源（primary/fallback/fusion）
    hand_label: str  # 手部标签（Left/Right）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'quality_metrics': self.quality_metrics.to_dict() if self.quality_metrics else None,
            'detection_source': self.detection_source,
            'hand_label': self.hand_label
        }


class SkinColorHandDetector:
    """基于肤色的备用手部检测器"""
    
    def __init__(self):
        """初始化肤色检测器"""
        # HSV肤色范围
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 形态学操作核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        logger.info("SkinColorHandDetector initialized")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        基于肤色检测手部区域
        
        Args:
            image: 输入图像
            
        Returns:
            List[Dict]: 检测到的手部区域列表
        """
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 肤色掩码
            skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
            
            # 形态学操作去噪
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            hand_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 过滤小区域
                if area > 1000:  # 最小面积阈值
                    # 计算边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 计算置信度（基于面积和形状）
                    confidence = min(1.0, area / 10000.0)  # 归一化面积
                    
                    hand_regions.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'area': area,
                        'contour': contour
                    })
            
            # 按置信度排序
            hand_regions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return hand_regions[:4]  # 最多返回4个手部区域
            
        except Exception as e:
            logger.error(f"Error in skin color detection: {e}")
            return []


class EnhancedHandDetector:
    """增强的手部检测器
    
    集成MediaPipe检测、质量评估、备用检测方案和多模型融合。
    """
    
    def __init__(self,
                 detection_mode: DetectionMode = DetectionMode.WITH_FALLBACK,
                 max_num_hands: int = 4,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 quality_threshold: float = 0.7):
        """
        初始化增强手部检测器
        
        Args:
            detection_mode: 检测模式
            max_num_hands: 最大检测手部数量
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            quality_threshold: 质量阈值
        """
        self.detection_mode = detection_mode
        self.quality_threshold = quality_threshold
        
        # 初始化MediaPipe检测器
        self.mp_hands = mp.solutions.hands
        self.primary_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 初始化备用检测器
        if detection_mode in [DetectionMode.WITH_FALLBACK, DetectionMode.FUSION]:
            self.fallback_detector = SkinColorHandDetector()
        else:
            self.fallback_detector = None
        
        # 初始化质量评估器
        self.quality_assessor = HandDetectionQualityAssessor(
            history_size=10,
            min_confidence=min_detection_confidence,
            stability_threshold=0.02
        )
        
        # 检测统计
        self.detection_stats = {
            'total_detections': 0,
            'primary_success': 0,
            'fallback_used': 0,
            'fusion_used': 0,
            'quality_failures': 0
        }
        
        logger.info(f"EnhancedHandDetector initialized with mode: {detection_mode.value}")
    
    def detect_hands_robust(self, image: np.ndarray) -> List[HandDetectionResult]:
        """
        鲁棒的手部检测
        
        Args:
            image: 输入图像
            
        Returns:
            List[HandDetectionResult]: 检测结果列表
        """
        self.detection_stats['total_detections'] += 1
        
        try:
            # 主检测器检测
            primary_results = self._detect_with_primary(image)
            
            # 质量评估
            quality_results = self._assess_detection_quality(primary_results, image)
            
            # 根据检测模式和质量决定最终结果
            if self.detection_mode == DetectionMode.PRIMARY_ONLY:
                return self._process_primary_results(primary_results, quality_results)
            
            elif self.detection_mode == DetectionMode.WITH_FALLBACK:
                return self._detect_with_fallback(image, primary_results, quality_results)
            
            elif self.detection_mode == DetectionMode.FUSION:
                return self._detect_with_fusion(image, primary_results, quality_results)
            
            else:
                logger.warning(f"Unknown detection mode: {self.detection_mode}")
                return self._process_primary_results(primary_results, quality_results)
                
        except Exception as e:
            logger.error(f"Error in robust hand detection: {e}")
            return []
    
    def _detect_with_primary(self, image: np.ndarray) -> Any:
        """
        使用主检测器进行检测
        
        Args:
            image: 输入图像
            
        Returns:
            MediaPipe检测结果
        """
        try:
            # 转换图像格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe检测
            results = self.primary_detector.process(rgb_image)
            
            if results and results.multi_hand_landmarks:
                self.detection_stats['primary_success'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in primary detection: {e}")
            return None
    
    def _assess_detection_quality(self, 
                                results: Any, 
                                image: np.ndarray) -> List[QualityMetrics]:
        """
        评估检测质量
        
        Args:
            results: MediaPipe检测结果
            image: 输入图像
            
        Returns:
            List[QualityMetrics]: 质量评估结果
        """
        try:
            return self.quality_assessor.assess_quality(results, image)
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return []
    
    def _process_primary_results(self, 
                               results: Any,
                               quality_results: List[QualityMetrics]) -> List[HandDetectionResult]:
        """
        处理主检测器结果
        
        Args:
            results: MediaPipe检测结果
            quality_results: 质量评估结果
            
        Returns:
            List[HandDetectionResult]: 处理后的检测结果
        """
        if not results or not results.multi_hand_landmarks:
            return []
        
        detection_results = []
        
        for i, (landmarks, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness or [])):
            
            # 获取质量评估结果
            quality_metrics = quality_results[i] if i < len(quality_results) else None
            
            # 计算边界框
            bbox = self._calculate_bbox_from_landmarks(landmarks)
            
            # 获取手部标签
            hand_label = handedness.classification[0].label if handedness else "Unknown"
            
            # 获取置信度
            confidence = handedness.classification[0].score if handedness else 0.5
            
            detection_result = HandDetectionResult(
                landmarks=landmarks,
                bbox=bbox,
                confidence=confidence,
                quality_metrics=quality_metrics,
                detection_source="primary",
                hand_label=hand_label
            )
            
            detection_results.append(detection_result)
        
        return detection_results
    
    def _detect_with_fallback(self, 
                            image: np.ndarray,
                            primary_results: Any,
                            quality_results: List[QualityMetrics]) -> List[HandDetectionResult]:
        """
        使用备用检测器的检测策略
        
        Args:
            image: 输入图像
            primary_results: 主检测器结果
            quality_results: 质量评估结果
            
        Returns:
            List[HandDetectionResult]: 最终检测结果
        """
        # 处理主检测器结果
        primary_detections = self._process_primary_results(primary_results, quality_results)
        
        # 检查质量是否可接受
        if self.quality_assessor.is_quality_acceptable(quality_results, self.quality_threshold):
            return primary_detections
        
        # 质量不佳，使用备用检测器
        self.detection_stats['fallback_used'] += 1
        self.detection_stats['quality_failures'] += 1
        
        logger.info("Primary detection quality poor, using fallback detector")
        
        try:
            fallback_regions = self.fallback_detector.detect(image)
            fallback_results = []
            
            for i, region in enumerate(fallback_regions):
                detection_result = HandDetectionResult(
                    landmarks=None,  # 备用检测器不提供关键点
                    bbox=region['bbox'],
                    confidence=region['confidence'],
                    quality_metrics=None,
                    detection_source="fallback",
                    hand_label=f"Hand_{i}"
                )
                fallback_results.append(detection_result)
            
            # 如果备用检测器也没有结果，返回主检测器结果
            return fallback_results if fallback_results else primary_detections
            
        except Exception as e:
            logger.error(f"Error in fallback detection: {e}")
            return primary_detections
    
    def _detect_with_fusion(self, 
                          image: np.ndarray,
                          primary_results: Any,
                          quality_results: List[QualityMetrics]) -> List[HandDetectionResult]:
        """
        多模型融合检测策略
        
        Args:
            image: 输入图像
            primary_results: 主检测器结果
            quality_results: 质量评估结果
            
        Returns:
            List[HandDetectionResult]: 融合后的检测结果
        """
        self.detection_stats['fusion_used'] += 1
        
        # 获取主检测器结果
        primary_detections = self._process_primary_results(primary_results, quality_results)
        
        # 获取备用检测器结果
        try:
            fallback_regions = self.fallback_detector.detect(image)
        except Exception as e:
            logger.error(f"Error in fallback detection during fusion: {e}")
            fallback_regions = []
        
        # 融合结果
        fused_results = self._fuse_detection_results(primary_detections, fallback_regions, image)
        
        return fused_results
    
    def _fuse_detection_results(self, 
                              primary_detections: List[HandDetectionResult],
                              fallback_regions: List[Dict[str, Any]],
                              image: np.ndarray) -> List[HandDetectionResult]:
        """
        融合主检测器和备用检测器的结果
        
        Args:
            primary_detections: 主检测器结果
            fallback_regions: 备用检测器结果
            image: 输入图像
            
        Returns:
            List[HandDetectionResult]: 融合后的结果
        """
        fused_results = []
        
        # 如果主检测器质量良好，优先使用主检测器结果
        high_quality_primary = []
        low_quality_primary = []
        
        for detection in primary_detections:
            if (detection.quality_metrics and 
                detection.quality_metrics.overall_quality >= self.quality_threshold):
                high_quality_primary.append(detection)
            else:
                low_quality_primary.append(detection)
        
        # 添加高质量的主检测器结果
        fused_results.extend(high_quality_primary)
        
        # 对于低质量的主检测器结果，尝试与备用检测器结果匹配
        used_fallback_indices = set()
        
        for primary_det in low_quality_primary:
            if primary_det.bbox:
                # 寻找最匹配的备用检测结果
                best_match_idx = self._find_best_matching_region(
                    primary_det.bbox, fallback_regions, used_fallback_indices
                )
                
                if best_match_idx is not None:
                    # 融合结果
                    fallback_region = fallback_regions[best_match_idx]
                    
                    # 创建融合结果
                    fused_detection = HandDetectionResult(
                        landmarks=primary_det.landmarks,  # 保留关键点信息
                        bbox=self._merge_bboxes(primary_det.bbox, fallback_region['bbox']),
                        confidence=(primary_det.confidence + fallback_region['confidence']) / 2,
                        quality_metrics=primary_det.quality_metrics,
                        detection_source="fusion",
                        hand_label=primary_det.hand_label
                    )
                    
                    fused_results.append(fused_detection)
                    used_fallback_indices.add(best_match_idx)
                else:
                    # 没有匹配的备用结果，保留原始结果
                    fused_results.append(primary_det)
        
        # 添加未使用的备用检测结果
        for i, fallback_region in enumerate(fallback_regions):
            if i not in used_fallback_indices:
                detection_result = HandDetectionResult(
                    landmarks=None,
                    bbox=fallback_region['bbox'],
                    confidence=fallback_region['confidence'],
                    quality_metrics=None,
                    detection_source="fallback",
                    hand_label=f"Hand_{i}"
                )
                fused_results.append(detection_result)
        
        return fused_results
    
    def _find_best_matching_region(self, 
                                 primary_bbox: List[int],
                                 fallback_regions: List[Dict[str, Any]],
                                 used_indices: set) -> Optional[int]:
        """
        寻找与主检测器边界框最匹配的备用检测区域
        
        Args:
            primary_bbox: 主检测器边界框
            fallback_regions: 备用检测器区域列表
            used_indices: 已使用的索引集合
            
        Returns:
            Optional[int]: 最佳匹配的索引，如果没有找到则返回None
        """
        best_iou = 0.0
        best_idx = None
        
        for i, region in enumerate(fallback_regions):
            if i in used_indices:
                continue
            
            # 计算IoU
            iou = self._calculate_bbox_iou(primary_bbox, region['bbox'])
            
            if iou > best_iou and iou > 0.3:  # IoU阈值
                best_iou = iou
                best_idx = i
        
        return best_idx
    
    def _calculate_bbox_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            bbox1: 边界框1 [x1, y1, x2, y2]
            bbox2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        try:
            # 计算交集
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            
            # 计算并集
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _merge_bboxes(self, bbox1: List[int], bbox2: List[int]) -> List[int]:
        """
        合并两个边界框
        
        Args:
            bbox1: 边界框1
            bbox2: 边界框2
            
        Returns:
            List[int]: 合并后的边界框
        """
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        
        return [x1, y1, x2, y2]
    
    def _calculate_bbox_from_landmarks(self, landmarks: Any) -> List[int]:
        """
        从关键点计算边界框
        
        Args:
            landmarks: MediaPipe关键点数据
            
        Returns:
            List[int]: 边界框 [x1, y1, x2, y2]
        """
        try:
            if not landmarks or not landmarks.landmark:
                return [0, 0, 0, 0]
            
            # 提取所有关键点坐标
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            
            # 计算边界框（归一化坐标）
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 添加边距
            margin = 0.05
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(1, x_max + margin)
            y_max = min(1, y_max + margin)
            
            return [x_min, y_min, x_max, y_max]
            
        except Exception as e:
            logger.error(f"Error calculating bbox from landmarks: {e}")
            return [0, 0, 0, 0]
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.detection_stats.copy()
        
        if stats['total_detections'] > 0:
            stats['primary_success_rate'] = stats['primary_success'] / stats['total_detections']
            stats['fallback_usage_rate'] = stats['fallback_used'] / stats['total_detections']
            stats['quality_failure_rate'] = stats['quality_failures'] / stats['total_detections']
        
        # 添加质量评估统计
        quality_summary = self.quality_assessor.get_quality_summary()
        stats.update(quality_summary)
        
        return stats
    
    def reset_stats(self) -> None:
        """
        重置统计信息
        """
        self.detection_stats = {
            'total_detections': 0,
            'primary_success': 0,
            'fallback_used': 0,
            'fusion_used': 0,
            'quality_failures': 0
        }
        
        self.quality_assessor.reset_history()
        logger.info("Detection stats and quality history reset")
    
    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'primary_detector'):
                self.primary_detector.close()
        except Exception as e:
            logger.error(f"Error closing MediaPipe detector: {e}")