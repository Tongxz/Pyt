import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class HumanDetector:
    """人体检测器
    
    基于YOLOv8的人体检测模块，支持实时检测和批量处理
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'auto'):
        """
        初始化人体检测器
        
        Args:
            model_path: YOLO模型路径
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        logger.info(f"HumanDetector initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """获取计算设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_path: str) -> YOLO:
        """加载YOLO模型"""
        try:
            model = YOLO(model_path)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人体
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个元素包含bbox、confidence、class_id等信息
        """
        try:
            # YOLO推理
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[0],  # 只检测人体 (COCO class 0)
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 提取检测信息
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': 'person'
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        批量检测多张图像
        
        Args:
            images: 图像列表
            
        Returns:
            每张图像的检测结果列表
        """
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def set_iou_threshold(self, threshold: float):
        """设置IoU阈值"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"IoU threshold set to {self.iou_threshold}")
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            
        Returns:
            标注后的图像
        """
        vis_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(vis_image, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_image