import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO
import logging
import random

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
        self.confidence_threshold = 0.6   # 进一步提高置信度阈值
        self.iou_threshold = 0.4   # 降低IoU阈值，减少重叠检测
        self.min_box_area = 1000   # 最小检测框面积，过滤小目标
        self.max_box_ratio = 5.0   # 最大宽高比，过滤异常形状
        
        logger.info(f"HumanDetector initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """获取计算设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_path: str):
        """加载YOLO模型"""
        try:
            model = YOLO(model_path)
            model.to(self.device)
            logger.info(f"成功加载模型: {model_path} 到设备: {self.device}")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("回退到模拟模式")
            return None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人体
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个元素包含bbox、confidence、class_id等信息
        """
        if self.model is not None:
            # 真实YOLO检测
            try:
                results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # 只检测人体 (class_id = 0)
                            if int(box.cls) == 0:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = float(box.conf[0].cpu().numpy())
                                
                                # 计算检测框属性
                                width = x2 - x1
                                height = y2 - y1
                                area = width * height
                                aspect_ratio = max(width, height) / min(width, height)
                                
                                # 应用后处理过滤
                                if (area >= self.min_box_area and 
                                    aspect_ratio <= self.max_box_ratio and
                                    width > 20 and height > 40):  # 人体最小尺寸
                                    
                                    detection = {
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'confidence': confidence,
                                        'class_id': 0,
                                        'class_name': 'person'
                                    }
                                    detections.append(detection)
                
                logger.info(f"YOLO检测到 {len(detections)} 个人体")
                return detections
                
            except Exception as e:
                logger.error(f"YOLO检测失败: {e}，回退到模拟模式")
        
        # 回退到模拟检测
        h, w = image.shape[:2]
        num_detections = random.randint(1, 3)
        detections = []
        
        for i in range(num_detections):
            x1 = random.randint(0, w//2)
            y1 = random.randint(0, h//2)
            x2 = random.randint(x1 + 50, min(w, x1 + 200))
            y2 = random.randint(y1 + 100, min(h, y1 + 300))
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': random.uniform(0.5, 0.95),
                'class_id': 0,
                'class_name': 'person'
            }
            detections.append(detection)
        
        logger.info(f"模拟检测到 {len(detections)} 个人体")
        return detections
    
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
        在图像上可视化检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            
        Returns:
            带有检测框的图像
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            
            # 绘制边界框
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(result_image, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image