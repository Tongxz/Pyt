#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8 发网检测器实现

基于 YOLOv8 的发网检测器，可以直接检测图像中的发网，无需先检测人体再提取头部区域
"""

import os
import sys
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
try:
    from ultralytics import YOLO
except ImportError:
    logging.error("未安装 ultralytics 库，请使用 'pip install ultralytics' 安装")
    raise

logger = logging.getLogger(__name__)

class YOLOHairnetDetector:
    """
    基于 YOLOv8 的发网检测器
    
    直接使用 YOLOv8 模型检测图像中的发网，无需先检测人体再提取头部区域
    """
    
    def __init__(self, model_path: str = 'models/hairnet_detection.pt', 
                 device: str = 'auto', 
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45):
        """
        初始化 YOLOv8 发网检测器
        
        Args:
            model_path: YOLOv8 模型路径，默认为 'models/hairnet_detection.pt'
            device: 计算设备，可选 'cpu', 'cuda', 'auto'
            conf_thres: 置信度阈值，默认为 0.25
            iou_thres: IoU 阈值，默认为 0.45
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = self._load_model()
        
        # 统计信息
        self.total_detections = 0
        self.hairnet_detections = 0
        
        logger.info(f"YOLOHairnetDetector 初始化成功，使用设备: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """
        获取计算设备
        
        Args:
            device: 指定的设备，'auto' 表示自动选择
            
        Returns:
            实际使用的设备名称
        """
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """
        加载 YOLOv8 模型
        
        Returns:
            加载的 YOLOv8 模型
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"模型文件不存在: {self.model_path}，请确保已训练模型")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            model = YOLO(self.model_path)
            logger.info(f"成功加载 YOLOv8 模型: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"加载 YOLOv8 模型失败: {e}")
            raise
    
    def detect(self, image: Union[str, np.ndarray], 
               conf_thres: Optional[float] = None,
               iou_thres: Optional[float] = None) -> Dict[str, Any]:
        """
        检测图像中的发网
        
        Args:
            image: 输入图像路径或 numpy 数组
            conf_thres: 置信度阈值，如果为 None 则使用初始化时设置的值
            iou_thres: IoU 阈值，如果为 None 则使用初始化时设置的值
            
        Returns:
            检测结果字典，包含以下字段:
            - wearing_hairnet: 是否佩戴发网
            - detections: 检测到的所有目标列表，每个目标包含类别、置信度和边界框
            - visualization: 可视化结果图像
        """
        try:
            # 检查输入图像是否有效
            if image is None:
                return self._create_error_result("输入图像为空")
            
            if isinstance(image, str) and not os.path.exists(image):
                return self._create_error_result(f"图像文件不存在: {image}")
            
            if isinstance(image, np.ndarray) and image.size == 0:
                return self._create_error_result("输入图像为空数组")
            
            # 使用传入的阈值或默认阈值
            conf = conf_thres if conf_thres is not None else self.conf_thres
            iou = iou_thres if iou_thres is not None else self.iou_thres
            
            # 运行推理
            results = self.model(image, conf=conf, iou=iou, verbose=False)
            
            # 处理结果
            detections = []
            has_hairnet = False
            hairnet_confidence = 0.0
            
            for r in results:
                boxes = r.boxes  # 边界框
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 边界框坐标
                    conf = float(box.conf[0])  # 置信度
                    cls = int(box.cls[0])  # 类别
                    cls_name = self.model.names[cls]  # 类别名称
                    
                    # 确保所有值都是Python原生类型，可以被JSON序列化
                    detection = {
                        'class': str(cls_name),
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    detections.append(detection)
                    
                    # 检查是否为发网类别
                    if cls_name.lower() == 'hairnet' and conf > hairnet_confidence:
                        has_hairnet = True
                        hairnet_confidence = conf
            
            # 更新统计信息
            self.total_detections += 1
            if has_hairnet:
                self.hairnet_detections += 1
            
            # 创建结果
            # 注意：visualization是numpy数组，需要转换为可序列化的格式
            visualization = results[0].plot() if results else None
            
            result = {
                'wearing_hairnet': has_hairnet,
                'has_hairnet': has_hairnet,  # 兼容旧接口
                'confidence': float(hairnet_confidence),  # 确保是Python原生float类型
                'detections': detections,
                'visualization': visualization,  # 这里visualization仍然是numpy数组，但在API返回前会被转换为base64
                'error': None
            }
            
            logger.info(f"发网检测结果: 佩戴={has_hairnet}, 置信度={hairnet_confidence:.3f}, 检测到的目标数量={len(detections)}")
            return result
            
        except Exception as e:
            logger.error(f"发网检测失败: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        创建错误结果
        
        Args:
            error_message: 错误信息
            
        Returns:
            错误结果字典
        """
        return {
            'wearing_hairnet': False,
            'has_hairnet': False,
            'confidence': 0.0,
            'detections': [],
            'visualization': None,
            'error': error_message
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Returns:
            统计信息字典
        """
        hairnet_rate = 0.0
        if self.total_detections > 0:
            hairnet_rate = self.hairnet_detections / self.total_detections
        
        return {
            'total_detections': int(self.total_detections),  # 确保是Python原生int类型
            'hairnet_detections': int(self.hairnet_detections),  # 确保是Python原生int类型
            'hairnet_rate': float(hairnet_rate)  # 确保是Python原生float类型
        }
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.total_detections = 0
        self.hairnet_detections = 0
        logger.info("统计信息已重置")