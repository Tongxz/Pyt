import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# 导入统一参数配置
from src.config.unified_params import get_unified_params

logger = logging.getLogger(__name__)


class HumanDetector:
    """人体检测器

    基于YOLOv8的人体检测模块，支持实时检测和批量处理
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        初始化人体检测器

        Args:
            model_path: YOLO模型路径，如果为None则使用统一配置
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        # 获取统一参数配置
        self.params = get_unified_params().human_detection

        # 使用统一配置或传入参数
        if model_path is None:
            model_path = self.params.model_path
        if device == "auto":
            device = self.params.device

        self.device = self._get_device(device)
        self.model = self._load_model(model_path)

        # 使用统一参数配置
        self.confidence_threshold = self.params.confidence_threshold
        self.iou_threshold = self.params.iou_threshold
        self.min_box_area = self.params.min_box_area
        self.max_box_ratio = self.params.max_box_ratio
        self.min_width = self.params.min_width
        self.min_height = self.params.min_height
        self.nms_threshold = self.params.nms_threshold
        self.max_detections = self.params.max_detections

        logger.info(
            f"HumanDetector initialized on {self.device} with unified params: "
            f"conf={self.confidence_threshold}, iou={self.iou_threshold}, "
            f"min_area={self.min_box_area}"
        )

    def _get_device(self, device: str) -> str:
        """获取计算设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, model_path: str):
        """加载YOLO模型"""
        try:
            model = YOLO(model_path)
            # 在测试环境中使用的 DummyYOLO 可能不实现 .to 方法，这里做兼容处理
            if hasattr(model, "to"):
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
        if self.model is None:
            error_msg = "YOLO模型未正确加载，无法进行人体检测"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            logger.info(
                f"开始YOLO检测，图像尺寸: {image.shape}, 置信度阈值: {self.confidence_threshold}, IoU阈值: {self.iou_threshold}"
            )

            results = self.model(
                image, conf=self.confidence_threshold, iou=self.iou_threshold
            )
            detections = []
            total_boxes = 0
            filtered_boxes = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    total_boxes += len(boxes)
                    logger.info(f"YOLO原始检测到 {len(boxes)} 个目标")

                    for box in boxes:
                        # 只检测人体 (class_id = 0)
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())

                            # 计算检测框属性
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            aspect_ratio = max(width, height) / min(width, height)

                            logger.debug(
                                f"检测框: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), 置信度: {confidence:.3f}, 面积: {area:.1f}, 宽高比: {aspect_ratio:.2f}"
                            )

                            # 应用后处理过滤
                            if (
                                area >= self.min_box_area
                                and aspect_ratio <= self.max_box_ratio
                                and width > self.min_width
                                and height > self.min_height
                            ):  # 使用配置的最小尺寸要求
                                detection = {
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": confidence,
                                    "class_id": 0,
                                    "class_name": "person",
                                }
                                detections.append(detection)
                                logger.debug(f"检测框通过过滤: {detection}")
                            else:
                                filtered_boxes += 1
                                logger.debug(
                                    f"检测框被过滤: 面积={area:.1f} (最小={self.min_box_area}), 宽高比={aspect_ratio:.2f} (最大={self.max_box_ratio}), 尺寸={width:.1f}x{height:.1f}"
                                )

            logger.info(
                f"YOLO检测完成: 原始检测框={total_boxes}, 过滤后={len(detections)}, 被过滤={filtered_boxes}"
            )
            return detections

        except Exception as e:
            error_msg = f"YOLO检测过程中发生错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

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

    def visualize_detections(
        self, image: np.ndarray, detections: List[Dict]
    ) -> np.ndarray:
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
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]

            x1, y1, x2, y2 = bbox

            # 绘制边界框
            cv2.rectangle(
                result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )

            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result_image,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return result_image
