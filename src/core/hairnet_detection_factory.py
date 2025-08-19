#!/usr/bin/env python

"""
发网检测器工厂

提供不同类型发网检测器的创建和管理
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Union

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

# 尝试导入 YOLOv8 发网检测器
try:
    from src.core.yolo_hairnet_detector import YOLOHairnetDetector

    YOLO_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"导入YOLOHairnetDetector失败: {e}")
    YOLO_DETECTOR_AVAILABLE = False
    YOLOHairnetDetector = None

logger = logging.getLogger(__name__)


class HairnetDetectionFactory:
    """
    发网检测器工厂类

    用于创建和管理YOLOv8发网检测器
    """

    @staticmethod
    def create_detector(detector_type: str = "auto", **kwargs) -> Any:
        """
        创建发网检测器

        Args:
            detector_type: 检测器类型，可选值:
                - 'yolo': 基于YOLOv8的发网检测器（默认）
                - 'auto': 自动选择检测器（目前仅支持YOLOv8）
            **kwargs: 传递给检测器的其他参数
                - model_path: 模型路径
                - device: 计算设备
                - conf_thres: 置信度阈值
                - iou_thres: IoU阈值

        Returns:
            创建的YOLOv8发网检测器实例
        """
        # 获取参数
        model_path = kwargs.get("model_path")
        device = kwargs.get("device", "auto")
        conf_thres = kwargs.get("conf_thres", 0.25)
        iou_thres = kwargs.get("iou_thres", 0.45)

        # 检查YOLOv8是否可用
        if not YOLO_DETECTOR_AVAILABLE:
            raise RuntimeError(
                "YOLOv8 发网检测器不可用。请检查：\n"
                "1. ultralytics 是否正确安装\n"
                "2. PyTorch 是否正确安装\n"
                "3. 相关依赖是否完整"
            )

        # 确保 model_path 不为 None
        if model_path is None:
            model_path = "models/hairnet_detection/hairnet_detection.pt"
            logger.warning(f"未指定模型路径，使用默认路径: {model_path}")

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}。请检查：\n"
                "1. 模型文件路径是否正确\n"
                "2. 模型文件是否已下载\n"
                "3. 文件权限是否正确"
            )

        logger.info(f"创建 YOLOv8 发网检测器，模型: {model_path}, 设备: {device}")
        try:
            # 再次检查YOLOHairnetDetector是否可用
            if YOLOHairnetDetector is None:
                raise ImportError(
                    "YOLOHairnetDetector不可用。请确保已安装ultralytics库: pip install ultralytics"
                )
            # 使用已导入的 YOLOHairnetDetector
            return YOLOHairnetDetector(
                model_path=model_path,
                device=device,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )
        except Exception as e:
            raise RuntimeError(f"创建 YOLOv8 发网检测器失败: {e}")

    @staticmethod
    def is_yolo_available() -> bool:
        """
        检查YOLOv8检测器是否可用

        Returns:
            YOLOv8检测器是否可用
        """
        return YOLO_DETECTOR_AVAILABLE

    @staticmethod
    def get_available_detector_types() -> Dict[str, str]:
        """
        获取可用的检测器类型

        Returns:
            可用的检测器类型字典，键为类型名称，值为描述
        """
        types = {"auto": "自动选择最佳检测器"}

        if YOLO_DETECTOR_AVAILABLE:
            types["yolo"] = "基于YOLOv8的发网检测器"

        return types
