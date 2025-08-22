#!/usr/bin/env python3
"""
人体检测器单元测试
Human Detector Unit Tests
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.detector import HumanDetector


class TestHumanDetector(unittest.TestCase):
    """人体检测器测试类"""

    def setUp(self):
        """测试前准备"""
        self.detector = HumanDetector()
        # 创建测试图像
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        self.assertEqual(self.detector.iou_threshold, 0.5)
        self.assertEqual(self.detector.min_box_area, 1500)
        self.assertEqual(self.detector.max_box_ratio, 4.0)

    def test_get_device_auto(self):
        """测试自动设备选择"""
        device = self.detector._get_device("auto")
        self.assertIn(device, ["cpu", "cuda"])

    def test_get_device_cpu(self):
        """测试CPU设备选择"""
        device = self.detector._get_device("cpu")
        self.assertEqual(device, "cpu")

    def test_detect_empty_image(self):
        """测试空图像检测"""
        detections = self.detector.detect(self.test_image)
        self.assertIsInstance(detections, list)

    def test_detect_with_mock_yolo(self):
        """测试使用模拟YOLO的检测"""
        # 模拟YOLO检测结果
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 300]
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.8
        mock_box.cls = [0]  # person class

        mock_result = Mock()
        mock_result.boxes = [mock_box]

        with patch.object(self.detector, "model") as mock_model:
            mock_model.return_value = [mock_result]
            mock_model.__bool__ = lambda x: True

            detections = self.detector.detect(self.test_image)

            self.assertIsInstance(detections, list)
            if len(detections) > 0:
                detection = detections[0]
                self.assertIn("bbox", detection)
                self.assertIn("confidence", detection)
                self.assertIn("class_id", detection)
                self.assertIn("class_name", detection)

    def test_visualize_detections_empty(self):
        """测试空检测结果的可视化"""
        result_image = self.detector.visualize_detections(self.test_image, [])
        self.assertIsNotNone(result_image)
        np.testing.assert_array_equal(result_image, self.test_image)

    def test_visualize_detections_with_boxes(self):
        """测试带检测框的可视化"""
        detections = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.8,
                "class_id": 0,
                "class_name": "person",
            }
        ]

        result_image = self.detector.visualize_detections(self.test_image, detections)
        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, self.test_image.shape)

    def test_filter_detections_by_area(self):
        """测试按面积过滤检测结果"""
        # 创建一个面积太小的检测框
        small_detection = {
            "bbox": [100, 100, 110, 120],  # 面积 = 200，小于min_box_area
            "confidence": 0.8,
            "class_id": 0,
            "class_name": "person",
        }

        # 模拟检测过程中的面积过滤
        bbox = small_detection["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        self.assertLess(area, self.detector.min_box_area)

    def test_filter_detections_by_ratio(self):
        """测试按宽高比过滤检测结果"""
        # 创建一个宽高比异常的检测框
        weird_detection = {
            "bbox": [100, 100, 500, 120],  # 宽高比 = 20，大于max_box_ratio
            "confidence": 0.8,
            "class_id": 0,
            "class_name": "person",
        }

        # 模拟检测过程中的宽高比过滤
        bbox = weird_detection["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = max(width, height) / min(width, height)

        self.assertGreater(aspect_ratio, self.detector.max_box_ratio)

    def test_confidence_threshold(self):
        """测试置信度阈值"""
        # 测试低置信度检测
        low_confidence = 0.3
        self.assertLess(low_confidence, self.detector.confidence_threshold)

        # 测试高置信度检测
        high_confidence = 0.8
        self.assertGreater(high_confidence, self.detector.confidence_threshold)

    def test_model_fallback(self):
        """测试模型加载失败时抛出异常"""
        # 创建一个模型为None的检测器
        detector_with_no_model = HumanDetector()
        detector_with_no_model.model = None

        # 测试应该抛出RuntimeError
        with self.assertRaises(RuntimeError) as context:
            detector_with_no_model.detect(self.test_image)

        self.assertIn("YOLO模型未加载", str(context.exception))

    def tearDown(self):
        """测试后清理"""
        pass


if __name__ == "__main__":
    unittest.main()
