#!/usr/bin/env python3
"""
发网检测器单元测试
Hairnet Detector Unit Tests
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# 模拟检测器类
class MockPersonDetector:
    def detect(self, frame):
        return [{"bbox": [0, 0, 100, 100], "confidence": 0.9}]


class MockHairnetDetector:
    def _detect_hairnet_with_pytorch(self, frame):
        return {"wearing_hairnet": True, "has_hairnet": True, "confidence": 0.85}


from src.core.hairnet_detector import (
    HairnetCNN,
    HairnetDetectionPipeline,
    HairnetDetector,
)


class TestHairnetCNN(unittest.TestCase):
    """发网CNN模型测试类"""

    def setUp(self):
        """测试前准备"""
        self.model = HairnetCNN(num_classes=2)
        self.test_input = torch.randn(1, 3, 64, 64)

    def test_init(self):
        """测试模型初始化"""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsNotNone(self.model.features)
        self.assertIsNotNone(self.model.classifier)

    def test_forward(self):
        """测试前向传播"""
        output = self.model(self.test_input)
        self.assertEqual(output.shape, (1, 2))
        self.assertIsInstance(output, torch.Tensor)

    def test_model_parameters(self):
        """测试模型参数"""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)

        # 检查参数是否可训练
        for param in params:
            self.assertTrue(param.requires_grad)


class TestHairnetDetector(unittest.TestCase):
    """发网检测器测试类"""

    def setUp(self):
        """测试前准备"""
        self.detector = HairnetDetector()
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_get_device(self):
        """测试设备选择"""
        device = self.detector._get_device("auto")
        self.assertIn(device, ["cpu", "cuda"])

        device_cpu = self.detector._get_device("cpu")
        self.assertEqual(device_cpu, "cpu")

    def test_get_transform(self):
        """测试图像变换"""
        transform = self.detector._get_transform()
        self.assertIsNotNone(transform)

        # 测试变换是否正常工作
        pil_image = Image.fromarray(self.test_image)
        transformed = transform(pil_image)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(len(transformed.shape), 3)  # C, H, W

    def test__extract_head_roi_from_bbox_bbox(self):
        """测试基于边界框的头部区域提取"""
        bbox = [50, 20, 150, 180]  # x1, y1, x2, y2
        head_region = self.detector._extract_head_roi_from_bbox(self.test_image, bbox)

        self.assertIsNotNone(head_region)
        self.assertIsInstance(head_region, np.ndarray)
        self.assertEqual(len(head_region.shape), 3)

    def test__extract_head_roi_from_bbox_keypoints(self):
        """测试基于关键点的头部区域提取"""
        # 模拟关键点数据
        keypoints = {
            "nose": [75, 50],
            "left_eye": [70, 45],
            "right_eye": [80, 45],
            "left_ear": [65, 50],
            "right_ear": [85, 50],
        }

        head_region = self.detector._optimize_head_roi_with_keypoints(
            self.test_image, None, keypoints
        )

        self.assertIsNotNone(head_region)
        self.assertIsInstance(head_region, np.ndarray)

    def test_detect_hairnet_with_mock_model(self):
        """测试使用模拟模型的发网检测"""
        head_region = self.test_image[20:80, 30:90]  # 提取头部区域

        with patch.object(self.detector, "model") as mock_model:
            # 模拟模型输出
            mock_output = torch.tensor([[0.3, 0.7]])  # [no_hairnet, has_hairnet]
            mock_model.return_value = mock_output
            mock_model.__bool__ = lambda x: True

            result = self.detector.detect_hairnet(head_region)

            self.assertIsInstance(result, dict)
            self.assertIn("wearing_hairnet", result)
            self.assertIn("confidence", result)
            self.assertIsInstance(result["wearing_hairnet"], bool)
            self.assertIsInstance(result["confidence"], float)

    def test_detect_hairnet_fallback(self):
        """测试模型不可用时返回错误结果"""
        detector_no_model = HairnetDetector()
        detector_no_model.model = None

        # 提供一个有效的人体边界框，确保能够提取头部ROI
        human_bbox = [50, 20, 150, 200]  # x1, y1, x2, y2

        # 测试应该返回错误结果
        result = detector_no_model.detect_hairnet(
            self.test_image, human_bbox=human_bbox
        )

        # 检查返回的错误结果
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("CNN 发网检测模型未加载", result["error"])

    def test_preprocess_image(self):
        """测试图像预处理"""
        processed = self.detector._preprocess_image(self.test_image)

        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(len(processed.shape), 4)  # B, C, H, W
        self.assertEqual(processed.shape[0], 1)  # batch size

    def test_confidence_threshold(self):
        """测试置信度阈值"""
        # 测试低置信度
        low_conf_output = torch.tensor([[0.8, 0.2]])
        confidence = torch.softmax(low_conf_output, dim=1).max().item()
        self.assertLess(confidence, self.detector.confidence_threshold)

        # 测试高置信度
        high_conf_output = torch.tensor([[0.1, 0.9]])
        confidence = torch.softmax(high_conf_output, dim=1).max().item()
        self.assertGreater(confidence, self.detector.confidence_threshold)


class TestHairnetDetectionPipeline(unittest.TestCase):
    """发网检测流水线测试类"""

    def setUp(self):
        """测试前准备"""
        self.pipeline = HairnetDetectionPipeline(
            MockPersonDetector(), MockHairnetDetector()
        )
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_detect_hairnet_compliance_empty_frame(self):
        """测试空帧的发网合规检测"""
        result = self.pipeline.detect_hairnet_compliance(self.test_frame)

        self.assertIsInstance(result, dict)
        self.assertIn("total_persons", result)
        self.assertIn("persons_with_hairnet", result)
        self.assertIn("persons_without_hairnet", result)
        self.assertIn("compliance_rate", result)
        self.assertIn("detections", result)
        self.assertIn("average_confidence", result)

    def test_detect_hairnet_compliance_with_mock_detections(self):
        """测试使用模拟检测结果的发网合规检测"""
        # 模拟人体检测结果
        mock_detections = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.8,
                "class_id": 0,
                "class_name": "person",
            }
        ]

        with patch.object(self.pipeline.person_detector, "detect") as mock_detect:
            mock_detect.return_value = mock_detections

            with patch.object(
                self.pipeline.hairnet_detector, "detect_hairnet"
            ) as mock_hairnet:
                mock_hairnet.return_value = {
                    "wearing_hairnet": True,
                    "confidence": 0.85,
                }

                result = self.pipeline.detect_hairnet_compliance(self.test_frame)

                self.assertEqual(result["total_persons"], 1)
                self.assertEqual(result["persons_with_hairnet"], 1)
                self.assertEqual(result["persons_without_hairnet"], 0)
                self.assertEqual(result["compliance_rate"], 1.0)
                self.assertEqual(len(result["detections"]), 1)

    def testcalculate_compliance_rate(self):
        """测试合规率计算"""
        # 测试100%合规
        rate = self.pipeline.calculate_compliance_rate(5, 5)
        self.assertEqual(rate, 1.0)

        # 测试50%合规
        rate = self.pipeline.calculate_compliance_rate(2, 4)
        self.assertEqual(rate, 0.5)

        # 测试0%合规
        rate = self.pipeline.calculate_compliance_rate(0, 3)
        self.assertEqual(rate, 0.0)

        # 测试无人员情况
        rate = self.pipeline.calculate_compliance_rate(0, 0)
        self.assertEqual(rate, 1.0)

    def test_visualize_detections(self):
        """测试发网检测结果可视化"""
        detections = [
            {
                "bbox": [100, 100, 200, 300],
                "wearing_hairnet": True,
                "confidence": 0.85,
                "hairnet_confidence": 0.85,
            }
        ]

        result_image = self.pipeline.visualize_detections(self.test_frame, detections)

        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, self.test_frame.shape)

    def test_get_detection_statistics(self):
        """测试统计信息获取"""
        stats = self.pipeline.get_detection_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_detections", stats)
        self.assertIn("hairnet_detections", stats)
        self.assertIn("compliance_rate", stats)

    def tearDown(self):
        """测试后清理"""
        pass


if __name__ == "__main__":
    unittest.main()
