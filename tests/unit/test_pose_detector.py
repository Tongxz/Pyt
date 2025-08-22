#!/usr/bin/env python3
"""
姿态检测器单元测试

测试 PoseDetectorFactory、MediaPipePoseDetector 和 YOLOv8PoseDetector 类的功能
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from src.core.pose_detector import (
    PoseDetectorFactory,
    MediaPipePoseDetector,
    YOLOv8PoseDetector,
)


class TestPoseDetectorFactory(unittest.TestCase):
    """测试 PoseDetectorFactory 类"""

    def test_create_yolov8_detector(self):
        """测试创建 YOLOv8 姿态检测器"""
        with patch("src.core.pose_detector.YOLOv8PoseDetector") as mock_yolo:
            detector = PoseDetectorFactory.create(
                backend="yolov8",
                model_path="models/yolov8n-pose.pt",
                device="cpu",
            )
            self.assertIsNotNone(detector)
            mock_yolo.assert_called_once()

    def test_create_mediapipe_detector(self):
        """测试创建 MediaPipe 姿态检测器"""
        with patch("src.core.pose_detector.MediaPipePoseDetector") as mock_mp:
            with self.assertRaises(NotImplementedError):
                detector = PoseDetectorFactory.create(backend="mediapipe")

    def test_create_invalid_backend(self):
        """测试创建无效的后端"""
        with self.assertRaises(ValueError):
            PoseDetectorFactory.create(backend="invalid")


class TestMediaPipePoseDetector(unittest.TestCase):
    """测试 MediaPipePoseDetector 类"""

    def setUp(self):
        """设置测试环境"""
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 创建一个简单的测试图像
        cv2_available = True
        try:
            import cv2
            # 在图像中心绘制一个白色矩形作为人体
            cv2.rectangle(self.test_image, (250, 150), (390, 400), (255, 255, 255), -1)
        except ImportError:
            cv2_available = False
        self.cv2_available = cv2_available

    @patch("src.core.pose_detector.mp")
    def test_init(self, mock_mp):
        """测试初始化"""
        mock_mp.solutions.pose.Pose.return_value = Mock()
        detector = MediaPipePoseDetector()
        self.assertIsNotNone(detector)

    @patch("src.core.pose_detector.mp")
    def test_detect(self, mock_mp):
        """测试检测方法"""
        # 模拟 MediaPipe 检测结果
        mock_results = Mock()
        mock_results.pose_landmarks = Mock()
        mock_results.pose_landmarks.landmark = [
            Mock(x=0.5, y=0.3, z=0.0, visibility=0.9),  # 鼻子
            Mock(x=0.4, y=0.4, z=0.0, visibility=0.8),  # 左眼
            Mock(x=0.6, y=0.4, z=0.0, visibility=0.8),  # 右眼
        ]

        mock_pose = Mock()
        mock_pose.process.return_value = mock_results
        mock_mp.solutions.pose.Pose.return_value = mock_pose

        detector = MediaPipePoseDetector()
        results = detector.detect(self.test_image)

        self.assertIsInstance(results, list)
        if results:
            detection = results[0]
            self.assertIn("bbox", detection)
            self.assertIn("confidence", detection)
            self.assertIn("keypoints", detection)
            self.assertIn("xy", detection["keypoints"])
            self.assertIn("conf", detection["keypoints"])

    @patch("src.core.pose_detector.mp")
    def test_cleanup(self, mock_mp):
        """测试资源清理"""
        mock_pose = Mock()
        mock_mp.solutions.pose.Pose.return_value = mock_pose

        detector = MediaPipePoseDetector()
        detector.cleanup()
        # 验证 pose.close() 被调用
        mock_pose.close.assert_called_once()


class TestYOLOv8PoseDetector(unittest.TestCase):
    """测试 YOLOv8PoseDetector 类"""

    def setUp(self):
        """设置测试环境"""
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 创建一个简单的测试图像
        cv2_available = True
        try:
            import cv2
            # 在图像中心绘制一个白色矩形作为人体
            cv2.rectangle(self.test_image, (250, 150), (390, 400), (255, 255, 255), -1)
        except ImportError:
            cv2_available = False
        self.cv2_available = cv2_available

    @patch("src.core.pose_detector.YOLO")
    def test_detect(self, mock_yolo):
        """测试检测方法"""
        # 模拟YOLOv8检测结果
        # 1. 创建模拟的张量 (Tensor-like objects)
        #    - 使用MagicMock来模拟具有 .item() 和 .cpu().numpy() 方法的张量
        
        # 模拟 box.cls[0] -> 0
        mock_cls_tensor = MagicMock()
        mock_cls_tensor.item.return_value = 0

        # 模拟 box.conf[0] -> 0.9
        mock_conf_tensor = MagicMock()
        mock_conf_tensor.item.return_value = 0.9

        # 模拟 box.xyxy[0] -> np.array(...)
        mock_xyxy_tensor = MagicMock()
        mock_xyxy_tensor.cpu.return_value.numpy.return_value = np.array([50, 50, 150, 200])

        # 模拟 keypoints.xy[0] -> np.array(...)
        mock_kpts_xy_tensor = MagicMock()
        kpts_data = np.random.rand(17, 2) * 100
        mock_kpts_xy_tensor.cpu.return_value.numpy.return_value = kpts_data

        # 模拟 keypoints.conf[0] -> np.array(...)
        mock_kpts_conf_tensor = MagicMock()
        kpts_conf_data = np.random.rand(17)
        mock_kpts_conf_tensor.cpu.return_value.numpy.return_value = kpts_conf_data

        # 2. 组装模拟的 Box 和 Keypoints 对象
        mock_box = Mock()
        mock_box.xyxy = [mock_xyxy_tensor]
        mock_box.conf = [mock_conf_tensor]
        mock_box.cls = [mock_cls_tensor]

        mock_keypoints = Mock()
        mock_keypoints.xy = [mock_kpts_xy_tensor]
        mock_keypoints.conf = [mock_kpts_conf_tensor]

        # 3. 组装模拟的 Result 对象
        mock_result = Mock()
        # result.boxes 和 result.keypoints 是可迭代的
        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_keypoints]

        # 4. 组装模拟的 Model 对象
        mock_model = Mock()
        # model(image, ...) 返回一个包含 result 对象的列表
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        # 5. 执行测试
        detector = YOLOv8PoseDetector(
            model_path="models/yolov8n-pose.pt",
            device="cpu",
        )
        results = detector.detect(self.test_image)

        # 6. 断言
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        detection = results[0]
        self.assertIn("bbox", detection)
        self.assertIn("confidence", detection)
        self.assertIn("keypoints", detection)
        self.assertIn("xy", detection["keypoints"])
        self.assertEqual(detection["keypoints"]["xy"].shape, (17, 2))
        self.assertIn("conf", detection["keypoints"])
        self.assertEqual(detection["keypoints"]["conf"].shape, (17,))

    @patch("src.core.pose_detector.YOLO")
    def test_visualize(self, mock_yolo):
        """测试可视化方法"""
        mock_yolo.return_value = Mock()
        detector = YOLOv8PoseDetector(
            model_path="models/yolov8n-pose.pt",
            device="cpu",
        )

        # 创建模拟检测结果，包含17个关键点以避免IndexError
        kpts_xy = np.random.randint(50, 200, size=(17, 2))
        kpts_conf = np.random.uniform(0.6, 1.0, size=(17,))
        
        detections = [{
            "bbox": [50, 50, 150, 200],
            "confidence": 0.9,
            "keypoints": {
                "xy": kpts_xy,
                "conf": kpts_conf
            }
        }]

        vis_image = detector.visualize(self.test_image, detections)
        self.assertEqual(vis_image.shape, self.test_image.shape)


if __name__ == "__main__":
    unittest.main()
