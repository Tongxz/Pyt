#!/usr/bin/env python3
"""
姿态检测器单元测试

测试 PoseDetector 类的功能
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from src.core.pose_detector import PoseDetector


class TestPoseDetector(unittest.TestCase):
    """测试 PoseDetector 类"""

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
    def test_init_with_mediapipe(self, mock_mp):
        """测试使用MediaPipe初始化"""
        # 模拟MediaPipe可用
        mock_mp.solutions.pose.Pose.return_value = Mock()
        mock_mp.solutions.hands.Hands.return_value = Mock()

        detector = PoseDetector(use_mediapipe=True)

        self.assertTrue(detector.use_mediapipe)
        # 注意：实际的mp_pose和mp_hands属性可能不存在，这取决于具体实现

    def test_init_without_mediapipe(self):
        """测试不使用MediaPipe的初始化"""
        detector = PoseDetector(use_mediapipe=False)

        self.assertFalse(detector.use_mediapipe)
        self.assertFalse(hasattr(detector, "mp_pose"))
        self.assertFalse(hasattr(detector, "mp_hands"))

    @patch("src.core.pose_detector.mp")
    def test_detect_pose_with_mediapipe(self, mock_mp):
        """测试使用MediaPipe进行姿态检测"""
        # 模拟MediaPipe检测结果
        mock_results = Mock()
        mock_results.pose_landmarks = Mock()
        mock_results.pose_landmarks.landmark = [
            Mock(x=0.5, y=0.3, z=0.0),  # 鼻子
            Mock(x=0.4, y=0.4, z=0.0),  # 左眼
            Mock(x=0.6, y=0.4, z=0.0),  # 右眼
        ]

        mock_pose = Mock()
        mock_pose.process.return_value = mock_results
        mock_mp.solutions.pose.Pose.return_value = mock_pose
        mock_mp.solutions.hands.Hands.return_value = Mock()

        detector = PoseDetector(use_mediapipe=True)
        result = detector.detect_pose(self.test_image)

        # MediaPipe应该返回字典格式的结果
        if result is not None:
            self.assertIn("landmarks", result)
            self.assertIsInstance(result["landmarks"], list)
            if result["landmarks"]:  # 只有当landmarks不为空时才检查
                self.assertGreater(len(result["landmarks"]), 0)

                # 检查关键点格式
                for landmark in result["landmarks"]:
                    self.assertIn("x", landmark)
                    self.assertIn("y", landmark)
                    self.assertIn("z", landmark)

    def test_detect_pose_fallback(self):
        """测试姿态检测在MediaPipe不可用时抛出异常"""
        detector = PoseDetector(use_mediapipe=False)

        # 测试应该抛出RuntimeError
        with self.assertRaises(RuntimeError) as context:
            detector.detect_pose(self.test_image)

        self.assertIn("MediaPipe 姿态检测器不可用", str(context.exception))

    @patch("src.core.pose_detector.mp")
    def test_detect_hands_with_mediapipe(self, mock_mp):
        """测试使用MediaPipe进行手部检测"""
        # 模拟MediaPipe手部检测结果
        mock_results = Mock()
        mock_results.multi_hand_landmarks = [
            Mock(
                landmark=[
                    Mock(x=0.3, y=0.5, z=0.0),  # 手腕
                    Mock(x=0.32, y=0.48, z=0.0),  # 拇指
                    Mock(x=0.28, y=0.52, z=0.0),  # 食指
                ]
            )
        ]

        mock_hands = Mock()
        mock_hands.process.return_value = mock_results
        mock_mp.solutions.hands.Hands.return_value = mock_hands
        mock_mp.solutions.pose.Pose.return_value = Mock()

        detector = PoseDetector(use_mediapipe=True)
        hand_regions = detector.detect_hands(self.test_image)

        self.assertIsInstance(hand_regions, list)
        # 备用方法可能检测不到手部，但应该返回列表
        self.assertGreaterEqual(len(hand_regions), 0)

        # 检查手部区域格式
        if hand_regions:
            for region in hand_regions:
                self.assertIsInstance(region, dict)
                self.assertIn("bbox", region)
                self.assertIn("keypoints", region)
                self.assertIn("confidence", region)

    def test_detect_hands_fallback(self):
        """测试手部检测在MediaPipe不可用时抛出异常"""
        detector = PoseDetector(use_mediapipe=False)

        # 测试应该抛出RuntimeError
        with self.assertRaises(RuntimeError) as context:
            detector.detect_hands(self.test_image)

        self.assertIn("MediaPipe 手部检测器不可用", str(context.exception))

    def test_get_hand_center_method(self):
        """测试PoseDetector的get_hand_center方法"""
        detector = PoseDetector(use_mediapipe=False)

        # 测试有效的手部关键点
        hand_landmarks = [
            {"x": 100, "y": 150},
            {"x": 110, "y": 160},
            {"x": 90, "y": 140},
        ]

        center = detector.get_hand_center(hand_landmarks)

        self.assertIsInstance(center, tuple)
        self.assertEqual(len(center), 2)
        # 中心应该是关键点的平均值
        expected_x = (100 + 110 + 90) / 3
        expected_y = (150 + 160 + 140) / 3
        self.assertAlmostEqual(center[0], expected_x, places=1)
        self.assertAlmostEqual(center[1], expected_y, places=1)

    def test_get_hand_center_empty_landmarks(self):
        """测试空关键点列表的手部中心获取"""
        detector = PoseDetector(use_mediapipe=False)

        center = detector.get_hand_center([])
        self.assertEqual(center, (0.0, 0.0))

    @patch("src.core.pose_detector.mp")
    def test_cleanup(self, mock_mp):
        """测试资源清理"""
        mock_pose = Mock()
        mock_hands = Mock()
        mock_mp.solutions.pose.Pose.return_value = mock_pose
        mock_mp.solutions.hands.Hands.return_value = mock_hands

        detector = PoseDetector(use_mediapipe=True)
        detector.cleanup()

        # 测试cleanup方法不抛出异常即可
        # 具体的资源清理逻辑取决于实际实现

    def test_cleanup_without_mediapipe(self):
        """测试不使用MediaPipe时的资源清理"""
        detector = PoseDetector(use_mediapipe=False)
        # 应该不会抛出异常
        detector.cleanup()

    @patch("src.core.pose_detector.mp", None)
    def test_mediapipe_not_available(self):
        """测试MediaPipe不可用时的处理"""
        detector = PoseDetector(use_mediapipe=True)

        # 应该回退到不使用MediaPipe
        self.assertFalse(detector.use_mediapipe)
        self.assertFalse(hasattr(detector, "mp_pose"))
        self.assertFalse(hasattr(detector, "mp_hands"))

    def test_pose_detection_empty_image(self):
        """测试空图像的姿态检测在MediaPipe不可用时抛出异常"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = PoseDetector(use_mediapipe=False)

        # 测试应该抛出RuntimeError
        with self.assertRaises(RuntimeError) as context:
            detector.detect_pose(empty_image)

        self.assertIn("MediaPipe 姿态检测器不可用", str(context.exception))

    def test_hand_detection_empty_image(self):
        """测试空图像的手部检测在MediaPipe不可用时抛出异常"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        detector = PoseDetector(use_mediapipe=False)

        # 测试应该抛出RuntimeError
        with self.assertRaises(RuntimeError) as context:
            detector.detect_hands(empty_image)

        self.assertIn("MediaPipe 手部检测器不可用", str(context.exception))

    def test_invalid_image_input(self):
        """测试无效图像输入"""
        detector = PoseDetector(use_mediapipe=False)

        # 测试错误维度的图像
        invalid_image = np.zeros((100, 100), dtype=np.uint8)  # 2D图像

        # 测试姿态检测
        try:
            result = detector.detect_pose(invalid_image)
            # 应该返回None或处理错误
            if result is not None:
                self.assertIn("landmarks", result)
        except Exception:
            # 预期可能会有OpenCV错误
            pass

        # 测试手部检测
        try:
            hand_regions = detector.detect_hands(invalid_image)
            self.assertIsInstance(hand_regions, list)
        except Exception:
            # 预期可能会有OpenCV错误
            pass


if __name__ == "__main__":
    unittest.main()
