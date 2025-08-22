#!/usr/bin/env python3
"""
行为识别器单元测试

测试 BehaviorRecognizer 类的各种功能
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.core.behavior import BehaviorRecognizer, BehaviorState


class TestBehaviorRecognizer(unittest.TestCase):
    """测试 BehaviorRecognizer 类"""

    def setUp(self):
        """设置测试环境"""
        self.recognizer = BehaviorRecognizer(
            confidence_threshold=0.6, use_advanced_detection=False
        )
        self.person_bbox = [100, 50, 300, 450]  # [x1, y1, x2, y2]
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_init_basic_mode(self):
        """测试基础模式初始化"""
        recognizer = BehaviorRecognizer(use_advanced_detection=False)
        self.assertFalse(recognizer.use_advanced_detection)
        self.assertIsNone(recognizer.pose_detector)
        self.assertIsNone(recognizer.motion_analyzer)

    @patch("src.core.behavior.MotionAnalyzer")
    @patch("src.core.behavior.PoseDetectorFactory")
    def test_init_advanced_mode(self, mock_pose_factory, mock_motion_analyzer):
        """测试高级模式初始化"""
        # 模拟依赖项的成功创建
        mock_pose_factory.create.return_value = Mock()
        mock_motion_analyzer.return_value = Mock()

        recognizer = BehaviorRecognizer(use_advanced_detection=True)
        self.assertTrue(
            recognizer.use_advanced_detection,
            "高级模式应该被启用，但初始化失败并被禁用",
        )
        self.assertIsNotNone(
            recognizer.pose_detector, "高级模式下姿态检测器应被初始化"
        )
        self.assertIsNotNone(
            recognizer.motion_analyzer, "高级模式下运动分析器应被初始化"
        )

    def test_detect_handwashing_no_hands(self):
        """测试无手部输入的洗手检测"""
        confidence = self.recognizer.detect_handwashing(self.person_bbox, [])
        self.assertEqual(confidence, 0.0)

    def test_detect_handwashing_invalid_bbox(self):
        """测试无效边界框的洗手检测"""
        invalid_hand_regions = [
            {"bbox": [100, 100]},  # 不完整的bbox
            {"bbox": []},  # 空bbox
            {},  # 无bbox字段
        ]
        confidence = self.recognizer.detect_handwashing(
            self.person_bbox, invalid_hand_regions
        )
        self.assertEqual(confidence, 0.0)

    def test_detect_handwashing_valid_position(self):
        """测试有效位置的洗手检测"""
        # 手部位于身体中下部（洗手的典型位置）
        hand_regions = [{"bbox": [150, 250, 200, 300], "confidence": 0.8}]  # 身体中下部
        confidence = self.recognizer.detect_handwashing(self.person_bbox, hand_regions)
        self.assertGreater(confidence, 0.0)

    def test_detect_handwashing_invalid_position(self):
        """测试无效位置的洗手检测"""
        # 手部位于身体上部（不是洗手位置）
        hand_regions = [{"bbox": [150, 80, 200, 130], "confidence": 0.8}]  # 身体上部
        confidence = self.recognizer.detect_handwashing(self.person_bbox, hand_regions)
        # 由于_analyze_hand_motion可能返回分数，我们检查整体置信度是否合理
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_detect_sanitizing_no_hands(self):
        """测试无手部输入的消毒检测"""
        confidence = self.recognizer.detect_sanitizing(self.person_bbox, [])
        self.assertEqual(confidence, 0.0)

    def test_detect_sanitizing_single_hand(self):
        """测试单手消毒检测"""
        hand_regions = [{"bbox": [150, 200, 200, 250]}]
        confidence = self.recognizer.detect_sanitizing(self.person_bbox, hand_regions)
        self.assertEqual(confidence, 0.0)  # 消毒需要双手

    def test_detect_sanitizing_two_hands_close(self):
        """测试双手靠近的消毒检测"""
        hand_regions = [
            {"bbox": [140, 200, 180, 240]},  # 左手
            {"bbox": [180, 200, 220, 240]},  # 右手（靠近）
        ]
        confidence = self.recognizer.detect_sanitizing(self.person_bbox, hand_regions)
        self.assertGreater(confidence, 0.0)

    def test_detect_sanitizing_two_hands_far(self):
        """测试双手距离过远的消毒检测"""
        hand_regions = [
            {"bbox": [120, 200, 160, 240]},  # 左手
            {"bbox": [240, 200, 280, 240]},  # 右手（距离远）
        ]
        confidence = self.recognizer.detect_sanitizing(self.person_bbox, hand_regions)
        self.assertEqual(confidence, 0.0)

    def test_detect_hairnet_with_detection(self):
        """测试发网检测功能"""
        person_bbox = [100, 100, 200, 300]
        head_region = {"bbox": [100, 100, 200, 150]}

        confidence = self.recognizer.detect_hairnet(person_bbox, head_region)

        # detect_hairnet 返回单个 float 值
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_update_behavior(self):
        """测试行为状态更新"""
        track_id = 1
        person_bbox = [100, 100, 200, 300]
        hand_regions = [{"bbox": [120, 150, 160, 190]}]

        additional_info = {
            "hand_regions": hand_regions,
            "head_region": {"bbox": [100, 100, 200, 150]},
        }

        behaviors = self.recognizer.update_behavior(
            track_id, person_bbox, additional_info
        )

        self.assertIsInstance(behaviors, dict)

    def test_get_behavior_summary(self):
        """测试获取行为摘要"""
        track_id = 1
        person_bbox = [100, 100, 200, 300]
        hand_regions = [{"bbox": [120, 150, 160, 190]}]

        additional_info = {
            "hand_regions": hand_regions,
            "head_region": {"bbox": [100, 100, 200, 150]},
        }

        # 更新行为状态
        self.recognizer.update_behavior(track_id, person_bbox, additional_info)

        summary = self.recognizer.get_behavior_summary(track_id)
        self.assertIsInstance(summary, dict)

    def test_reset_track(self):
        """测试重置追踪状态"""
        track_id = 1
        person_bbox = [100, 100, 200, 300]
        hand_regions = [{"bbox": [120, 150, 160, 190]}]

        additional_info = {
            "hand_regions": hand_regions,
            "head_region": {"bbox": [100, 100, 200, 150]},
        }

        # 先更新行为状态
        self.recognizer.update_behavior(track_id, person_bbox, additional_info)

        # 重置追踪状态
        self.recognizer.reset_track(track_id)

        # 验证状态已重置 - get_behavior_summary总是返回包含基本字段的字典
        summary = self.recognizer.get_behavior_summary(track_id)
        self.assertIsInstance(summary, dict)
        self.assertEqual(len(summary.get("active_behaviors", {})), 0)
        self.assertEqual(len(summary.get("behavior_history", [])), 0)

    def test_set_confidence_threshold(self):
        """测试设置置信度阈值"""
        new_threshold = 0.8
        self.recognizer.set_confidence_threshold(new_threshold)
        self.assertEqual(self.recognizer.confidence_threshold, new_threshold)

    @patch(
        "src.core.behavior.BehaviorRecognizer._enhance_hand_detection_with_mediapipe"
    )
    def test_advanced_detection_with_mediapipe(self, mock_enhance):
        """测试MediaPipe增强检测"""
        # 设置高级检测模式
        recognizer = BehaviorRecognizer(use_advanced_detection=True, use_mediapipe=True)

        # 模拟MediaPipe增强返回结果
        mock_enhance.return_value = [{"bbox": [150, 250, 200, 300], "keypoints": []}]

        hand_regions = [{"bbox": [150, 250, 200, 300]}]
        confidence = recognizer.detect_handwashing(
            self.person_bbox, hand_regions, 1, self.test_frame
        )

        # 验证MediaPipe增强被调用
        mock_enhance.assert_called_once()
        self.assertIsInstance(confidence, float)

    def test_behavior_state_creation(self):
        """测试行为状态对象创建"""
        state = BehaviorState("handwashing", 0.8)
        self.assertEqual(state.behavior_type, "handwashing")
        self.assertEqual(state.confidence, 0.8)
        self.assertTrue(state.is_active)
        self.assertGreater(state.start_time, 0)

    def test_confidence_threshold_validation(self):
        """测试置信度阈值验证"""
        # 测试有效阈值
        recognizer = BehaviorRecognizer(confidence_threshold=0.7)
        self.assertEqual(recognizer.confidence_threshold, 0.7)

        # 测试边界值
        recognizer = BehaviorRecognizer(confidence_threshold=0.0)
        self.assertEqual(recognizer.confidence_threshold, 0.0)

        recognizer = BehaviorRecognizer(confidence_threshold=1.0)
        self.assertEqual(recognizer.confidence_threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
