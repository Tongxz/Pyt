#!/usr/bin/env python3
"""
运动分析器单元测试

测试 MotionAnalyzer 和 MotionTracker 类的功能
"""

import unittest
from unittest.mock import Mock

import numpy as np

from src.core.motion_analyzer import MotionAnalyzer, MotionTracker


class TestMotionTracker(unittest.TestCase):
    """测试 MotionTracker 类"""

    def setUp(self):
        """设置测试环境"""
        self.tracker = MotionTracker(max_history=10)

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.tracker.max_history, 10)
        self.assertEqual(len(self.tracker.position_history), 0)
        self.assertEqual(len(self.tracker.velocity_history), 0)

    def test_add_position(self):
        """测试添加位置点"""
        position = (100, 200)
        timestamp = 1.0

        self.tracker.update(position, timestamp)

        self.assertEqual(len(self.tracker.position_history), 1)
        self.assertEqual(self.tracker.position_history[0][0], position)
        self.assertEqual(self.tracker.position_history[0][1], timestamp)

    def test_max_history_limit(self):
        """测试历史记录长度限制"""
        # 添加超过最大历史记录数量的位置
        for i in range(15):
            self.tracker.update((i, i), float(i))

        # 检查是否保持在最大历史记录限制内
        self.assertEqual(len(self.tracker.position_history), 10)

        # 检查是否保留了最新的记录
        self.assertEqual(self.tracker.position_history[-1][0], (14, 14))
        self.assertEqual(self.tracker.position_history[-1][1], 14.0)

    def test_get_average_speed_empty(self):
        """测试空轨迹的平均速度"""
        stats = self.tracker.get_motion_stats()
        self.assertEqual(stats["avg_speed"], 0.0)

    def test_get_average_speed_single_point(self):
        """测试单点轨迹的平均速度"""
        self.tracker.update((100, 200), 1.0)
        stats = self.tracker.get_motion_stats()
        self.assertEqual(stats["avg_speed"], 0.0)

    def test_get_average_speed_multiple_points(self):
        """测试多点轨迹的平均速度"""
        # 添加直线运动轨迹
        self.tracker.update((0, 0), 0.0)
        self.tracker.update((10, 0), 1.0)  # 10像素/秒
        self.tracker.update((20, 0), 2.0)  # 10像素/秒

        stats = self.tracker.get_motion_stats()
        self.assertGreater(stats["avg_speed"], 0.0)

    def test_get_motion_stats_empty(self):
        """测试空轨迹的运动统计"""
        stats = self.tracker.get_motion_stats()
        expected = {
            "avg_speed": 0.0,
            "horizontal_movement": 0.0,
            "vertical_movement": 0.0,
            "movement_ratio": 0.0,
            "position_variance": 0.0,
            "horizontal_move_std": 0.0,
            "vertical_move_std": 0.0,
            "move_frequency_hz": 0.0,
        }
        self.assertEqual(stats, expected)

    def test_get_motion_stats_horizontal_motion(self):
        """测试水平运动统计"""
        # 添加水平运动轨迹
        self.tracker.update((0, 100), 0.0)
        self.tracker.update((50, 100), 1.0)
        self.tracker.update((100, 100), 2.0)

        stats = self.tracker.get_motion_stats()

        self.assertGreater(stats["horizontal_movement"], 0)
        self.assertEqual(stats["vertical_movement"], 0)
        self.assertGreater(stats["movement_ratio"], 0)
        self.assertGreater(stats["position_variance"], 0)

    def test_get_motion_stats_vertical_motion(self):
        """测试垂直运动统计"""
        # 添加垂直运动轨迹
        self.tracker.update((100, 0), 0.0)
        self.tracker.update((100, 50), 1.0)
        self.tracker.update((100, 100), 2.0)

        stats = self.tracker.get_motion_stats()

        self.assertEqual(stats["horizontal_movement"], 0)
        self.assertGreater(stats["vertical_movement"], 0)
        self.assertEqual(stats["movement_ratio"], 0.0)  # 纯垂直运动时比例为0
        self.assertGreater(stats["position_variance"], 0)

    def test_clear(self):
        """测试清空轨迹"""
        # 添加一些位置
        self.tracker.update((100, 200), 1.0)
        self.tracker.update((150, 250), 2.0)

        # 清空
        self.tracker.clear()

        # 检查是否清空
        self.assertEqual(len(self.tracker.position_history), 0)
        self.assertEqual(len(self.tracker.velocity_history), 0)


class TestMotionAnalyzer(unittest.TestCase):
    """测试 MotionAnalyzer 类"""

    def setUp(self):
        """设置测试环境"""
        self.analyzer = MotionAnalyzer()
        self.track_id = 1

    def test_update_hand_motion_with_keypoints(self):
        """测试使用关键点更新手部运动"""
        hand_regions = [
            {
                "bbox": [100, 100, 150, 150],
                "keypoints": [(120, 120), (130, 130)],  # 手部关键点
            }
        ]

        self.analyzer.update_hand_motion(self.track_id, hand_regions)

        # 检查是否创建了追踪器
        self.assertIn(self.track_id, self.analyzer.hand_trackers)
        self.assertEqual(
            len(self.analyzer.hand_trackers[self.track_id]), 3
        )  # left, right, unknown

    def test_update_hand_motion_with_bbox_only(self):
        """测试仅使用边界框更新手部运动"""
        hand_regions = [{"bbox": [100, 100, 150, 150]}]  # 无关键点

        self.analyzer.update_hand_motion(self.track_id, hand_regions)

        # 检查是否创建了追踪器
        self.assertIn(self.track_id, self.analyzer.hand_trackers)
        self.assertEqual(
            len(self.analyzer.hand_trackers[self.track_id]), 3
        )  # left, right, unknown

    def test_update_hand_motion_multiple_hands(self):
        """测试多手部运动更新"""
        hand_regions = [
            {"bbox": [100, 100, 150, 150]},  # 左手
            {"bbox": [200, 100, 250, 150]},  # 右手
        ]

        self.analyzer.update_hand_motion(self.track_id, hand_regions)

        # 检查是否为两只手创建了追踪器
        self.assertIn(self.track_id, self.analyzer.hand_trackers)
        self.assertEqual(
            len(self.analyzer.hand_trackers[self.track_id]), 3
        )  # left, right, unknown

    def test_analyze_handwashing_no_motion_data(self):
        """测试无运动数据的洗手分析"""
        confidence = self.analyzer.analyze_handwashing(self.track_id)
        self.assertEqual(confidence, 0.0)

    def test_analyze_handwashing_insufficient_motion(self):
        """测试运动数据不足的洗手分析"""
        # 添加少量运动数据
        hand_regions = [{"bbox": [100, 100, 150, 150]}]
        self.analyzer.update_hand_motion(self.track_id, hand_regions)

        confidence = self.analyzer.analyze_handwashing(self.track_id)
        self.assertEqual(confidence, 0.0)

    def test_analyze_handwashing_with_motion(self):
        """测试有运动数据的洗手分析"""
        hand_regions = [{"bbox": [100, 100, 150, 150]}]

        # 添加足够的运动数据
        for i in range(10):
            # 模拟洗手运动模式
            hand_regions[0]["bbox"] = [100 + i * 2, 100 + i, 150 + i * 2, 150 + i]
            self.analyzer.update_hand_motion(self.track_id, hand_regions)

        confidence = self.analyzer.analyze_handwashing(self.track_id)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_analyze_sanitizing_single_hand(self):
        """测试单手消毒分析"""
        hand_regions = [{"bbox": [100, 100, 150, 150]}]

        # 添加运动数据
        for i in range(10):
            self.analyzer.update_hand_motion(self.track_id, hand_regions)

        confidence = self.analyzer.analyze_sanitizing(self.track_id)
        self.assertEqual(confidence, 0.0)  # 单手不能消毒

    def test_analyze_sanitizing_two_hands(self):
        """测试双手消毒分析"""
        hand_regions = [
            {"bbox": [100, 100, 150, 150]},  # 左手
            {"bbox": [160, 100, 210, 150]},  # 右手（靠近）
        ]

        # 添加运动数据
        for i in range(10):
            # 模拟消毒运动模式
            hand_regions[0]["bbox"] = [100 + i, 100, 150 + i, 150]
            hand_regions[1]["bbox"] = [160 - i, 100, 210 - i, 150]
            self.analyzer.update_hand_motion(self.track_id, hand_regions)

        confidence = self.analyzer.analyze_sanitizing(self.track_id)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_reset_track(self):
        """测试重置追踪数据"""
        # 添加运动数据
        hand_regions = [{"bbox": [100, 100, 150, 150]}]
        self.analyzer.update_hand_motion(self.track_id, hand_regions)

        # 重置追踪数据
        self.analyzer.reset_track(self.track_id)

        # 检查是否重置
        self.assertNotIn(self.track_id, self.analyzer.hand_trackers)

    def test_get_motion_summary(self):
        """测试获取运动摘要"""
        # 添加运动数据
        hand_regions = [{"bbox": [100, 100, 150, 150]}]
        for i in range(5):
            self.analyzer.update_hand_motion(self.track_id, hand_regions)

        summary = self.analyzer.get_motion_summary(self.track_id)

        self.assertIsInstance(summary, dict)
        self.assertIn("track_id", summary)
        self.assertIn("hands", summary)

    def test_get_motion_summary_no_data(self):
        """测试无数据时的运动摘要"""
        summary = self.analyzer.get_motion_summary(999)  # 不存在的track_id

        expected = {"track_id": 999, "hands": {}}
        self.assertEqual(summary, expected)

    def test_hand_distance_calculation(self):
        """测试手部距离计算"""
        hand_regions = [
            {"bbox": [100, 100, 150, 150]},  # 左手
            {"bbox": [200, 100, 250, 150]},  # 右手
        ]

        # 计算手部中心距离
        left_center = (125, 125)
        right_center = (225, 125)
        expected_distance = 100.0  # 水平距离

        # 通过分析消毒行为间接测试距离计算
        for i in range(10):
            self.analyzer.update_hand_motion(self.track_id, hand_regions)

        confidence = self.analyzer.analyze_sanitizing(self.track_id)
        # 距离过远应该导致较低的置信度
        self.assertLessEqual(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
