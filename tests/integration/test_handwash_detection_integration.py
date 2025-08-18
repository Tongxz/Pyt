#!/usr/bin/env python3
"""
洗手行为检测集成测试

测试洗手行为检测的完整流程
"""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.core.behavior import BehaviorRecognizer
from src.core.detector import HumanDetector
from src.core.motion_analyzer import MotionAnalyzer
from src.core.pose_detector import PoseDetector
from src.utils.data_collector import DataCollector


class TestHandwashDetectionIntegration(unittest.TestCase):
    """洗手行为检测集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = self.create_test_image()

        # 初始化检测器
        self.human_detector = HumanDetector()
        self.pose_detector = PoseDetector(use_mediapipe=False)
        self.behavior_recognizer = BehaviorRecognizer(
            confidence_threshold=0.5, use_advanced_detection=False
        )
        self.motion_analyzer = MotionAnalyzer()

    def tearDown(self):
        """清理测试环境"""
        # 清理临时目录
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_image(self):
        """创建测试图像"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            # 绘制人体轮廓
            cv2.rectangle(image, (250, 100), (390, 400), (128, 128, 128), -1)

            # 绘制头部
            cv2.circle(image, (320, 130), 30, (200, 200, 200), -1)

            # 绘制手部区域（洗手位置）
            cv2.rectangle(image, (280, 250), (320, 290), (255, 255, 255), -1)  # 左手
            cv2.rectangle(image, (320, 250), (360, 290), (255, 255, 255), -1)  # 右手

        except NameError:
            # 如果cv2不可用，创建简单的测试图像
            image[100:400, 250:390] = [128, 128, 128]  # 人体
            image[100:160, 290:350] = [200, 200, 200]  # 头部
            image[250:290, 280:360] = [255, 255, 255]  # 手部

        return image

    def test_complete_handwash_detection_pipeline(self):
        """测试完整的洗手检测流程"""
        # 1. 人体检测
        persons = self.human_detector.detect_humans(self.test_image)
        self.assertGreater(len(persons), 0, "应该检测到至少一个人")

        person = persons[0]
        person_bbox = person["bbox"]

        # 2. 姿态检测
        keypoints = self.pose_detector.detect_pose(self.test_image)
        self.assertIsInstance(keypoints, list)

        # 3. 手部检测
        hand_regions = self.pose_detector.detect_hands(self.test_image)
        self.assertIsInstance(hand_regions, list)

        # 如果没有检测到手部，创建模拟手部区域
        if not hand_regions:
            hand_regions = [
                {
                    "bbox": [280, 250, 320, 290],
                    "confidence": 0.8,
                    "source": "simulated",
                },
                {
                    "bbox": [320, 250, 360, 290],
                    "confidence": 0.8,
                    "source": "simulated",
                },
            ]

        # 4. 洗手行为检测
        handwash_confidence = self.behavior_recognizer.detect_handwashing(
            person_bbox, hand_regions, frame=self.test_image
        )

        self.assertIsInstance(handwash_confidence, float)
        self.assertGreaterEqual(handwash_confidence, 0.0)
        self.assertLessEqual(handwash_confidence, 1.0)

        # 5. 行为状态更新
        track_id = 1
        self.behavior_recognizer.update_behavior(
            track_id, person_bbox, hand_regions, self.test_image
        )

        # 6. 获取行为摘要
        summary = self.behavior_recognizer.get_behavior_summary(track_id)
        self.assertIsInstance(summary, dict)
        self.assertIn("handwashing", summary)

    def test_advanced_detection_pipeline(self):
        """测试高级检测流程"""
        # 使用高级检测模式
        advanced_recognizer = BehaviorRecognizer(
            confidence_threshold=0.5, use_advanced_detection=True
        )

        # 模拟人体和手部检测结果
        person_bbox = [250, 100, 390, 400]
        hand_regions = [
            {"bbox": [280, 250, 320, 290], "confidence": 0.8},
            {"bbox": [320, 250, 360, 290], "confidence": 0.8},
        ]

        track_id = 1

        # 执行高级洗手检测
        confidence = advanced_recognizer.detect_handwashing(
            person_bbox, hand_regions, track_id, self.test_image
        )

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_motion_analysis_integration(self):
        """测试运动分析集成"""
        track_id = 1

        # 模拟多帧手部运动数据
        for i in range(10):
            hand_regions = [
                {
                    "bbox": [280 + i * 2, 250 + i, 320 + i * 2, 290 + i],
                    "confidence": 0.8,
                }
            ]

            # 更新运动数据
            self.motion_analyzer.update_hand_motion(track_id, hand_regions)

        # 分析洗手运动
        handwash_confidence = self.motion_analyzer.analyze_handwashing(
            track_id, hand_regions
        )

        self.assertIsInstance(handwash_confidence, float)
        self.assertGreaterEqual(handwash_confidence, 0.0)
        self.assertLessEqual(handwash_confidence, 1.0)

        # 获取运动摘要
        summary = self.motion_analyzer.get_motion_summary(track_id)
        self.assertIsInstance(summary, dict)
        self.assertIn("hand_count", summary)
        self.assertIn("total_motion_points", summary)

    def test_data_collection_integration(self):
        """测试数据收集集成"""
        # 创建数据收集器
        collector = DataCollector(output_dir=self.temp_dir, confidence_threshold=0.6)

        try:
            track_id = 1

            # 模拟行为检测结果
            behavior_results = {"handwashing": 0.8, "sanitizing": 0.3, "other": 0.1}

            frame_metadata = {
                "timestamp": "2024-01-01T12:00:00",
                "person_bbox": [250, 100, 390, 400],
                "hand_regions": [
                    {"bbox": [280, 250, 320, 290]},
                    {"bbox": [320, 250, 360, 290]},
                ],
            }

            # 更新检测结果
            collector.update_detection(
                track_id, self.test_image, behavior_results, frame_metadata
            )

            # 获取统计信息
            stats = collector.get_stats()
            self.assertIsInstance(stats, dict)

        finally:
            collector.cleanup()

    def test_multi_person_detection(self):
        """测试多人检测场景"""
        # 创建包含多个人的测试图像
        multi_person_image = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            # 绘制第一个人
            cv2.rectangle(
                multi_person_image, (150, 100), (250, 400), (128, 128, 128), -1
            )
            cv2.rectangle(
                multi_person_image, (180, 250), (220, 290), (255, 255, 255), -1
            )

            # 绘制第二个人
            cv2.rectangle(
                multi_person_image, (400, 100), (500, 400), (128, 128, 128), -1
            )
            cv2.rectangle(
                multi_person_image, (430, 250), (470, 290), (255, 255, 255), -1
            )

        except NameError:
            # 如果cv2不可用，创建简单的多人图像
            multi_person_image[100:400, 150:250] = [128, 128, 128]  # 第一个人
            multi_person_image[250:290, 180:220] = [255, 255, 255]  # 第一个人的手
            multi_person_image[100:400, 400:500] = [128, 128, 128]  # 第二个人
            multi_person_image[250:290, 430:470] = [255, 255, 255]  # 第二个人的手

        # 检测多个人
        persons = self.human_detector.detect_humans(multi_person_image)

        # 为每个人进行洗手检测
        for i, person in enumerate(persons):
            person_bbox = person["bbox"]

            # 模拟手部检测
            if i == 0:  # 第一个人
                hand_regions = [{"bbox": [180, 250, 220, 290], "confidence": 0.8}]
            else:  # 第二个人
                hand_regions = [{"bbox": [430, 250, 470, 290], "confidence": 0.8}]

            # 洗手检测
            confidence = self.behavior_recognizer.detect_handwashing(
                person_bbox, hand_regions, frame=multi_person_image
            )

            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_error_handling(self):
        """测试错误处理"""
        # 测试空图像
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)

        persons = self.human_detector.detect_humans(empty_image)
        self.assertIsInstance(persons, list)

        # 测试无效的人体边界框
        invalid_bbox = []
        hand_regions = [{"bbox": [50, 50, 100, 100]}]

        confidence = self.behavior_recognizer.detect_handwashing(
            invalid_bbox, hand_regions, frame=self.test_image
        )
        self.assertEqual(confidence, 0.0)

        # 测试无效的手部区域
        valid_bbox = [100, 100, 200, 300]
        invalid_hands = []

        confidence = self.behavior_recognizer.detect_handwashing(
            valid_bbox, invalid_hands, frame=self.test_image
        )
        self.assertEqual(confidence, 0.0)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 测试检测速度
        start_time = time.time()

        for _ in range(10):
            persons = self.human_detector.detect_humans(self.test_image)
            if persons:
                person_bbox = persons[0]["bbox"]
                hand_regions = [{"bbox": [280, 250, 320, 290]}]

                self.behavior_recognizer.detect_handwashing(
                    person_bbox, hand_regions, frame=self.test_image
                )

        end_time = time.time()
        avg_time = (end_time - start_time) / 10

        # 检测时间应该在合理范围内（小于1秒）
        self.assertLess(avg_time, 1.0, "平均检测时间应该小于1秒")

    def test_confidence_threshold_effects(self):
        """测试置信度阈值的影响"""
        person_bbox = [250, 100, 390, 400]
        hand_regions = [{"bbox": [280, 250, 320, 290], "confidence": 0.7}]

        # 测试不同的置信度阈值
        thresholds = [0.3, 0.5, 0.7, 0.9]

        for threshold in thresholds:
            recognizer = BehaviorRecognizer(
                confidence_threshold=threshold, use_advanced_detection=False
            )

            confidence = recognizer.detect_handwashing(
                person_bbox, hand_regions, frame=self.test_image
            )

            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

            # 检查阈值设置是否生效
            self.assertEqual(recognizer.confidence_threshold, threshold)


if __name__ == "__main__":
    unittest.main()
