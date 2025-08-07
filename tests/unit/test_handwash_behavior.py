"""Handwash behavior detection unit tests"""

import unittest

import numpy as np

from src.core.behavior import BehaviorRecognizer


class TestHandwashBehavior(unittest.TestCase):
    """测试 BehaviorRecognizer.detect_handwashing 逻辑"""

    def setUp(self):
        # 将置信度阈值调低，便于触发检测成功
        self.recognizer = BehaviorRecognizer(confidence_threshold=0.5)
        # 构造一个典型的人体 bbox (x1, y1, x2, y2)
        self.person_bbox = [100, 50, 300, 450]  # 宽 200，高 400

    def test_detect_handwashing_positive(self):
        """手部位于人体中下部且 _analyze_hand_motion 返回非零 -> 应检测到洗手"""
        # 手中心在相对 y 比例 0.55 处， 属于 0.4~0.8 范围
        hand_h = 60
        hand_w = 40
        person_x1, person_y1, person_x2, person_y2 = self.person_bbox
        hand_y_center = person_y1 + int(0.55 * (person_y2 - person_y1))
        hand_bbox = [
            person_x1 + 20,
            hand_y_center - hand_h // 2,
            person_x1 + 20 + hand_w,
            hand_y_center + hand_h // 2,
        ]

        confidence = self.recognizer.detect_handwashing(
            self.person_bbox, [{"bbox": hand_bbox}]
        )
        # 默认 _analyze_hand_motion 返回 0.6
        self.assertEqual(confidence, 0.6)
        self.assertGreater(confidence, self.recognizer.confidence_threshold)

    def test_detect_handwashing_negative(self):
        """手部位于人体上部，应该返回 0.0"""
        # 手中心在相对 y 比例 0.2 处， 不在 0.4~0.8 范围
        hand_h = 60
        hand_w = 40
        person_x1, person_y1, person_x2, person_y2 = self.person_bbox
        hand_y_center = person_y1 + int(0.2 * (person_y2 - person_y1))
        hand_bbox = [
            person_x1 + 20,
            hand_y_center - hand_h // 2,
            person_x1 + 20 + hand_w,
            hand_y_center + hand_h // 2,
        ]

        confidence = self.recognizer.detect_handwashing(
            self.person_bbox, [{"bbox": hand_bbox}]
        )
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
