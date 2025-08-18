#!/usr/bin/env python3
"""
洗手行为检测测试脚本

测试新实现的洗手和消毒行为检测功能
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging

from src.core.behavior import BehaviorRecognizer
from src.core.motion_analyzer import MotionAnalyzer
from src.core.pose_detector import PoseDetector
from src.utils.data_collector import DataCollector

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_frame(width=640, height=480):
    """
    创建测试帧
    """
    # 创建一个简单的测试图像
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 添加一些背景色
    frame[:, :] = (50, 50, 50)

    # 模拟一个人的轮廓
    cv2.rectangle(frame, (200, 100), (400, 400), (100, 100, 100), -1)

    # 模拟手部区域
    cv2.circle(frame, (250, 250), 30, (150, 150, 150), -1)  # 左手
    cv2.circle(frame, (350, 250), 30, (150, 150, 150), -1)  # 右手

    return frame


def test_pose_detector():
    """
    测试姿态检测器
    """
    logger.info("测试姿态检测器...")

    try:
        pose_detector = PoseDetector()
        test_frame = create_test_frame()

        # 测试姿态检测
        pose_data = pose_detector.detect_pose(test_frame)
        logger.info(f"姿态检测结果: {pose_data is not None}")

        # 测试手部检测
        hands_data = pose_detector.detect_hands(test_frame)
        logger.info(f"手部检测结果: {len(hands_data) if hands_data else 0} 只手")

        # 清理资源
        pose_detector.cleanup()

        return True

    except Exception as e:
        logger.error(f"姿态检测器测试失败: {e}")
        return False


def test_motion_analyzer():
    """
    测试运动分析器
    """
    logger.info("测试运动分析器...")

    try:
        motion_analyzer = MotionAnalyzer()

        # 模拟手部区域数据
        hand_regions = [
            {"bbox": [240, 240, 280, 280]},  # 左手
            {"bbox": [340, 240, 380, 280]},  # 右手
        ]

        track_id = 1

        # 模拟多帧数据
        for i in range(10):
            # 模拟手部运动
            for j, region in enumerate(hand_regions):
                # 添加一些随机运动
                region["bbox"][0] += np.random.randint(-5, 6)
                region["bbox"][1] += np.random.randint(-3, 4)
                region["bbox"][2] += np.random.randint(-5, 6)
                region["bbox"][3] += np.random.randint(-3, 4)

            motion_analyzer.update_hand_motion(track_id, hand_regions)

        # 测试洗手分析
        handwash_confidence = motion_analyzer.analyze_handwashing(track_id)
        logger.info(f"洗手行为置信度: {handwash_confidence:.3f}")

        # 测试消毒分析
        sanitize_confidence = motion_analyzer.analyze_sanitizing(track_id)
        logger.info(f"消毒行为置信度: {sanitize_confidence:.3f}")

        return True

    except Exception as e:
        logger.error(f"运动分析器测试失败: {e}")
        return False


def test_behavior_recognizer():
    """
    测试行为识别器
    """
    logger.info("测试行为识别器...")

    try:
        # 测试基础检测
        basic_recognizer = BehaviorRecognizer(use_advanced_detection=False)

        person_bbox = [200, 100, 400, 400]
        hand_regions = [
            {"bbox": [240, 240, 280, 280]},  # 左手
            {"bbox": [340, 240, 380, 280]},  # 右手
        ]

        # 测试基础洗手检测
        basic_handwash = basic_recognizer.detect_handwashing(person_bbox, hand_regions)
        logger.info(f"基础洗手检测置信度: {basic_handwash:.3f}")

        # 测试基础消毒检测
        basic_sanitize = basic_recognizer.detect_sanitizing(person_bbox, hand_regions)
        logger.info(f"基础消毒检测置信度: {basic_sanitize:.3f}")

        # 测试高级检测
        advanced_recognizer = BehaviorRecognizer(use_advanced_detection=True)
        test_frame = create_test_frame()
        track_id = 1

        # 测试高级洗手检测
        advanced_handwash = advanced_recognizer.detect_handwashing(
            person_bbox, hand_regions, track_id, test_frame
        )
        logger.info(f"高级洗手检测置信度: {advanced_handwash:.3f}")

        # 测试高级消毒检测
        advanced_sanitize = advanced_recognizer.detect_sanitizing(
            person_bbox, hand_regions, track_id, test_frame
        )
        logger.info(f"高级消毒检测置信度: {advanced_sanitize:.3f}")

        # 清理资源
        if advanced_recognizer.pose_detector:
            advanced_recognizer.pose_detector.cleanup()

        return True

    except Exception as e:
        logger.error(f"行为识别器测试失败: {e}")
        return False


def test_data_collector():
    """
    测试数据收集器
    """
    logger.info("测试数据收集器...")

    try:
        data_collector = DataCollector()

        # 模拟行为检测结果
        test_frame = create_test_frame()
        track_id = 1

        # 模拟多帧洗手行为检测
        for i in range(10):
            behavior_results = {
                "handwashing": 0.8,  # 高置信度洗手行为
                "sanitizing": 0.2,  # 低置信度消毒行为
                "other": 0.1,  # 其他行为
            }

            frame_metadata = {
                "frame_id": i,
                "timestamp": datetime.now().isoformat(),
                "person_bbox": [200, 100, 400, 400],
                "hand_regions": [
                    {"bbox": [240, 240, 280, 280]},
                    {"bbox": [340, 240, 380, 280]},
                ],
            }

            data_collector.update_detection(
                track_id, test_frame, behavior_results, frame_metadata
            )

            # 模拟时间间隔
            import time

            time.sleep(0.1)

        # 等待一段时间让保存线程处理
        time.sleep(1.0)

        # 获取统计信息
        stats = data_collector.get_stats()
        logger.info(f"数据收集统计: {stats}")

        # 清理资源
        data_collector.cleanup()

        return True

    except Exception as e:
        logger.error(f"数据收集器测试失败: {e}")
        return False


def main():
    """
    主测试函数
    """
    logger.info("开始洗手行为检测功能测试")

    tests = [
        ("姿态检测器", test_pose_detector),
        ("运动分析器", test_motion_analyzer),
        ("行为识别器", test_behavior_recognizer),
        ("数据收集器", test_data_collector),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{test_name} 测试{'成功' if result else '失败'}")
        except Exception as e:
            logger.error(f"{test_name} 测试异常: {e}")
            results.append((test_name, False))

    # 输出测试总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        logger.info("🎉 所有测试通过！洗手行为检测功能已就绪。")
    else:
        logger.warning(f"⚠️  {total - passed} 个测试失败，请检查相关功能。")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
