#!/usr/bin/env python3
"""
测试 MediaPipe 集成的洗手检测功能
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.behavior import BehaviorRecognizer
from src.core.optimized_detection_pipeline import DetectionResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image_with_hands():
    """
    创建一个包含手部的测试图像
    """
    # 创建一个简单的测试图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 绘制一个简单的人形轮廓
    # 头部
    cv2.circle(img, (320, 100), 40, (200, 200, 200), -1)

    # 身体
    cv2.rectangle(img, (280, 140), (360, 300), (150, 150, 150), -1)

    # 手臂和手部区域（模拟洗手姿势）
    # 左手
    cv2.ellipse(img, (250, 220), (30, 50), 0, 0, 360, (180, 150, 120), -1)
    # 右手
    cv2.ellipse(img, (390, 220), (30, 50), 0, 0, 360, (180, 150, 120), -1)

    return img


def test_mediapipe_integration():
    """
    测试 MediaPipe 集成功能
    """
    logger.info("开始测试 MediaPipe 集成的洗手检测功能")

    # 初始化行为识别器（启用 MediaPipe）
    behavior_recognizer = BehaviorRecognizer(
        confidence_threshold=0.3, use_advanced_detection=False, use_mediapipe=True
    )

    # 创建测试图像
    test_image = create_test_image_with_hands()

    # 模拟人体检测结果
    person_bbox = [250, 80, 390, 320]  # [x1, y1, x2, y2]

    # 模拟手部检测结果
    hand_regions = [
        {"bbox": [220, 195, 280, 245], "confidence": 0.7, "source": "yolo"},
        {"bbox": [360, 195, 420, 245], "confidence": 0.8, "source": "yolo"},
    ]

    logger.info(f"原始手部检测结果: {len(hand_regions)} 只手")

    # 测试洗手检测（包含 MediaPipe 增强）
    handwashing_confidence = behavior_recognizer.detect_handwashing(
        person_bbox=person_bbox, hand_regions=hand_regions, track_id=1, frame=test_image
    )

    logger.info(f"洗手检测置信度: {handwashing_confidence:.3f}")

    # 测试 MediaPipe 手部增强功能
    enhanced_regions = behavior_recognizer._enhance_hand_detection_with_mediapipe(
        test_image, hand_regions
    )

    logger.info(f"MediaPipe 增强后的手部检测结果: {len(enhanced_regions)} 只手")

    for i, region in enumerate(enhanced_regions):
        logger.info(
            f"手部 {i+1}: bbox={region.get('bbox')}, "
            f"confidence={region.get('confidence'):.3f}, "
            f"source={region.get('source')}, "
            f"landmarks_count={len(region.get('landmarks', []))}"
        )

    # 保存测试图像
    output_path = "test_mediapipe_integration.png"
    cv2.imwrite(output_path, test_image)
    logger.info(f"测试图像已保存到: {output_path}")

    # 如果有 MediaPipe 检测结果，绘制关键点
    if enhanced_regions:
        annotated_image = test_image.copy()

        for region in enhanced_regions:
            if "landmarks" in region and region["landmarks"]:
                landmarks = region["landmarks"]
                h, w = annotated_image.shape[:2]

                # 绘制手部关键点
                for landmark in landmarks:
                    x = int(landmark["x"] * w)
                    y = int(landmark["y"] * h)
                    cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)

                # 绘制边界框
                bbox = region["bbox"]
                cv2.rectangle(
                    annotated_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2,
                )

                # 添加标签
                label = f"MediaPipe Hand ({region.get('confidence', 0):.2f})"
                cv2.putText(
                    annotated_image,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # 保存标注图像
        annotated_path = "test_mediapipe_annotated.png"
        cv2.imwrite(annotated_path, annotated_image)
        logger.info(f"标注图像已保存到: {annotated_path}")

    return handwashing_confidence, enhanced_regions


def test_with_real_image():
    """
    使用 tests/fixtures 目录下的真实图像测试
    """
    fixtures_dir = Path("tests/fixtures")

    # 搜索所有可能的图像文件
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    test_images = []

    for ext in image_extensions:
        test_images.extend(fixtures_dir.rglob(ext))

    if not test_images:
        logger.warning("在 tests/fixtures 目录下未找到任何图像文件")
        return

    logger.info(f"在 tests/fixtures 目录下找到 {len(test_images)} 个图像文件")

    for image_path in test_images[:3]:  # 限制测试前3个图像
        logger.info(f"测试真实图像: {image_path}")

        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"无法读取图像: {image_path}")
            continue

        # 初始化行为识别器
        behavior_recognizer = BehaviorRecognizer(
            confidence_threshold=0.3, use_mediapipe=True
        )

        # 模拟检测结果（实际应用中这些来自目标检测模型）
        h, w = image.shape[:2]
        person_bbox = [w // 4, h // 4, 3 * w // 4, 3 * h // 4]
        hand_regions = [
            {
                "bbox": [w // 3, h // 2, 2 * w // 3, 2 * h // 3],
                "confidence": 0.6,
                "source": "simulated",
            }
        ]

        # 测试洗手检测
        confidence = behavior_recognizer.detect_handwashing(
            person_bbox=person_bbox, hand_regions=hand_regions, frame=image
        )

        logger.info(f"{image_path.name} 洗手检测置信度: {confidence:.3f}")

        # 测试 MediaPipe 增强功能
        enhanced_regions = behavior_recognizer._enhance_hand_detection_with_mediapipe(
            image, hand_regions
        )
        logger.info(f"{image_path.name} MediaPipe 增强后检测到 {len(enhanced_regions)} 只手")


if __name__ == "__main__":
    try:
        # 测试基本功能
        confidence, regions = test_mediapipe_integration()

        # 测试真实图像
        test_with_real_image()

        logger.info("\n=== 测试总结 ===")
        logger.info(f"MediaPipe 集成测试完成")
        logger.info(f"洗手检测置信度: {confidence:.3f}")
        logger.info(f"增强后手部检测数量: {len(regions)}")

        if confidence > 0.3:
            logger.info("✅ 洗手行为检测成功")
        else:
            logger.info("❌ 洗手行为检测置信度较低")

    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
