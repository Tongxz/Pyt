#!/usr/bin/env python3
"""
人体检测器调试脚本
用于分析人体检测不准确的问题
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.detector import HumanDetector
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__, level="DEBUG")


def test_human_detection_detailed():
    """详细测试人体检测器"""

    # 初始化检测器
    detector = HumanDetector()

    # 测试图像路径
    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"

    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return

    # 读取图像
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法读取图像: {test_image_path}")
        return

    print(f"图像尺寸: {image.shape}")

    # 测试不同的检测参数
    test_configs = [
        {"conf": 0.1, "min_area": 500, "max_ratio": 6.0, "name": "当前配置"},
        {"conf": 0.05, "min_area": 300, "max_ratio": 8.0, "name": "宽松配置"},
        {"conf": 0.25, "min_area": 1000, "max_ratio": 4.0, "name": "严格配置"},
        {"conf": 0.01, "min_area": 100, "max_ratio": 10.0, "name": "极宽松配置"},
    ]

    for config in test_configs:
        print(f"\n=== 测试 {config['name']} ===")
        print(
            f"置信度阈值: {config['conf']}, 最小面积: {config['min_area']}, 最大宽高比: {config['max_ratio']}"
        )

        # 更新检测器参数
        detector.confidence_threshold = config["conf"]
        detector.min_box_area = config["min_area"]
        detector.max_box_ratio = config["max_ratio"]

        try:
            # 执行检测
            detections = detector.detect(image)
            print(f"检测结果: 发现 {len(detections)} 个人")

            # 显示详细信息
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                conf = det["confidence"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                ratio = max(width, height) / min(width, height)
                print(
                    f"  人员 {i+1}: 位置=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), "
                    f"置信度={conf:.3f}, 尺寸={width:.0f}x{height:.0f}, 面积={area:.0f}, 宽高比={ratio:.2f}"
                )

        except Exception as e:
            print(f"检测失败: {e}")


def test_raw_yolo_output():
    """测试原始YOLO输出，不经过过滤"""
    print("\n=== 原始YOLO输出测试 ===")

    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolov8n.pt")

    # 读取图像
    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"
    image = cv2.imread(test_image_path)

    if image is None:
        print(f"无法读取图像: {test_image_path}")
        return

    # 使用不同置信度阈值测试
    conf_thresholds = [0.01, 0.05, 0.1, 0.25, 0.5]

    for conf in conf_thresholds:
        print(f"\n置信度阈值: {conf}")
        results = model(image, conf=conf, iou=0.5)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"  原始检测框数量: {len(boxes)}")

                person_count = 0
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id == 0:  # person class
                        person_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        print(
                            f"    人员: 位置=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), "
                            f"置信度={confidence:.3f}, 尺寸={width:.0f}x{height:.0f}, 面积={area:.0f}"
                        )

                print(f"  检测到的人员数量: {person_count}")
            else:
                print("  未检测到任何目标")


def create_visualization():
    """创建可视化结果"""
    print("\n=== 创建可视化结果 ===")

    detector = HumanDetector()

    # 读取图像
    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"
    image = cv2.imread(test_image_path)

    if image is None:
        print(f"无法读取图像: {test_image_path}")
        return

    # 执行检测
    detections = detector.detect(image)

    # 创建可视化图像
    vis_image = detector.visualize_detections(image, detections)

    # 保存结果
    output_path = "human_detection_debug.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"可视化结果已保存到: {output_path}")


if __name__ == "__main__":
    print("人体检测器调试分析")
    print("=" * 50)

    # 测试详细检测
    test_human_detection_detailed()

    # 测试原始YOLO输出
    test_raw_yolo_output()

    # 创建可视化
    create_visualization()

    print("\n调试分析完成")
