#!/usr/bin/env python3
"""YOLO模型对比测试脚本.

比较models/yolo/yolov8n.pt和models/yolo/yolov8m.pt的人体检测效果.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.detector import HumanDetector
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__, level="INFO")


def test_model_comparison():
    """对比不同YOLO模型的检测效果."""
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
    print("=" * 80)

    # 测试模型列表
    models_to_test = [
        {
            "name": "YOLOv8n (Nano)",
            "path": "models/yolo/yolov8n.pt",
            "description": "最小模型，速度最快",
        },
        {
            "name": "YOLOv8s (Small)",
            "path": "models/yolo/yolov8s.pt",
            "description": "小型模型，平衡速度和精度",
        },
        {
            "name": "YOLOv8m (Medium)",
            "path": "models/yolo/yolov8m.pt",
            "description": "中型模型，更高精度",
        },
        {
            "name": "YOLOv8l (Large)",
            "path": "models/yolo/yolov8l.pt",
            "description": "大型模型，高精度",
        },
    ]

    results = []

    for model_info in models_to_test:
        print(f"\n=== 测试 {model_info['name']} ===")
        print(f"描述: {model_info['description']}")

        try:
            # 加载模型
            start_time = time.time()
            model = YOLO(model_info["path"])
            load_time = time.time() - start_time
            print(f"模型加载时间: {load_time:.3f}秒")

            # 测试不同置信度阈值
            conf_thresholds = [0.1, 0.25, 0.5]

            for conf in conf_thresholds:
                print(f"\n  置信度阈值: {conf}")

                # 执行检测
                start_time = time.time()
                results_yolo = model(image, conf=conf, iou=0.5, verbose=False)
                inference_time = time.time() - start_time

                # 统计人员检测结果
                person_count = 0
                high_conf_count = 0  # 高置信度检测数量
                total_confidence = 0

                for result in results_yolo:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            if class_id == 0:  # person class
                                confidence = float(box.conf[0].cpu().numpy())
                                person_count += 1
                                total_confidence += confidence
                                if confidence > 0.7:
                                    high_conf_count += 1

                avg_confidence = (
                    total_confidence / person_count if person_count > 0 else 0
                )

                print(f"    检测到人员: {person_count}")
                print(f"    高置信度(>0.7): {high_conf_count}")
                print(f"    平均置信度: {avg_confidence:.3f}")
                print(f"    推理时间: {inference_time:.3f}秒")

                # 保存结果
                results.append(
                    {
                        "model": model_info["name"],
                        "conf_threshold": conf,
                        "person_count": person_count,
                        "high_conf_count": high_conf_count,
                        "avg_confidence": avg_confidence,
                        "inference_time": inference_time,
                        "load_time": load_time,
                    }
                )

        except Exception as e:
            print(f"模型 {model_info['name']} 测试失败: {e}")
            continue

    # 生成对比报告
    print("\n" + "=" * 80)
    print("模型对比报告")
    print("=" * 80)

    # 按置信度阈值分组显示结果
    for conf in [0.1, 0.25, 0.5]:
        print(f"\n置信度阈值 {conf} 的结果对比:")
        print("-" * 60)
        print(f"{'模型':<15} {'人员数':<8} {'高置信度':<10} {'平均置信度':<12} {'推理时间':<10}")
        print("-" * 60)

        for result in results:
            if result["conf_threshold"] == conf:
                print(
                    f"{result['model']:<15} {result['person_count']:<8} {result['high_conf_count']:<10} "
                    f"{result['avg_confidence']:<12.3f} {result['inference_time']:<10.3f}"
                )

    # 推荐最佳模型
    print("\n" + "=" * 80)
    print("推荐分析")
    print("=" * 80)

    # 分析0.25置信度阈值下的结果（常用阈值）
    conf_025_results = [r for r in results if r["conf_threshold"] == 0.25]

    if conf_025_results:
        # 按检测人数排序
        by_detection = sorted(
            conf_025_results, key=lambda x: x["person_count"], reverse=True
        )
        # 按平均置信度排序
        by_confidence = sorted(
            conf_025_results, key=lambda x: x["avg_confidence"], reverse=True
        )
        # 按速度排序
        by_speed = sorted(conf_025_results, key=lambda x: x["inference_time"])

        print(
            f"检测人数最多: {by_detection[0]['model']} ({by_detection[0]['person_count']}人)"
        )
        print(
            f"平均置信度最高: {by_confidence[0]['model']} ({by_confidence[0]['avg_confidence']:.3f})"
        )
        print(f"推理速度最快: {by_speed[0]['model']} ({by_speed[0]['inference_time']:.3f}秒)")

        # 综合评分（检测数量 * 0.4 + 置信度 * 0.4 + 速度评分 * 0.2）
        print("\n综合评分排名:")
        for result in conf_025_results:
            detection_score = (
                result["person_count"]
                / max([r["person_count"] for r in conf_025_results])
                * 0.4
            )
            confidence_score = result["avg_confidence"] * 0.4
            speed_score = (
                (1 / result["inference_time"])
                / max([1 / r["inference_time"] for r in conf_025_results])
                * 0.2
            )
            total_score = detection_score + confidence_score + speed_score
            print(f"  {result['model']}: {total_score:.3f}分")


def test_custom_detector_with_different_models():
    """测试自定义检测器使用不同模型的效果."""
    print("\n" + "=" * 80)
    print("自定义检测器模型对比")
    print("=" * 80)

    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"
    image = cv2.imread(test_image_path)

    if image is None:
        print(f"无法读取图像: {test_image_path}")
        return

    models_to_test = [
        "models/yolo/yolov8n.pt",
        "models/yolo/yolov8s.pt",
        "models/yolo/yolov8m.pt",
    ]

    for model_path in models_to_test:
        print(f"\n=== 测试自定义检测器 + {model_path} ===")

        try:
            # 创建检测器
            detector = HumanDetector(model_path=model_path)

            # 执行检测
            start_time = time.time()
            detections = detector.detect(image)
            detection_time = time.time() - start_time

            print(f"检测结果: {len(detections)}人")
            print(f"检测时间: {detection_time:.3f}秒")

            # 显示详细信息
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                conf = det["confidence"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                print(f"  人员{i+1}: 置信度={conf:.3f}, 尺寸={width:.0f}x{height:.0f}")

            # 创建可视化结果
            vis_image = detector.visualize_detections(image, detections)
            output_path = f"detection_result_{model_path.replace('.pt', '')}.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f"可视化结果保存到: {output_path}")

        except Exception as e:
            print(f"测试失败: {e}")


if __name__ == "__main__":
    print("YOLO模型对比测试")
    print("=" * 80)

    # 原始YOLO模型对比
    test_model_comparison()

    # 自定义检测器对比
    test_custom_detector_with_different_models()

    print("\n测试完成！")
