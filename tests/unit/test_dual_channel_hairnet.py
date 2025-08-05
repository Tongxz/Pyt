#!/usr/bin/env python3
"""
双通道策略发网检测测试脚本

测试新增的关键点优化和双通道策略功能
"""

import os
import sys

import cv2
import numpy as np


# 模拟检测器类
class MockPersonDetector:
    def detect(self, frame):
        return [{"bbox": [0, 0, 100, 100], "confidence": 0.9}]


class MockHairnetDetector:
    def _detect_hairnet_with_pytorch(self, frame):
        return {"wearing_hairnet": True, "has_hairnet": True, "confidence": 0.85}


from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.hairnet_detector import HairnetDetectionPipeline


def test_dual_channel_detection():
    """
    测试双通道策略发网检测
    """
    print("=== 双通道策略发网检测测试 ===")

    # 初始化检测管道
    pipeline = HairnetDetectionPipeline(MockPersonDetector(), MockHairnetDetector())

    # 加载测试图像
    test_image_path = "test_images/test_image.jpg"

    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return

    image = cv2.imread(test_image_path)
    if image is None:
        print(f"❌ 无法加载图像: {test_image_path}")
        return

    print(f"✅ 成功加载测试图像: {test_image_path}")
    print(f"图像尺寸: {image.shape}")

    # 执行检测
    print("\n开始执行双通道策略检测...")
    detections = pipeline.detect_hairnet_compliance(image)

    # 显示检测结果
    print(f"\n=== 检测结果 ===")
    print(f"检测到人数: {detections.get('total_persons', 0)}")
    print(f"佩戴发网人数: {detections.get('hairnet_persons', 0)}")
    print(f"合规率: {detections.get('compliance_rate', 0):.2%}")
    print(f"平均置信度: {detections.get('average_confidence', 0):.3f}")

    # 详细分析每个检测结果
    if "detections" in detections:
        print("\n=== 详细分析 ===")
        for i, detection in enumerate(detections["detections"], 1):
            print(f"\n人员 {i}:")
            print(f"  人体位置: {detection.get('bbox', [])}")
            print(f"  头部区域: {detection.get('head_coords', [])}")
            print(f"  ROI策略: {detection.get('roi_strategy', 'unknown')}")
            print(f"  发网检测: {'是' if detection.get('has_hairnet', False) else '否'}")
            print(f"  置信度: {detection.get('confidence', 0):.3f}")

            # 显示调试信息
            debug_info = detection.get("debug_info", {})
            if debug_info:
                print(f"  调试信息:")
                print(f"    基础边缘密度: {debug_info.get('basic_edge_density', 0):.4f}")
                print(f"    敏感边缘密度: {debug_info.get('sensitive_edge_density', 0):.4f}")
                print(f"    最终边缘密度: {debug_info.get('edge_density', 0):.4f}")
                print(f"    轮廓数量: {debug_info.get('contour_count', 0)}")
                print(f"    浅蓝色比例: {debug_info.get('light_blue_ratio', 0):.4f}")
                print(f"    浅色比例: {debug_info.get('light_ratio', 0):.4f}")
                print(f"    上部边缘密度: {debug_info.get('upper_edge_density', 0):.4f}")
                print(f"    综合得分: {debug_info.get('composite_score', 0):.3f}")

    # 显示统计信息
    print("\n=== 算法统计 ===")
    hairnet_detector = pipeline.hairnet_detector
    print(f"关键点成功次数: {hairnet_detector.stats.get('keypoint_success', 0)}")
    print(f"BBox回退次数: {hairnet_detector.stats.get('bbox_fallback', 0)}")
    print(f"总检测次数: {hairnet_detector.total_detections}")
    print(f"发网检测次数: {hairnet_detector.hairnet_detections}")

    # 性能评估
    total_persons = detections.get("total_persons", 0)
    hairnet_persons = detections.get("hairnet_persons", 0)

    if total_persons > 0:
        print("\n=== 算法性能评估 ===")
        coverage_rate = total_persons / total_persons * 100
        detection_rate = (
            hairnet_persons / total_persons * 100 if total_persons > 0 else 0
        )
        avg_confidence = detections.get("average_confidence", 0)

        print(f"检测覆盖率: {total_persons}/{total_persons} = {coverage_rate:.1f}%")
        print(f"发网识别率: {hairnet_persons}/{total_persons} = {detection_rate:.1f}%")
        print(f"平均置信度: {avg_confidence:.3f}")

        if avg_confidence >= 0.8:
            print("✅ 算法置信度高")
        elif avg_confidence >= 0.6:
            print("⚠ 算法置信度中等")
        else:
            print("❌ 算法置信度低")

    print("\n=== 测试完成 ===")


def test_keypoint_detection():
    """
    测试关键点检测功能（模拟）
    """
    print("\n=== 关键点检测功能测试 ===")

    # 初始化检测器
    from src.core.hairnet_detector import HairnetDetector

    detector = HairnetDetector()

    # 创建模拟图像和关键点
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 模拟COCO格式关键点 (x, y, confidence)
    mock_keypoints = np.array(
        [
            [320, 100, 0.9],  # 鼻子
            [310, 90, 0.8],  # 左眼
            [330, 90, 0.8],  # 右眼
            [300, 95, 0.7],  # 左耳
            [340, 95, 0.7],  # 右耳
        ]
    )

    mock_bbox = [280, 80, 360, 200]  # 模拟人体边界框

    print(f"模拟图像尺寸: {test_image.shape}")
    print(f"模拟关键点数量: {len(mock_keypoints)}")
    print(f"模拟人体边界框: {mock_bbox}")

    # 测试双通道检测
    result = detector.detect_hairnet_with_keypoints(
        frame=test_image, human_bbox=mock_bbox, keypoints=mock_keypoints
    )

    print(f"\n检测结果:")
    print(f"  佩戴发网: {result.get('wearing_hairnet', False)}")
    print(f"  发网颜色: {result.get('hairnet_color', 'unknown')}")
    print(f"  置信度: {result.get('confidence', 0):.3f}")
    print(f"  ROI策略: {result.get('roi_strategy', 'unknown')}")
    print(f"  关键点数量: {result.get('keypoint_count', 0)}")
    print(f"  头部ROI坐标: {result.get('head_roi_coords', [])}")

    if "error" in result:
        print(f"  错误信息: {result['error']}")


if __name__ == "__main__":
    try:
        # 测试双通道策略检测
        test_dual_channel_detection()

        # 测试关键点检测功能
        test_keypoint_detection()

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
