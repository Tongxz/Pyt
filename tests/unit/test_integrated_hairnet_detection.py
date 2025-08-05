#!/usr/bin/env python3
"""
集成测试：验证增强ROI算法集成到发网检测系统
"""

import os
import sys
from typing import Any, Dict

import cv2
import numpy as np


def get_fixtures_dir():
    """获取测试数据目录"""
    return Path(__file__).parent.parent / "fixtures"


from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from src.core.detector import HumanDetector
    from src.core.hairnet_detector import HairnetDetectionPipeline, HairnetDetector
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


def test_integrated_detection():
    """
    测试集成后的发网检测系统
    """
    print("=== 集成发网检测系统测试 ===")

    # 检查测试图像
    test_image_path = "tests/fixtures/images/person/test_person.png"
    if not os.path.exists(test_image_path):
        print(f"错误: 未找到测试图像 '{test_image_path}'")
        return False

    try:
        # 加载测试图像
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"错误: 无法加载图像 '{test_image_path}'")
            return False

        print(f"成功加载测试图像: {image.shape}")

        # 初始化检测器
        print("\n=== 初始化检测器 ===")
        human_detector = HumanDetector()
        hairnet_detector = HairnetDetector()
        pipeline = HairnetDetectionPipeline(human_detector, hairnet_detector)

        # 检查是否成功集成增强ROI
        if hasattr(hairnet_detector, "use_enhanced_roi"):
            print(f"增强ROI集成状态: {hairnet_detector.use_enhanced_roi}")
            if hairnet_detector.enhanced_roi_extractor:
                print("增强ROI提取器已成功初始化")
            else:
                print("增强ROI提取器初始化失败")
        else:
            print("警告: 未检测到增强ROI集成")

        # 执行发网检测
        print("\n=== 执行发网检测 ===")
        results = pipeline.detect_hairnet_compliance(image)

        # 显示检测结果
        print("\n=== 检测结果 ===")
        print(f"检测到人数: {results.get('total_persons', 0)}")
        print(f"佩戴发网: {results.get('persons_with_hairnet', 0)} 人")
        print(f"未佩戴发网: {results.get('persons_without_hairnet', 0)} 人")
        print(f"合规率: {results.get('compliance_rate', 0) * 100:.2f}%")
        print(f"平均置信度: {results.get('average_confidence', 0):.3f}")

        # 详细分析每个检测结果
        detections = results.get("detections", [])
        for i, detection in enumerate(detections, 1):
            print(f"\n--- 人员 {i} ---")
            print(f"人体边界框: {detection.get('bbox', [])}")
            print(f"头部区域: {detection.get('head_coords', [])}")
            print(f"ROI策略: {detection.get('roi_strategy', 'unknown')}")
            print(f"发网检测: {'是' if detection.get('has_hairnet', False) else '否'}")
            print(f"置信度: {detection.get('confidence', 0):.3f}")

            # 显示增强ROI的详细信息
            if detection.get("enhanced_roi_used", False):
                print("✓ 使用了增强ROI算法")
                roi_quality = detection.get("roi_quality_score", 0)
                print(f"ROI质量评分: {roi_quality:.3f}")

                roi_info = detection.get("roi_method_info", {})
                if roi_info:
                    print(f"最佳方法: {roi_info.get('best_method', 'unknown')}")
                    print(f"方法评分: {roi_info.get('method_scores', {})}")
            else:
                print("使用了传统ROI方法")

        # 测试单独的发网检测器
        print("\n=== 测试单独发网检测器 ===")
        single_result = hairnet_detector.detect_hairnet(image)
        print(f"单独检测结果: {single_result.get('has_hairnet', False)}")
        print(f"置信度: {single_result.get('confidence', 0):.3f}")
        print(f"ROI策略: {single_result.get('roi_strategy', 'unknown')}")

        if single_result.get("enhanced_roi_used", False):
            print("✓ 单独检测也使用了增强ROI算法")

        print("\n=== 测试完成 ===")
        return True

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_roi_comparison():
    """
    对比传统ROI和增强ROI的效果
    """
    print("\n=== ROI方法对比测试 ===")

    test_image_path = "tests/fixtures/images/person/test_person.png"
    if not os.path.exists(test_image_path):
        print(f"跳过对比测试: 未找到 '{test_image_path}'")
        return

    try:
        image = cv2.imread(test_image_path)
        human_detector = HumanDetector()

        # 检查图像是否成功加载
        if image is None:
            print("跳过对比测试: 图像加载失败")
            return

        # 检测人体
        persons = human_detector.detect(image)
        if not persons:
            print("未检测到人体，跳过对比测试")
            return

        person = persons[0]  # 使用第一个检测到的人
        bbox = person["bbox"]

        print(f"人体边界框: {bbox}")

        # 测试增强ROI
        hairnet_detector = HairnetDetector()
        if hairnet_detector.use_enhanced_roi:
            enhanced_result = hairnet_detector._extract_head_roi_enhanced(image, bbox)
            if enhanced_result:
                print(
                    f"增强ROI: {enhanced_result['strategy']}, 质量: {enhanced_result.get('quality_score', 0):.3f}"
                )
            else:
                print("增强ROI提取失败")

        # 测试传统ROI
        traditional_result = hairnet_detector._extract_head_roi_from_bbox(image, bbox)
        if traditional_result:
            print(f"传统ROI: {traditional_result['strategy']}")
        else:
            print("传统ROI提取失败")

    except Exception as e:
        print(f"对比测试失败: {e}")


if __name__ == "__main__":
    success = test_integrated_detection()
    test_roi_comparison()

    if success:
        print("\n✓ 集成测试成功完成")
    else:
        print("\n✗ 集成测试失败")
        sys.exit(1)
