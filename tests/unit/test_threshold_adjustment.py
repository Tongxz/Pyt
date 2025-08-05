#!/usr/bin/env python3
"""
发网检测阈值调整测试脚本
用于验证调整后的检测阈值效果
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np


def get_fixtures_dir():
    """获取测试数据目录"""
    return Path(__file__).parent.parent / "fixtures"


# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.detector import HumanDetector
from src.core.hairnet_detector import HairnetDetector


def test_threshold_adjustment():
    """
    测试阈值调整后的检测效果
    """
    print("=== 发网检测阈值调整测试 ===")

    # 初始化检测器
    try:
        hairnet_detector = HairnetDetector()
        human_detector = HumanDetector()
        print("✓ 检测器初始化成功")
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        return

    # 加载测试图像
    test_image_candidates = [
        "realistic_test_image.jpg",
        "real_hairnet_test.jpg",
        "tests/fixtures/images/person/factory_test.jpg",
        "test_light_blue_hairnet.jpg",
        "test_person.jpg",
    ]

    # 尝试所有测试图像，找到包含人员的图像
    test_image_path = None
    image = None
    persons = []

    for candidate in test_image_candidates:
        if os.path.exists(candidate):
            temp_image = cv2.imread(candidate)
            if temp_image is not None:
                print(f"尝试图像: {candidate}")
                try:
                    temp_persons = human_detector.detect(temp_image)
                    print(f"  检测到 {len(temp_persons)} 个人")
                    if len(temp_persons) > 0:
                        test_image_path = candidate
                        image = temp_image
                        persons = temp_persons
                        break
                except Exception as e:
                    print(f"  人体检测失败: {e}")

    if test_image_path is None or image is None or len(persons) == 0:
        print("✗ 未找到包含人员的测试图像")
        print(f"尝试的路径: {test_image_candidates}")
        return

    print(f"\n✓ 使用测试图像: {test_image_path}")
    print(f"图像尺寸: {image.shape}")
    print(f"✓ 检测到 {len(persons)} 个人")

    # 对每个人进行发网检测
    print("\n=== 详细检测结果 ===")
    total_with_hairnet = 0

    for i, person in enumerate(persons, 1):
        print(f"\n--- 人员 {i} ---")
        bbox = person["bbox"]
        keypoints = person.get("keypoints")

        print(f"人体边界框: {bbox}")

        # 使用发网检测器进行检测
        try:
            result = hairnet_detector.detect(image, [person])

            wearing_hairnet = result.get("wearing_hairnet", False)
            confidence = result.get("confidence", 0.0)
            roi_strategy = result.get("roi_strategy", "unknown")

            print(f"发网检测: {'是' if wearing_hairnet else '否'}")
            print(f"置信度: {confidence:.3f}")
            print(f"ROI策略: {roi_strategy}")

            # 显示调试信息
            debug_info = result.get("debug_info", {})
            if debug_info:
                print("调试信息:")
                print(f"  边缘密度: {debug_info.get('basic_edge_density', 0):.4f}")
                print(f"  敏感边缘密度: {debug_info.get('sensitive_edge_density', 0):.4f}")
                print(f"  轮廓数量: {result.get('contour_count', 0)}")
                print(f"  浅蓝色比例: {debug_info.get('light_blue_ratio', 0):.4f}")
                print(f"  浅色比例: {debug_info.get('light_color_ratio', 0):.4f}")
                print(f"  白色比例: {debug_info.get('white_ratio', 0):.4f}")
                print(f"  绿色比例: {debug_info.get('green_ratio', 0):.4f}")
                print(f"  总颜色比例: {debug_info.get('total_color_ratio', 0):.4f}")
                print(f"  上部边缘密度: {debug_info.get('upper_edge_density', 0):.4f}")
                print(f"  综合得分: {debug_info.get('total_score', 0):.4f}")

                # 显示各种检测条件的结果
                print("检测条件满足情况:")
                print(f"  浅蓝色发网: {debug_info.get('has_light_blue_hairnet', False)}")
                print(f"  一般发网: {debug_info.get('has_general_hairnet', False)}")
                print(f"  浅色发网: {debug_info.get('has_light_hairnet', False)}")
                print(f"  基础发网: {debug_info.get('has_basic_hairnet', False)}")
                print(f"  最低标准: {debug_info.get('has_minimal_hairnet', False)}")

            if wearing_hairnet:
                total_with_hairnet += 1

        except Exception as e:
            print(f"✗ 发网检测失败: {e}")

    # 总结
    print(f"\n=== 检测总结 ===")
    print(f"总人数: {len(persons)}")
    print(f"佩戴发网: {total_with_hairnet} 人")
    print(f"未佩戴发网: {len(persons) - total_with_hairnet} 人")
    print(f"合规率: {(total_with_hairnet / len(persons) * 100):.1f}%")

    # 阈值建议
    print(f"\n=== 阈值调整建议 ===")
    if total_with_hairnet == 0:
        print("⚠️  当前阈值可能过于严格，建议适当降低")
        print("   可以考虑降低边缘密度和轮廓数量要求")
    elif total_with_hairnet == len(persons):
        print("⚠️  当前阈值可能过于宽松，建议适当提高")
        print("   可以考虑提高颜色比例和综合得分要求")
    else:
        print("✓ 当前阈值设置较为合理")


if __name__ == "__main__":
    test_threshold_adjustment()
