#!/usr/bin/env python3
"""
发网检测参数调试脚本
用于分析当前检测参数并提供优化建议
"""

import os
import sys

import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.append("/Users/zhou/Code/python/Pyt")

from src.core.detector import HumanDetector
from src.core.hairnet_detector import HairnetDetector


def analyze_image_with_debug(image_path):
    """分析图像并输出详细的调试信息"""
    print(f"\n=== 分析图像: {os.path.basename(image_path)} ===")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    print(f"图像尺寸: {image.shape}")

    # 初始化检测器
    human_detector = HumanDetector()
    hairnet_detector = HairnetDetector()

    # 检测人体
    detections = human_detector.detect(image)
    print(f"检测到人数: {len(detections)}")

    if not detections:
        print("未检测到人员，无法进行发网分析")
        return

    # 分析每个人
    for i, detection in enumerate(detections, 1):
        print(f"\n--- 人员 {i} 详细分析 ---")
        bbox = detection["bbox"]
        print(f"人体边界框: {bbox}")

        # 提取头部ROI
        x1, y1, x2, y2 = bbox
        person_roi = image[y1:y2, x1:x2]

        if person_roi.size == 0:
            print("ROI区域为空")
            continue

        print(f"ROI尺寸: {person_roi.shape}")

        # 调用发网检测并获取详细信息
        result = hairnet_detector._detect_hairnet_with_pytorch(person_roi)

        print(f"发网检测结果: {result['has_hairnet']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"ROI策略: {result.get('roi_strategy', 'unknown')}")

        # 如果有详细的检测参数，输出它们
        if "debug_info" in result:
            debug_info = result["debug_info"]
            print("\n检测参数详情:")
            for key, value in debug_info.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        # 分析颜色特征
        analyze_color_features(person_roi)

        # 分析边缘特征
        analyze_edge_features(person_roi)


def analyze_color_features(roi):
    """分析ROI的颜色特征"""
    print("\n颜色特征分析:")

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 浅蓝色掩码 (发网常见颜色)
    light_blue_mask = cv2.inRange(hsv, (90, 30, 100), (130, 255, 255))
    light_blue_ratio = np.sum(light_blue_mask > 0) / (roi.shape[0] * roi.shape[1])

    # 浅色掩码
    light_color_mask = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
    light_color_ratio = np.sum(light_color_mask > 0) / (roi.shape[0] * roi.shape[1])

    # 白色掩码
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    white_ratio = np.sum(white_mask > 0) / (roi.shape[0] * roi.shape[1])

    # 绿色掩码
    green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
    green_ratio = np.sum(green_mask > 0) / (roi.shape[0] * roi.shape[1])

    total_color_ratio = light_blue_ratio + light_color_ratio + white_ratio + green_ratio

    print(f"  浅蓝色比例: {light_blue_ratio:.4f}")
    print(f"  浅色比例: {light_color_ratio:.4f}")
    print(f"  白色比例: {white_ratio:.4f}")
    print(f"  绿色比例: {green_ratio:.4f}")
    print(f"  总颜色比例: {total_color_ratio:.4f}")

    # 上部区域分析
    upper_roi = roi[: roi.shape[0] // 3, :]
    if upper_roi.size > 0:
        upper_hsv = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2HSV)
        upper_light_blue_mask = cv2.inRange(upper_hsv, (90, 30, 100), (130, 255, 255))
        upper_light_blue_ratio = np.sum(upper_light_blue_mask > 0) / (
            upper_roi.shape[0] * upper_roi.shape[1]
        )
        print(f"  上部浅蓝色比例: {upper_light_blue_ratio:.4f}")


def analyze_edge_features(roi):
    """分析ROI的边缘特征"""
    print("\n边缘特征分析:")

    # 转换为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])

    # 上部区域边缘密度
    upper_roi = roi[: roi.shape[0] // 3, :]
    if upper_roi.size > 0:
        upper_gray = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2GRAY)
        upper_edges = cv2.Canny(upper_gray, 50, 150)
        upper_edge_density = np.sum(upper_edges > 0) / (
            upper_roi.shape[0] * upper_roi.shape[1]
        )
    else:
        upper_edge_density = 0

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_contours = [c for c in contours if 5 <= cv2.contourArea(c) <= 100]

    print(f"  边缘密度: {edge_density:.4f}")
    print(f"  上部边缘密度: {upper_edge_density:.4f}")
    print(f"  总轮廓数: {len(contours)}")
    print(f"  小轮廓数: {len(small_contours)}")


def suggest_threshold_adjustments():
    """根据分析结果建议阈值调整"""
    print("\n=== 阈值调整建议 ===")
    print("基于当前检测结果，建议以下调整策略:")
    print("\n1. 如果所有人都被检测为未佩戴发网（当前情况）:")
    print("   - 降低边缘密度要求 (0.025 -> 0.020)")
    print("   - 降低轮廓数量要求 (5-7 -> 4-6)")
    print("   - 降低颜色比例要求 (0.012 -> 0.008)")
    print("   - 降低上部边缘密度要求 (0.030 -> 0.025)")

    print("\n2. 如果仍有误检（将未佩戴识别为佩戴）:")
    print("   - 增加多条件组合验证")
    print("   - 提高置信度阈值")
    print("   - 加强上部区域检测权重")

    print("\n3. 推荐的平衡调整:")
    print("   - edge_density: 0.020")
    print("   - upper_edge_density: 0.025")
    print("   - small_contours: > 4")
    print("   - total_color_ratio: 0.008")
    print("   - light_blue_ratio: 0.006")


def main():
    """主函数"""
    print("发网检测参数调试分析")
    print("=" * 50)

    # 查找测试图像
    test_images = [
        "realistic_test_image.jpg",
        "test_light_blue_hairnet.jpg",
        "real_hairnet_test.jpg",
        "factory_test.jpg",
        "test_image.jpg",
    ]

    base_dir = "/Users/zhou/Code/python/Pyt"

    found_image = None
    for img_name in test_images:
        img_path = os.path.join(base_dir, img_name)
        if os.path.exists(img_path):
            found_image = img_path
            break

    if found_image:
        analyze_image_with_debug(found_image)
    else:
        print("未找到可用的测试图像")
        print("请确保以下图像之一存在:")
        for img in test_images:
            print(f"  - {img}")

    suggest_threshold_adjustments()


if __name__ == "__main__":
    main()
