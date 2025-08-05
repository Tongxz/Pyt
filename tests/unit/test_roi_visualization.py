#!/usr/bin/env python3
"""
测试ROI可视化功能
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from pathlib import Path

from scripts.visualize_roi import ROIVisualizer


def get_fixtures_dir():
    """获取测试数据目录"""
    return Path(__file__).parent.parent / "fixtures"


def create_test_image():
    """创建一个测试图像，包含模拟的人体和发网"""
    # 创建一个640x480的测试图像
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 浅灰色背景

    # 绘制模拟的人体（矩形）
    # 人体1 - 左侧
    cv2.rectangle(image, (100, 150), (200, 400), (150, 150, 150), -1)  # 身体
    cv2.rectangle(image, (120, 120), (180, 180), (200, 180, 160), -1)  # 头部
    cv2.rectangle(image, (130, 130), (170, 150), (100, 200, 255), -1)  # 浅蓝色发网区域

    # 人体2 - 中间
    cv2.rectangle(image, (280, 140), (380, 420), (150, 150, 150), -1)  # 身体
    cv2.rectangle(image, (300, 110), (360, 170), (200, 180, 160), -1)  # 头部
    cv2.rectangle(image, (310, 120), (350, 140), (255, 255, 255), -1)  # 白色发网区域

    # 人体3 - 右侧（无发网）
    cv2.rectangle(image, (460, 160), (560, 410), (150, 150, 150), -1)  # 身体
    cv2.rectangle(image, (480, 130), (540, 190), (200, 180, 160), -1)  # 头部（无发网）

    # 添加一些噪声和纹理
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def test_roi_visualization():
    """测试ROI可视化功能"""
    print("创建测试图像...")
    test_image = create_test_image()

    # 保存测试图像
    test_image_path = "test_hairnet_image.jpg"
    cv2.imwrite(test_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    print(f"测试图像已保存: {test_image_path}")

    print("初始化ROI可视化器...")
    try:
        visualizer = ROIVisualizer()

        print("开始ROI可视化分析...")
        output_path = "test_roi_analysis.png"
        visualizer.visualize_single_image(test_image_path, output_path)

        print(f"ROI可视化分析完成，结果保存在: {output_path}")

    except Exception as e:
        print(f"ROI可视化测试失败: {e}")
        import traceback

        traceback.print_exc()

    # 清理测试文件
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"已清理测试图像: {test_image_path}")


def test_with_real_image():
    """使用真实图像测试（如果存在）"""
    # 常见的测试图像路径
    test_paths = [
        "tests/fixtures/images/person/test_image.jpg",
        "test.jpg",
        "sample.jpg",
        "hairnet_test.jpg",
    ]

    real_image_path = None
    for path in test_paths:
        if os.path.exists(path):
            real_image_path = path
            break

    if real_image_path:
        print(f"找到真实测试图像: {real_image_path}")
        try:
            visualizer = ROIVisualizer()
            output_path = f"real_image_roi_analysis.png"
            visualizer.visualize_single_image(real_image_path, output_path)
            print(f"真实图像ROI分析完成: {output_path}")
        except Exception as e:
            print(f"真实图像ROI分析失败: {e}")
    else:
        print("未找到真实测试图像，跳过真实图像测试")


if __name__ == "__main__":
    print("=== ROI可视化功能测试 ===")

    # 测试1: 使用模拟图像
    print("\n1. 测试模拟图像...")
    test_roi_visualization()

    # 测试2: 使用真实图像（如果存在）
    print("\n2. 测试真实图像...")
    test_with_real_image()

    print("\n=== 测试完成 ===")
    print("\n使用方法:")
    print("1. 单张图像: python visualize_roi.py --image your_image.jpg")
    print("2. 批量处理: python visualize_roi.py --dir your_image_directory")
    print("3. 指定输出: python visualize_roi.py --image test.jpg --save result.png")
