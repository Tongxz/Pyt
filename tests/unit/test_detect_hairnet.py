#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试发网检测的detect_hairnet方法
"""

import cv2
import numpy as np
import sys
import os

from pathlib import Path
# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.hairnet_detector import HairnetDetector

def test_detect_hairnet():
    """测试detect_hairnet方法"""
    print("=== 测试发网检测的detect_hairnet方法 ===")
    
    # 初始化检测器
    detector = HairnetDetector()
    
    # 加载测试图像
    test_image_path = "test_images/test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"警告: 测试图像不存在 {test_image_path}，使用模拟图像")
        # 创建模拟图像
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        frame = cv2.imread(test_image_path)
    
    print(f"图像尺寸: {frame.shape}")
    
    # 测试1: 不提供任何参数（完全自动检测）
    print("\n--- 测试1: 完全自动检测 ---")
    result1 = detector.detect_hairnet(frame)
    print(f"检测结果: {result1}")
    
    # 测试2: 提供人体边界框
    print("\n--- 测试2: 提供人体边界框 ---")
    # 模拟人体边界框 [x1, y1, x2, y2]
    human_bbox = [100, 50, 300, 400]
    result2 = detector.detect_hairnet(frame, human_bbox=human_bbox)
    print(f"检测结果: {result2}")
    
    # 测试3: 提供关键点（模拟）
    print("\n--- 测试3: 提供关键点 ---")
    # 模拟关键点数据（17个关键点，每个3个值：x, y, confidence）
    keypoints = np.random.rand(17, 3) * 100
    result3 = detector.detect_hairnet(frame, human_bbox=human_bbox, keypoints=keypoints)
    print(f"检测结果: {result3}")
    
    # 显示统计信息
    print("\n--- 检测器统计信息 ---")
    print(f"总检测次数: {detector.total_detections}")
    print(f"发网检测次数: {detector.hairnet_detections}")
    print(f"关键点成功次数: {detector.stats['keypoint_success']}")
    print(f"边界框回退次数: {detector.stats['bbox_fallback']}")
    
    if detector.total_detections > 0:
        success_rate = detector.hairnet_detections / detector.total_detections * 100
        print(f"发网检测率: {success_rate:.1f}%")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    detector = HairnetDetector()
    
    # 测试空图像
    print("\n--- 测试空图像 ---")
    try:
        result = detector.detect_hairnet(None)
        print(f"空图像结果: {result}")
    except Exception as e:
        print(f"空图像异常: {e}")
    
    # 测试无效边界框
    print("\n--- 测试无效边界框 ---")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    invalid_bbox = [1000, 1000, 1100, 1100]  # 超出图像范围
    result = detector.detect_hairnet(frame, human_bbox=invalid_bbox)
    print(f"无效边界框结果: {result}")

if __name__ == "__main__":
    test_detect_hairnet()
    test_error_handling()
    print("\n测试完成！")