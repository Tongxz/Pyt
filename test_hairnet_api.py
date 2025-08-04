#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的发网检测API测试脚本
"""

import sys
import os
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.yolo_hairnet_detector import YOLOHairnetDetector

def test_hairnet_detection():
    """测试发网检测功能"""
    print("\n=== 发网检测测试 ===\n")
    
    # 初始化YOLOv8检测器
    detector = YOLOHairnetDetector(model_path='models/hairnet_detection.pt', device='auto')
    
    # 加载测试图片
    test_image_path = 'tests/fixtures/images/person/test_person.png'
    print(f"加载测试图片: {test_image_path}")
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"错误: 无法加载图片 {test_image_path}")
        return
    
    # 执行发网检测
    print("执行发网检测...")
    result = detector.detect(img)
    
    # 获取检测结果
    wearing_hairnet = result.get('wearing_hairnet', False)
    confidence = result.get('confidence', 0.0)
    detections = result.get('detections', [])
    visualization = result.get('visualization')
    
    # 打印检测结果
    print(f"\n是否佩戴发网: {'是' if wearing_hairnet else '否'}")
    print(f"置信度: {confidence:.3f}")
    
    # 打印检测到的目标
    if detections:
        print(f"\n检测到的目标数量: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"  {i+1}. 类别: {det['class']}, 置信度: {det['confidence']:.3f}, "
                  f"边界框: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
    
    # 显示可视化结果
    if visualization is not None:
        cv2.imshow('发网检测结果', visualization)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
        

if __name__ == '__main__':
    test_hairnet_detection()