#!/usr/bin/env python
"""
测试YOLO发网检测模型的基本信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_model():
    """测试YOLO模型的基本信息"""
    
    model_path = "models/hairnet_detection.pt"
    
    print(f"测试YOLO模型: {model_path}")
    
    try:
        # 加载模型
        model = YOLO(model_path)
        print("模型加载成功")
        
        # 打印模型信息
        print(f"模型类别数量: {len(model.names)}")
        print(f"模型类别: {model.names}")
        
        # 测试图像
        test_images = [
            "tests/fixtures/images/hairnet/test_person.png",
            "tests/fixtures/images/hairnet/7月23日.png"
        ]
        
        for image_path in test_images:
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                continue
                
            print(f"\n测试图像: {image_path}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
                
            print(f"图像尺寸: {image.shape}")
            
            # 进行检测
            results = model(image, conf=0.1, iou=0.45)  # 降低置信度阈值
            
            print(f"检测结果数量: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"  结果 {i}:")
                if result.boxes is not None:
                    boxes = result.boxes
                    print(f"    检测框数量: {len(boxes)}")
                    
                    for j, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names.get(cls_id, f"unknown_{cls_id}")
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        print(f"      检测框 {j}: 类别={cls_name}({cls_id}), 置信度={conf:.3f}, 坐标={bbox}")
                else:
                    print("    没有检测到任何目标")
                    
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_model()