#!/usr/bin/env python3
"""
单独测试发网检测器功能
"""

import cv2
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.core.yolo_hairnet_detector import YOLOHairnetDetector
from src.utils.logger import get_logger

logger = get_logger(__name__, level="INFO")

def test_hairnet_detector():
    """测试发网检测器"""
    
    print("=== 测试发网检测器 ===")
    print()
    
    # 测试图像路径
    test_images = [
        "tests/fixtures/images/person/test_person.png",
        "tests/fixtures/images/hairnet/7月23日.png"
    ]
    
    try:
        # 初始化发网检测器
        print("初始化发网检测器...")
        hairnet_detector = YOLOHairnetDetector()
        print("发网检测器初始化完成")
        print()
        
        for image_path in test_images:
            print(f"测试图像: {image_path}")
            
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  ❌ 无法读取图像: {image_path}")
                    continue
                
                print(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}")
                
                # 直接对整个图像进行发网检测
                print("  执行发网检测...")
                
                # 先测试基础detect方法
                basic_result = hairnet_detector.detect(image)
                print(f"  基础检测结果: {basic_result}")
                
                # 再测试合规性检测方法
                compliance_result = hairnet_detector.detect_hairnet_compliance(image)
                
                print(f"  检测到 {compliance_result.get('total_persons', 0)} 个人")
                
                # 打印发网检测详情
                detections = compliance_result.get('detections', [])
                for i, detection in enumerate(detections):
                    has_hairnet = detection.get('has_hairnet', False)
                    confidence = detection.get('confidence', 0.0)
                    print(f"    人员 {i+1}: 佩戴发网={has_hairnet}，置信度={confidence:.3f}")
                print()
                
            except Exception as e:
                print(f"  ❌ 检测失败: {e}")
                import traceback
                traceback.print_exc()
                print()
                
    except Exception as e:
        print(f"❌ 发网检测器初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hairnet_detector()