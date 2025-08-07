#!/usr/bin/env python3
"""
测试性能优化效果
验证避免重复人体检测后的性能提升
"""

import cv2
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.detector import HumanDetector
from src.core.yolo_hairnet_detector import YOLOHairnetDetector
from src.core.optimized_detection_pipeline import OptimizedDetectionPipeline
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__, level="INFO")

def test_performance_comparison():
    """比较优化前后的性能"""
    
    print("=== 性能优化效果测试 ===")
    print()
    
    # 测试图像
    image_path = "tests/fixtures/images/hairnet/7月23日.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    print(f"测试图像: {image_path}")
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    print()
    
    # 方法1: 分别调用检测器（模拟优化前的重复检测）
    print("方法1: 分别调用检测器（模拟重复检测）")
    start_time = time.time()
    
    # 人体检测
    human_detector = HumanDetector()
    human_detections = human_detector.detect(image)
    
    # 发网检测（不传递人体检测结果，会重复检测）
    hairnet_detector = YOLOHairnetDetector()
    hairnet_result = hairnet_detector.detect_hairnet_compliance(image)
    
    method1_time = time.time() - start_time
    print(f"  总耗时: {method1_time:.3f}秒")
    print(f"  检测到人数: {len(human_detections)}")
    print(f"  发网合规率: {hairnet_result.get('compliance_rate', 0)*100:.1f}%")
    print()
    
    # 方法2: 使用优化的流水线（避免重复检测）
    print("方法2: 使用优化的流水线（避免重复检测）")
    start_time = time.time()
    
    pipeline = OptimizedDetectionPipeline(
        human_detector=human_detector,
        hairnet_detector=hairnet_detector
    )
    result = pipeline.detect_comprehensive(image, 
                                         enable_handwash=False, 
                                         enable_sanitize=False)
    
    method2_time = time.time() - start_time
    print(f"  总耗时: {method2_time:.3f}秒")
    print(f"  检测到人数: {len(result.person_detections)}")
    
    # 计算发网合规率
    total_persons = len(result.person_detections)
    persons_with_hairnet = sum(1 for r in result.hairnet_results if r.get('has_hairnet', False))
    compliance_rate = (persons_with_hairnet / total_persons * 100) if total_persons > 0 else 0
    print(f"  发网合规率: {compliance_rate:.1f}%")
    print()
    
    # 性能对比
    print("=== 性能对比结果 ===")
    improvement = ((method1_time - method2_time) / method1_time * 100) if method1_time > 0 else 0
    print(f"方法1耗时: {method1_time:.3f}秒")
    print(f"方法2耗时: {method2_time:.3f}秒")
    print(f"性能提升: {improvement:.1f}%")
    
    if improvement > 0:
        print("✅ 优化成功！避免了重复人体检测，提升了性能")
    else:
        print("⚠️  性能提升不明显，可能需要进一步优化")

def main():
    """主函数"""
    try:
        test_performance_comparison()
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()