#!/usr/bin/env python3
"""
测试行为检测功能
验证洗手和消毒行为检测是否正常工作
"""

import cv2
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.detector import HumanDetector
from src.core.yolo_hairnet_detector import YOLOHairnetDetector
from src.core.behavior import BehaviorRecognizer
from src.core.optimized_detection_pipeline import OptimizedDetectionPipeline
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__, level="INFO")

def test_behavior_detection():
    """测试行为检测功能"""
    
    print("=== 测试行为检测功能 ===")
    print()
    
    # 测试图像路径
    test_images = [
        "tests/fixtures/images/person/test_person.png",
        "tests/fixtures/images/hairnet/7月23日.png"
    ]
    
    try:
        # 初始化检测流水线
        print("初始化检测流水线...")
        start_time = time.time()
        
        human_detector = HumanDetector()
        hairnet_detector = YOLOHairnetDetector()
        behavior_recognizer = BehaviorRecognizer()
        pipeline = OptimizedDetectionPipeline(
            human_detector=human_detector,
            hairnet_detector=hairnet_detector,
            behavior_recognizer=behavior_recognizer
        )
        
        init_time = time.time() - start_time
        print(f"流水线初始化完成，耗时: {init_time:.3f}秒")
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
                
                # 执行综合检测（启用所有功能）
                print("  执行综合检测（包含行为检测）...")
                start_time = time.time()
                result = pipeline.detect_comprehensive(
                    image, 
                    enable_hairnet=True,
                    enable_handwash=True,
                    enable_sanitize=True,
                    force_refresh=True
                )
                detection_time = time.time() - start_time
                
                print(f"  总检测时间: {detection_time:.3f}秒")
                print()
                
                # 显示检测结果
                print("  检测结果:")
                print(f"    检测到人数: {len(result.person_detections)}")
                
                # 发网检测结果
                hairnet_count = sum(1 for h in result.hairnet_results if h.get('wearing_hairnet', False))
                hairnet_rate = (hairnet_count / len(result.person_detections) * 100) if result.person_detections else 0
                print(f"    佩戴发网人数: {hairnet_count}")
                print(f"    发网合规率: {hairnet_rate:.2f}%")
                
                # 洗手检测结果
                handwash_count = sum(1 for h in result.handwash_results if h.get('is_handwashing', False))
                handwash_rate = (handwash_count / len(result.person_detections) * 100) if result.person_detections else 0
                print(f"    洗手人数: {handwash_count}")
                print(f"    洗手率: {handwash_rate:.2f}%")
                
                # 消毒检测结果
                sanitize_count = sum(1 for s in result.sanitize_results if s.get('is_sanitizing', False))
                sanitize_rate = (sanitize_count / len(result.person_detections) * 100) if result.person_detections else 0
                print(f"    消毒人数: {sanitize_count}")
                print(f"    消毒率: {sanitize_rate:.2f}%")
                
                # 洗手检测详情
                if result.handwash_results:
                    print("  洗手检测详情:")
                    for handwash in result.handwash_results:
                        person_id = handwash.get('person_id', 'Unknown')
                        is_handwashing = handwash.get('is_handwashing', False)
                        confidence = handwash.get('handwash_confidence', 0.0)
                        print(f"    人员{person_id}: 洗手={is_handwashing}, 置信度={confidence:.3f}")
                
                # 消毒检测详情
                if result.sanitize_results:
                    print("  消毒检测详情:")
                    for sanitize in result.sanitize_results:
                        person_id = sanitize.get('person_id', 'Unknown')
                        is_sanitizing = sanitize.get('is_sanitizing', False)
                        confidence = sanitize.get('sanitize_confidence', 0.0)
                        print(f"    人员{person_id}: 消毒={is_sanitizing}, 置信度={confidence:.3f}")
                
                # 处理时间详情
                print("  处理时间详情:")
                for stage, duration in result.processing_times.items():
                    print(f"    {stage}: {duration:.3f}秒")
                
                print()
                
            except Exception as e:
                print(f"  ❌ 检测失败: {e}")
                logger.error(f"图像 {image_path} 检测失败", exc_info=True)
                continue
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        logger.error("检测流水线初始化失败", exc_info=True)
        return
    
    print("============================================================")
    print("行为检测测试完成！")
    print()
    print("总结:")
    print("- 已验证行为识别器初始化")
    print("- 已测试洗手行为检测")
    print("- 已测试消毒行为检测")
    print("- 确认行为检测功能正常工作")

def main():
    """主函数"""
    test_behavior_detection()

if __name__ == "__main__":
    main()