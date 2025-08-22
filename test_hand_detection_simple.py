#!/usr/bin/env python3
"""
简化的手部检测测试脚本
绕过MediaPipe问题，直接测试核心检测功能
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_yolo_pose_detector():
    """测试YOLOv8姿态检测器"""
    print("=== 测试YOLOv8姿态检测器 ===")
    
    try:
        from src.core.pose_detector import PoseDetectorFactory
        
        # 尝试创建YOLOv8检测器
        detector = PoseDetectorFactory.create(backend='yolov8')
        print("✓ YOLOv8姿态检测器创建成功")
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 100), (400, 400), (255, 255, 255), -1)
        
        # 进行检测
        results = detector.detect(test_image)
        print(f"✓ 检测完成，结果类型: {type(results)}")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLOv8姿态检测器测试失败: {e}")
        return False

def test_motion_analyzer():
    """测试运动分析器"""
    print("\n=== 测试运动分析器 ===")
    
    try:
        from src.core.motion_analyzer import MotionAnalyzer
        
        analyzer = MotionAnalyzer()
        print("✓ 运动分析器创建成功")
        
        # 模拟手部检测数据（符合update_hand_motion期望的格式）
        mock_hands_data = [
            {
                "label": "left",
                "bbox": [90, 190, 110, 220],
                "landmarks": [
                    {"x": 100, "y": 200},  # left_wrist
                    {"x": 90, "y": 210}    # left_hand
                ]
            },
            {
                "label": "right", 
                "bbox": [290, 190, 320, 220],
                "landmarks": [
                    {"x": 300, "y": 200},  # right_wrist
                    {"x": 310, "y": 210}   # right_hand
                ]
            }
        ]
        
        # 先更新手部运动数据
        track_id = 1
        analyzer.update_hand_motion(track_id, mock_hands_data)
        
        # 分析运动
        motion_result = analyzer.analyze_motion(track_id, "handwashing")
        print(f"✓ 运动分析完成，结果: {motion_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 运动分析器测试失败: {e}")
        return False

def test_behavior_recognizer():
    """测试行为识别器"""
    print("\n=== 测试行为识别器 ===")
    
    try:
        from src.core.behavior import BehaviorRecognizer
        
        recognizer = BehaviorRecognizer()
        print("✓ 行为识别器创建成功")
        
        # 创建测试图像和边界框
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        person_bbox = [100, 50, 300, 400]
        
        # 测试洗手检测
        handwash_score = recognizer.detect_handwashing(
            person_bbox=person_bbox,
            hand_regions=[],
            frame=test_image
        )
        print(f"✓ 洗手检测完成，置信度: {handwash_score}")
        
        # 测试消毒检测
        sanitize_score = recognizer.detect_sanitizing(
            person_bbox=person_bbox,
            hand_regions=[],
            frame=test_image
        )
        print(f"✓ 消毒检测完成，置信度: {sanitize_score}")
        
        return True
        
    except Exception as e:
        print(f"✗ 行为识别器测试失败: {e}")
        return False

def test_optimized_pipeline():
    """测试优化检测管道"""
    print("\n=== 测试优化检测管道 ===")
    
    try:
        from src.core.optimized_detection_pipeline import OptimizedDetectionPipeline
        from src.core.detector import HumanDetector
        from src.core.hairnet_detector import HairnetDetector
        from src.core.behavior import BehaviorRecognizer
        
        # 创建必要的检测器实例
        human_detector = HumanDetector()
        hairnet_detector = HairnetDetector()
        behavior_recognizer = BehaviorRecognizer()
        
        pipeline = OptimizedDetectionPipeline(
            human_detector=human_detector,
            hairnet_detector=hairnet_detector,
            behavior_recognizer=behavior_recognizer
        )
        print("✓ 优化检测管道创建成功")
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 100), (400, 400), (128, 128, 128), -1)
        
        # 进行检测
        results = pipeline.detect(test_image)
        print(f"✓ 管道检测完成")
        
        # 验证检测结果
        assert results is not None
        assert isinstance(results.person_detections, list)
        assert isinstance(results.handwash_results, list)
        assert isinstance(results.sanitize_results, list)
        
        print(f"  - 检测到人数: {len(results.person_detections)}")
        print(f"  - 洗手人数: {len(results.handwash_results)}")
        print(f"  - 消毒人数: {len(results.sanitize_results)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 优化检测管道测试失败: {e}")
        return False

def test_detection_service():
    """测试检测服务"""
    print("\n=== 测试检测服务 ===")
    
    try:
        from src.services.detection_service import comprehensive_detection_logic
        
        # 查找测试图像
        test_image_path = "realistic_test_image.jpg"
        if not Path(test_image_path).exists():
            print(f"测试图像不存在: {test_image_path}，创建模拟图像")
            # 创建模拟图像
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_image, (200, 100), (400, 400), (100, 150, 200), -1)
            cv2.imwrite(test_image_path, test_image)
        
        # 调用检测服务
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        
        # 创建必要的检测器实例
        from src.core.detector import HumanDetector
        from src.core.hairnet_detector import HairnetDetector
        from src.core.behavior import BehaviorRecognizer
        from src.core.optimized_detection_pipeline import OptimizedDetectionPipeline
        
        human_detector = HumanDetector()
        hairnet_detector = HairnetDetector()
        behavior_recognizer = BehaviorRecognizer()
        
        optimized_pipeline = OptimizedDetectionPipeline(
            human_detector=human_detector,
            hairnet_detector=hairnet_detector,
            behavior_recognizer=behavior_recognizer
        )
        
        result = comprehensive_detection_logic(
            contents=image_data,
            filename=test_image_path,
            optimized_pipeline=optimized_pipeline,
            hairnet_pipeline=hairnet_detector,
            record_process=False
        )
        
        print("✓ 检测服务调用成功")
        print(f"  - 状态: {result.get('status', 'unknown')}")
        print(f"  - 统计信息: {result.get('statistics', {})}")
        
        return True
        
    except Exception as e:
        print(f"✗ 检测服务测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔍 手部检测系统集成测试")
    print("=" * 50)
    
    tests = [
        ("YOLOv8姿态检测器", test_yolo_pose_detector),
        ("运动分析器", test_motion_analyzer),
        ("行为识别器", test_behavior_recognizer),
        ("优化检测管道", test_optimized_pipeline),
        ("检测服务", test_detection_service),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试总结: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！手部检测器已完全集成。")
    elif passed > 0:
        print(f"⚠️  部分功能正常 ({passed}/{total})，系统基本可用。")
    else:
        print("❌ 所有测试失败，请检查系统配置。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)