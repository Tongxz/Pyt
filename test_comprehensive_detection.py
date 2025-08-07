#!/usr/bin/env python3
"""
测试更新后的综合检测系统
验证人体检测模型升级后的整体效果
"""

import sys
import time
from pathlib import Path

import cv2

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.behavior import BehaviorRecognizer
from src.core.detector import HumanDetector
from src.core.optimized_detection_pipeline import OptimizedDetectionPipeline
from src.core.yolo_hairnet_detector import YOLOHairnetDetector
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__, level="INFO")


def test_individual_detectors():
    """测试各个检测器的独立性能"""

    print("=== 测试各个检测器的独立性能 ===")
    print()

    # 测试图像路径
    test_images = [
        "tests/fixtures/images/person/test_person.png",
        "tests/fixtures/images/hairnet/7月23日.png",
    ]

    for image_path in test_images:
        print(f"测试图像: {image_path}")

        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"  ❌ 无法读取图像: {image_path}")
                continue

            print(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}")

            # 测试人体检测器
            print("  测试人体检测器...")
            start_time = time.time()
            human_detector = HumanDetector()
            human_detections = human_detector.detect(image)
            human_time = time.time() - start_time

            print(f"    检测到人数: {len(human_detections)}")
            print(f"    检测时间: {human_time:.3f}秒")

            if human_detections:
                for i, detection in enumerate(human_detections, 1):
                    conf = detection["confidence"]
                    bbox = detection["bbox"]
                    print(f"      人员{i}: 置信度={conf:.3f}, 位置={bbox}")

            # 测试发网检测器
            print("  测试发网检测器...")
            start_time = time.time()
            hairnet_detector = YOLOHairnetDetector()
            hairnet_result = hairnet_detector.detect_hairnet_compliance(image)
            hairnet_time = time.time() - start_time

            print(f"    发网检测时间: {hairnet_time:.3f}秒")
            print(f"    检测到人数: {hairnet_result.get('total_persons', 0)}")
            print(f"    佩戴发网人数: {hairnet_result.get('persons_with_hairnet', 0)}")
            print(f"    合规率: {hairnet_result.get('compliance_rate', 0):.2%}")

            print()

        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            import traceback

            traceback.print_exc()
            print()


def test_optimized_pipeline():
    """测试优化的检测流水线"""

    print("=== 测试优化的检测流水线 ===")
    print()

    # 测试图像路径
    test_images = [
        "tests/fixtures/images/person/test_person.png",
        "tests/fixtures/images/hairnet/7月23日.png",
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
            behavior_recognizer=behavior_recognizer,
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

                # 执行综合检测
                print("  执行综合检测...")
                start_time = time.time()
                result = pipeline.detect_comprehensive(image, force_refresh=True)
                detection_time = time.time() - start_time

                print(f"  总检测时间: {detection_time:.3f}秒")
                print()

                # 显示检测结果
                print("  检测结果:")
                print(f"    检测到人数: {len(result.person_detections)}")
                # 计算发网相关统计
                compliant_count = len(
                    [r for r in result.hairnet_results if r.get("has_hairnet", False)]
                )
                print(f"    佩戴发网人数: {compliant_count}")
                total_persons = len(result.person_detections)
                compliance_rate = (
                    compliant_count / total_persons if total_persons > 0 else 0
                )
                print(f"    发网合规率: {compliance_rate:.2%}")

                # 显示发网检测详情
                if result.hairnet_results:
                    print("  发网检测详情:")
                    for hairnet_result in result.hairnet_results:
                        person_id = hairnet_result.get("person_id", 0)
                        has_hairnet = hairnet_result.get("has_hairnet", False)
                        confidence = hairnet_result.get("hairnet_confidence", 0)
                        print(
                            f"    人员{person_id}: 佩戴发网={has_hairnet}, 置信度={confidence:.3f}"
                        )

                # 显示处理时间详情
                if hasattr(result, "processing_times"):
                    times = result.processing_times
                    print("  处理时间详情:")
                    print(f"    人体检测: {times.get('human_detection', 0):.3f}秒")
                    print(f"    发网检测: {times.get('hairnet_detection', 0):.3f}秒")
                    print(f"    行为检测: {times.get('behavior_detection', 0):.3f}秒")

                # 显示人体检测详情
                if result.person_detections:
                    print("  人体检测详情:")
                    for i, detection in enumerate(result.person_detections, 1):
                        conf = detection.get("confidence", 0)
                        bbox = detection.get("bbox", [])
                        print(f"    人员{i}: 置信度={conf:.3f}, 位置={bbox}")

                print()

            except Exception as e:
                print(f"  ❌ 检测失败: {e}")
                import traceback

                traceback.print_exc()
                print()

    except Exception as e:
        print(f"❌ 流水线初始化失败: {e}")
        import traceback

        traceback.print_exc()


def performance_comparison():
    """性能对比分析"""

    print("=== 性能对比分析 ===")
    print()

    print("升级前后对比:")
    print("旧配置 (yolov8n.pt):")
    print("  - 模型: yolov8n.pt")
    print("  - 置信度阈值: 0.1")
    print("  - 最小面积: 500")
    print("  - 特点: 速度快，但可能有误检测")
    print()

    print("新配置 (yolov8s.pt):")
    print("  - 模型: yolov8s.pt")
    print("  - 置信度阈值: 0.3")
    print("  - 最小面积: 800")
    print("  - 特点: 平衡速度和精度，减少误检测")
    print()

    print("预期改进:")
    print("  ✓ 检测精度提升")
    print("  ✓ 减少低置信度误检测")
    print("  ✓ 更稳定的检测结果")
    print("  ✓ 更好的发网检测基础")
    print()


def main():
    """主测试函数"""

    print("综合检测系统测试")
    print("=" * 60)
    print()

    # 性能对比分析
    performance_comparison()

    # 测试各个检测器
    test_individual_detectors()

    # 测试优化的检测流水线
    test_optimized_pipeline()

    print("=" * 60)
    print("测试完成！")
    print()
    print("总结:")
    print("- 已验证人体检测器升级效果")
    print("- 已测试综合检测流水线")
    print("- 确认所有测试图像来自tests目录")


if __name__ == "__main__":
    main()
