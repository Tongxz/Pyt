#!/usr/bin/env python3
"""
测试更新后的人体检测器配置
验证从 yolov8n.pt 升级到 yolov8s.pt 的效果
"""

import sys
import time
from pathlib import Path

import cv2

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.detector import HumanDetector
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__, level="INFO")


def test_updated_detector():
    """测试更新后的检测器"""

    print("=== 测试更新后的人体检测器 ===")
    print("默认配置: yolov8s.pt + 置信度0.3 + 最小面积800")
    print()

    # 测试图像路径
    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"

    try:
        # 读取测试图像
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"无法读取测试图像: {test_image_path}")
            return

        print(f"测试图像: {test_image_path}")
        print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
        print()

        # 创建更新后的检测器（使用默认配置）
        print("初始化检测器...")
        start_time = time.time()
        detector = HumanDetector()  # 使用默认的 yolov8s.pt
        init_time = time.time() - start_time

        print(f"检测器初始化完成，耗时: {init_time:.3f}秒")
        print(f"置信度阈值: {detector.confidence_threshold}")
        print(f"最小检测框面积: {detector.min_box_area}")
        print(f"最大宽高比: {detector.max_box_ratio}")
        print()

        # 执行检测
        print("执行人体检测...")
        start_time = time.time()
        detections = detector.detect(image)
        detection_time = time.time() - start_time

        print(f"检测完成，耗时: {detection_time:.3f}秒")
        print(f"检测到人员数量: {len(detections)}")
        print()

        # 详细检测结果
        confidences = []
        areas = []

        if detections:
            print("检测详情:")

            for i, detection in enumerate(detections, 1):
                x1, y1, x2, y2 = detection["bbox"]
                confidence = detection["confidence"]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                ratio = max(width, height) / min(width, height)

                confidences.append(confidence)
                areas.append(area)

                print(f"  人员{i}: 置信度={confidence:.3f}, 位置=({x1},{y1},{x2},{y2})")
                print(f"         尺寸={width}x{height}, 面积={area:.0f}, 宽高比={ratio:.2f}")

            print()
            print("统计信息:")
            print(f"  平均置信度: {sum(confidences)/len(confidences):.3f}")
            print(f"  最低置信度: {min(confidences):.3f}")
            print(f"  最高置信度: {max(confidences):.3f}")
            print(f"  平均检测框面积: {sum(areas)/len(areas):.0f}")

            # 高置信度检测统计
            high_conf_count = sum(1 for conf in confidences if conf > 0.7)
            print(f"  高置信度检测(>0.7): {high_conf_count}/{len(detections)}")

        else:
            print("未检测到任何人员")

        print()

        # 创建可视化结果
        print("生成可视化结果...")
        result_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            # 绘制检测框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加置信度标签
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        # 保存结果
        output_path = "updated_detector_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"可视化结果保存到: {output_path}")

        return {
            "detection_count": len(detections),
            "detection_time": detection_time,
            "init_time": init_time,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "high_conf_count": sum(1 for conf in confidences if conf > 0.7)
            if confidences
            else 0,
        }

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        return None


def compare_with_old_config():
    """与旧配置进行对比"""

    print("\n=== 配置对比分析 ===")
    print()

    print("旧配置 (yolov8n.pt):")
    print("  - 模型: yolov8n.pt (最小最快)")
    print("  - 置信度阈值: 0.1 (很低，容易误检测)")
    print("  - 最小面积: 500 (较小，可能包含噪声)")
    print()

    print("新配置 (yolov8s.pt):")
    print("  - 模型: yolov8s.pt (小型，平衡速度和精度)")
    print("  - 置信度阈值: 0.3 (适中，减少误检测)")
    print("  - 最小面积: 800 (提高，过滤小目标噪声)")
    print()

    print("预期改进:")
    print("  ✓ 检测精度提升")
    print("  ✓ 减少低置信度误检测")
    print("  ✓ 过滤小尺寸噪声目标")
    print("  ✓ 更稳定的检测结果")
    print("  ⚠ 推理时间略有增加")
    print()


def performance_analysis(result):
    """性能分析"""

    if not result:
        return

    print("=== 性能分析 ===")
    print()

    print(f"检测性能:")
    print(f"  - 检测人数: {result['detection_count']}")
    print(f"  - 检测时间: {result['detection_time']:.3f}秒")
    print(f"  - 初始化时间: {result['init_time']:.3f}秒")

    if result["detection_count"] > 0:
        print(f"  - 平均置信度: {result['avg_confidence']:.3f}")
        print(f"  - 高置信度检测: {result['high_conf_count']}/{result['detection_count']}")

        # 性能评估
        if result["avg_confidence"] > 0.7:
            print("  ✓ 检测质量: 优秀")
        elif result["avg_confidence"] > 0.5:
            print("  ✓ 检测质量: 良好")
        else:
            print("  ⚠ 检测质量: 一般")

        if result["detection_time"] < 0.2:
            print("  ✓ 检测速度: 快")
        elif result["detection_time"] < 0.5:
            print("  ✓ 检测速度: 适中")
        else:
            print("  ⚠ 检测速度: 较慢")

    print()


if __name__ == "__main__":
    print("人体检测器配置更新测试")
    print("=" * 60)

    # 配置对比
    compare_with_old_config()

    # 测试更新后的检测器
    result = test_updated_detector()

    # 性能分析
    performance_analysis(result)

    print("测试完成！")
    print()
    print("总结:")
    print("- 已将默认模型从 yolov8n.pt 升级到 yolov8s.pt")
    print("- 优化了置信度阈值和过滤参数")
    print("- 预期获得更好的检测精度和稳定性")
