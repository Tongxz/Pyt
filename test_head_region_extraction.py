#!/usr/bin/env python3
"""
测试头部区域提取和发网检测
"""

import sys
from pathlib import Path

import cv2

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from src.core.detector import HumanDetector
from src.core.yolo_hairnet_detector import YOLOHairnetDetector
from src.utils.logger import get_logger

logger = get_logger(__name__, level="INFO")


def test_head_region_extraction():
    """测试头部区域提取和发网检测"""

    print("=== 测试头部区域提取和发网检测 ===")
    print()

    # 测试图像路径
    test_images = [
        "tests/fixtures/images/person/test_person.png",
        "tests/fixtures/images/hairnet/7月23日.png",
    ]

    try:
        # 初始化检测器
        print("初始化检测器...")
        human_detector = HumanDetector()
        hairnet_detector = YOLOHairnetDetector()
        print("检测器初始化完成")
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

                # 人体检测
                print("  执行人体检测...")
                person_detections = human_detector.detect(image)
                print(f"  检测到 {len(person_detections)} 个人")

                if not person_detections:
                    print("  没有检测到人体，跳过发网检测")
                    continue

                # 为每个人提取头部区域并进行发网检测
                for i, detection in enumerate(person_detections):
                    print(f"\n  人员 {i+1}:")
                    bbox = detection.get("bbox", [0, 0, 0, 0])
                    confidence = detection.get("confidence", 0)
                    print(f"    人体检测置信度: {confidence:.3f}")
                    print(f"    人体边界框: {bbox}")

                    x1, y1, x2, y2 = map(int, bbox)

                    # 提取头部区域（使用与优化流水线相同的逻辑）
                    head_height = int((y2 - y1) * 0.3)  # 头部约占人体高度的30%
                    head_y1 = max(0, y1)
                    head_y2 = min(image.shape[0], y1 + head_height)
                    head_x1 = max(0, x1)
                    head_x2 = min(image.shape[1], x2)

                    print(f"    头部区域: [{head_x1}, {head_y1}, {head_x2}, {head_y2}]")
                    print(f"    头部区域尺寸: {head_x2-head_x1}x{head_y2-head_y1}")

                    if head_y2 > head_y1 and head_x2 > head_x1:
                        head_region = image[head_y1:head_y2, head_x1:head_x2]

                        # 保存头部区域图像用于调试
                        head_filename = (
                            f"head_region_person_{i+1}_{Path(image_path).stem}.jpg"
                        )
                        cv2.imwrite(head_filename, head_region)
                        print(f"    头部区域已保存: {head_filename}")

                        # 发网检测
                        print("    执行发网检测...")
                        hairnet_result = hairnet_detector.detect_hairnet_compliance(
                            head_region
                        )

                        print(f"    发网检测结果:")
                        print(
                            f"      佩戴发网: {hairnet_result.get('wearing_hairnet', False)}"
                        )
                        print(f"      置信度: {hairnet_result.get('confidence', 0):.3f}")
                        print(
                            f"      头部ROI坐标: {hairnet_result.get('head_roi_coords', [])}"
                        )
                    else:
                        print("    头部区域无效，跳过发网检测")

                print()

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                import traceback

                traceback.print_exc()
                print()

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_head_region_extraction()
