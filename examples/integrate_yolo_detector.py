#!/usr/bin/env python

"""YOLOv8 发网检测器示例

演示如何使用 HairnetDetectionFactory 创建 YOLOv8 发网检测器

用法:
    python examples/integrate_yolo_detector.py --image path/to/image.jpg
    python examples/integrate_yolo_detector.py --image path/to/image.jpg --detector-type yolo
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入 HairnetDetectionFactory
from src.core.hairnet_detection_factory import HairnetDetectionFactory


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 发网检测器示例")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--detector-type",
        type=str,
        default="yolo",
        choices=["auto", "yolo"],
        help="检测器类型，可选 auto, yolo，默认为 yolo",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
        help="YOLOv8 模型路径，默认为 models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值，默认为 0.25")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU阈值，默认为 0.45")
    parser.add_argument(
        "--device", type=str, default="auto", help="计算设备，可选 cpu, cuda, auto"
    )
    parser.add_argument("--save", action="store_true", help="保存结果图像")
    parser.add_argument(
        "--output", type=str, default="result.jpg", help="输出图像路径，默认为 result.jpg"
    )
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 检查输入图像是否存在
    if not os.path.exists(args.image):
        print(f"错误: 输入图像不存在: {args.image}")
        return

    try:
        # 获取可用的检测器类型
        available_types = HairnetDetectionFactory.get_available_detector_types()
        print(f"可用的检测器类型: {', '.join(available_types.keys())}")

        # 检查 YOLOv8 检测器是否可用
        if HairnetDetectionFactory.is_yolo_available():
            print("YOLOv8 发网检测器可用")
        else:
            print("YOLOv8 发网检测器不可用")
            print("错误: YOLOv8 检测器不可用，请确保已安装所有依赖项")
            return

        # 创建发网检测器
        print(f"创建 {args.detector_type} 类型的发网检测器")
        detector = HairnetDetectionFactory.create_detector(
            detector_type=args.detector_type,
            model_path=args.model,
            device=args.device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
        )

        # 读取输入图像
        print(f"读取图像: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"错误: 无法读取图像: {args.image}")
            return

        # 执行发网检测
        print("执行发网检测...")

        # 使用YOLOv8检测器进行检测
        result = detector.detect(image)
        visualization = result.get("visualization")
        wearing_hairnet = result.get("wearing_hairnet", False)
        confidence = result.get("confidence", 0.0)
        detections = result.get("detections", [])

        # 打印检测结果
        print("\n检测结果:")
        print(f"是否佩戴发网: {'是' if wearing_hairnet else '否'}")
        print(f"置信度: {confidence:.3f}")

        # 打印检测到的目标
        if detections:
            print(f"\n检测到的目标数量: {len(detections)}")
            for i, det in enumerate(detections):
                print(
                    f"  {i+1}. 类别: {det['class']}, 置信度: {det['confidence']:.3f}, "
                    f"边界框: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
                )

        # 显示可视化结果
        if visualization is not None:
            # 保存结果图像
            if args.save:
                output_path = args.output
                cv2.imwrite(output_path, visualization)
                print(f"\n结果图像已保存: {output_path}")

            # 显示结果图像
            cv2.imshow("发网检测结果", visualization)
            print("\n按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 打印统计信息
        stats = detector.get_stats()
        print("\n统计信息:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
