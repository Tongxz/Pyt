#!/usr/bin/env python

"""
使用 YOLOv8 发网检测器的示例脚本

演示如何使用 YOLOHairnetDetector 类进行发网检测

用法:
    python examples/use_yolo_hairnet_detector.py --image path/to/image.jpg
    python examples/use_yolo_hairnet_detector.py --image path/to/image.jpg --model models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入 YOLOHairnetDetector
from src.core.yolo_hairnet_detector import YOLOHairnetDetector


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 发网检测器示例")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--model",
        type=str,
        default="models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
        help="YOLOv8 模型路径，默认为 models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值，默认为 0.25")
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

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return

    try:
        # 初始化 YOLOHairnetDetector
        print(f"初始化 YOLOHairnetDetector，模型: {args.model}，设备: {args.device}")
        detector = YOLOHairnetDetector(
            model_path=args.model, device=args.device, conf_thres=args.conf_thres
        )

        # 读取输入图像
        print(f"读取图像: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"错误: 无法读取图像: {args.image}")
            return

        # 执行发网检测
        print("执行发网检测...")
        result = detector.detect(image)

        # 打印检测结果
        print("\n检测结果:")
        print(f"是否佩戴发网: {'是' if result['wearing_hairnet'] else '否'}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"\n检测到的目标数量: {len(result['detections'])}")

        for i, det in enumerate(result["detections"]):
            print(
                f"  {i+1}. 类别: {det['class']}, 置信度: {det['confidence']:.3f}, "
                f"边界框: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
            )

        # 显示可视化结果
        if result["visualization"] is not None:
            # 保存结果图像
            if args.save:
                output_path = args.output
                cv2.imwrite(output_path, result["visualization"])
                print(f"\n结果图像已保存: {output_path}")

            # 显示结果图像
            cv2.imshow("YOLOv8 发网检测结果", result["visualization"])
            print("\n按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\n警告: 无可视化结果")

        # 打印统计信息
        stats = detector.get_stats()
        print("\n统计信息:")
        print(f"总检测次数: {stats['total_detections']}")
        print(f"发网检测次数: {stats['hairnet_detections']}")
        print(f"发网佩戴率: {stats['hairnet_rate']:.2%}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
