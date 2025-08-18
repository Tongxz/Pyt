#!/usr/bin/env python

"""
使用训练好的YOLOv8发网检测模型的示例代码

用法:
    python example_usage.py --weights models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt --source path/to/image.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="使用YOLOv8发网检测模型示例")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
        help="模型权重路径，默认为models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
    )
    parser.add_argument(
        "--source", type=str, required=True, help="输入源，可以是图像、视频路径或摄像头编号(0)"
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值，默认为0.25")
    parser.add_argument("--device", type=str, default="", help="推理设备，例如cuda:0或cpu")
    return parser.parse_args()


def check_dependencies():
    """检查依赖项"""
    try:
        import ultralytics
        from ultralytics import YOLO

        print(f"Ultralytics版本: {ultralytics.__version__}")
        return True
    except ImportError:
        print("错误: 未安装ultralytics库，请使用以下命令安装:")
        print("pip install ultralytics")
        return False


def detect_hairnet(model, image_path, conf_thres=0.25):
    """检测图像中的发网"""
    # 运行推理
    results = model(image_path, conf=conf_thres)

    # 处理结果
    detections = []
    for r in results:
        boxes = r.boxes  # 边界框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 边界框坐标
            conf = float(box.conf[0])  # 置信度
            cls = int(box.cls[0])  # 类别
            cls_name = model.names[cls]  # 类别名称

            detections.append(
                {"class": cls_name, "confidence": conf, "bbox": [x1, y1, x2, y2]}
            )

    return detections, results[0].plot()  # 返回检测结果和可视化图像


def process_image(args):
    """处理单张图像"""
    from ultralytics import YOLO

    # 加载模型
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)

    # 检测发网
    print(f"处理图像: {args.source}")
    detections, result_image = detect_hairnet(model, args.source, args.conf_thres)

    # 打印检测结果
    print(f"\n检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        print(
            f"  {i+1}. 类别: {det['class']}, 置信度: {det['confidence']:.2f}, "
            f"边界框: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
        )

    # 保存结果图像
    output_path = f"result_{os.path.basename(args.source)}"
    cv2.imwrite(output_path, result_image)
    print(f"\n结果图像已保存: {output_path}")

    # 显示结果图像
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(args):
    """处理视频或摄像头"""
    from ultralytics import YOLO

    # 加载模型
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)

    # 打开视频或摄像头
    if args.source.isdigit():
        print(f"打开摄像头: {args.source}")
        cap = cv2.VideoCapture(int(args.source))
    else:
        print(f"打开视频: {args.source}")
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"错误: 无法打开视频源: {args.source}")
        return

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化变量
    output_path = None
    out = None

    # 创建视频写入器
    if not args.source.isdigit():
        output_path = f"result_{os.path.basename(args.source)}"
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # 正确的fourcc调用方式
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 处理视频帧
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔10帧处理一次（提高性能）
        if frame_count % 10 == 0:
            # 运行推理
            results = model(frame, conf=args.conf_thres)

            # 获取可视化结果
            result_frame = results[0].plot()

            # 显示帧率
            cv2.putText(
                result_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # 显示结果
            cv2.imshow("Result", result_frame)

            # 保存结果
            if not args.source.isdigit():
                out.write(result_frame)
        else:
            # 显示原始帧
            cv2.imshow("Result", frame)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # 释放资源
    cap.release()
    if not args.source.isdigit():
        out.release()
    cv2.destroyAllWindows()

    if not args.source.isdigit():
        print(f"\n结果视频已保存: {output_path}")


def main():
    args = parse_args()

    # 检查依赖项
    if not check_dependencies():
        return 1

    # 检查模型文件
    if not os.path.exists(args.weights):
        print(f"错误: 模型文件不存在: {args.weights}")
        return 1

    # 检查输入源
    if not args.source.isdigit() and not os.path.exists(args.source):
        print(f"错误: 输入源不存在: {args.source}")
        return 1

    # 处理输入源
    try:
        if args.source.isdigit() or args.source.endswith((".mp4", ".avi", ".mov")):
            process_video(args)
        elif args.source.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            process_image(args)
        else:
            print(f"错误: 不支持的输入源格式: {args.source}")
            return 1

        return 0
    except Exception as e:
        print(f"\n处理过程中出错: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
