#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试YOLOv8发网检测模型

用法:
    python test_hairnet_model.py --weights models/hairnet_detection.pt --source path/to/image.jpg
    python test_hairnet_model.py --weights models/hairnet_detection.pt --source path/to/video.mp4
    python test_hairnet_model.py --weights models/hairnet_detection.pt --source 0  # 使用摄像头
"""

import os
import sys
import time
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='测试YOLOv8发网检测模型')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                        help='输入源，可以是图像、视频路径或摄像头编号(0)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='置信度阈值，默认为0.25')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU阈值，默认为0.45')
    parser.add_argument('--device', type=str, default='',
                        help='推理设备，例如cuda:0或cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='显示结果')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存结果为txt文件')
    parser.add_argument('--save-conf', action='store_true',
                        help='保存置信度到txt文件')
    parser.add_argument('--save-crop', action='store_true',
                        help='保存裁剪的预测框')
    parser.add_argument('--nosave', action='store_true',
                        help='不保存图像/视频')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='按类别过滤，例如--classes 0')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='保存结果的项目目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='保存结果的子目录名称')
    parser.add_argument('--exist-ok', action='store_true',
                        help='如果目录存在则不创建新目录')
    parser.add_argument('--line-thickness', type=int, default=3,
                        help='边界框厚度(像素)')
    parser.add_argument('--hide-labels', action='store_true',
                        help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true',
                        help='隐藏置信度')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='视频帧率步长')
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


def run_inference(args):
    """运行YOLOv8推理"""
    from ultralytics import YOLO
    
    # 检查模型文件
    if not os.path.exists(args.weights):
        print(f"错误: 模型文件不存在: {args.weights}")
        return False
    
    # 检查输入源
    if not args.source.isdigit() and not os.path.exists(args.source):
        print(f"错误: 输入源不存在: {args.source}")
        return False
    
    # 加载模型
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)
    
    # 设置推理参数
    inference_args = {
        'source': args.source,
        'conf': args.conf_thres,
        'iou': args.iou_thres,
        'show': args.view_img,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
        'save': not args.nosave,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'line_thickness': args.line_thickness,
        'hide_labels': args.hide_labels,
        'hide_conf': args.hide_conf,
        'vid_stride': args.vid_stride,
    }
    
    # 添加设备参数（如果指定）
    if args.device:
        inference_args['device'] = args.device
    
    # 添加类别过滤（如果指定）
    if args.classes is not None:
        inference_args['classes'] = args.classes
    
    # 开始推理
    print("\n开始推理...")
    start_time = time.time()
    results = model.predict(**inference_args)
    end_time = time.time()
    
    # 统计结果
    total_frames = len(results)
    total_objects = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    inference_time = end_time - start_time
    fps = total_frames / inference_time if inference_time > 0 else 0
    
    print(f"\n推理完成!")
    print(f"处理帧数: {total_frames}")
    print(f"检测到的目标数: {total_objects}")
    print(f"推理时间: {inference_time:.2f}秒")
    print(f"平均FPS: {fps:.2f}")
    
    # 获取结果路径
    if not args.nosave:
        result_path = os.path.join(args.project, args.name)
        print(f"结果保存在: {result_path}")
    
    return True


def main():
    args = parse_args()
    
    # 检查依赖项
    if not check_dependencies():
        return 1
    
    # 运行推理
    try:
        success = run_inference(args)
        if success:
            return 0
        else:
            return 1
    except Exception as e:
        print(f"\n推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())