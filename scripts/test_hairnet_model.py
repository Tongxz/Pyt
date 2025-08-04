#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试发网检测模型

使用训练好的YOLOv8模型进行发网检测测试
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hairnet_testing')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试发网检测模型')
    parser.add_argument('--model', type=str, default='./models/hairnet_detector.pt',
                        help='模型路径')
    parser.add_argument('--source', type=str, required=True,
                        help='测试图像或视频路径，或摄像头索引（0表示默认摄像头）')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU阈值')
    parser.add_argument('--device', type=str, default='auto',
                        help='推理设备，可选：cpu, cuda, auto')
    parser.add_argument('--save', action='store_true',
                        help='保存检测结果')
    parser.add_argument('--output', type=str, default='./results',
                        help='结果保存目录')
    
    return parser.parse_args()

def test_model(args):
    """测试YOLOv8模型"""
    # 检查模型路径
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'模型文件不存在: {args.model}')
    
    # 选择设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f'使用设备: {device}')
    
    # 加载模型
    logger.info(f'加载模型: {args.model}')
    model = YOLO(args.model)
    
    # 检查输入源
    source = args.source
    if source.isdigit():
        source = int(source)  # 摄像头索引
        logger.info(f'使用摄像头: {source}')
    elif not Path(source).exists():
        raise FileNotFoundError(f'输入源不存在: {source}')
    
    # 创建输出目录
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'结果将保存到: {output_dir}')
    
    # 运行推理
    logger.info('开始推理...')
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        device=device,
        save=args.save,
        project=args.output if args.save else None,
        name='hairnet_detection' if args.save else None,
        stream=True  # 流式处理，适用于视频
    )
    
    # 处理结果
    frame_count = 0
    detection_count = 0
    
    for result in results:
        frame_count += 1
        
        # 获取检测结果
        boxes = result.boxes
        detection_count += len(boxes)
        
        # 获取原始图像
        frame = result.orig_img
        
        # 显示结果（如果不是流式处理）
        if isinstance(source, (str, int)):
            # 在图像上绘制检测结果
            annotated_frame = result.plot()
            
            # 显示图像
            cv2.imshow('YOLOv8 Hairnet Detection', annotated_frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 关闭窗口
    cv2.destroyAllWindows()
    
    logger.info(f'推理完成! 处理了 {frame_count} 帧，检测到 {detection_count} 个目标')
    
    return {
        'frame_count': frame_count,
        'detection_count': detection_count
    }

def main():
    """主函数"""
    args = parse_args()
    
    logger.info('='*50)
    logger.info('发网检测模型测试开始')
    logger.info(f'模型路径: {args.model}')
    logger.info(f'输入源: {args.source}')
    logger.info(f'置信度阈值: {args.conf}')
    logger.info(f'IoU阈值: {args.iou}')
    logger.info('='*50)
    
    try:
        results = test_model(args)
        logger.info('测试完成!')
        logger.info(f'处理帧数: {results["frame_count"]}')
        logger.info(f'检测目标数: {results["detection_count"]}')
    except Exception as e:
        logger.error(f'测试过程中发生错误: {e}', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()