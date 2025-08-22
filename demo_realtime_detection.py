#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时手部行为检测演示

展示实时视频检测和可视化功能，包括：
- 实时手部检测和关键点显示
- 行为识别结果可视化
- 运动轨迹显示
- 检测统计信息实时更新

Author: Trae AI Assistant
Date: 2025-01-21
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

# 本地模块导入
try:
    from src.services.realtime_video_detection import RealtimeVideoDetector, VisualizationConfig
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖模块都已正确安装")
    sys.exit(1)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/realtime_detection.log', encoding='utf-8')
        ]
    )


def main():
    """主演示函数"""
    print("=== 实时手部行为检测演示 ===")
    print("功能特性:")
    print("- 实时手部检测和关键点显示")
    print("- 行为识别结果可视化")
    print("- 运动轨迹显示")
    print("- 检测统计信息实时更新")
    print("\n控制键:")
    print("- 'q': 退出")
    print("- 'p': 暂停/继续")
    print("- 's': 保存当前帧")
    print("- 'r': 重置统计")
    print("\n" + "="*50)
    
    # 设置日志
    setup_logging()
    logger = get_logger(__name__)
    
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    
    try:
        # 配置可视化选项
        vis_config = VisualizationConfig(
            show_landmarks=True,
            show_bbox=True,
            show_trajectory=True,
            show_behavior_text=True,
            show_statistics=True,
            trajectory_length=30,
            font_scale=0.6,
            line_thickness=2
        )
        
        # 创建实时检测器
        logger.info("初始化实时检测器...")
        detector = RealtimeVideoDetector(visualization_config=vis_config)
        logger.info("✓ 实时检测器初始化完成")
        
        # 由于没有摄像头硬件，默认使用视频文件进行测试
        print("\n由于没有摄像头硬件，使用视频文件进行实时检测演示")
        
        # 查找可用的测试视频
        test_videos = [
            "test_video.mp4",
            "demo_video.mp4", 
            "sample_video.mp4",
            "handwash_demo.mp4",
            "resources/test_videos/handwashing_demo.mp4"
        ]
        
        video_path = None
        for video in test_videos:
            if os.path.exists(video):
                video_path = video
                print(f"找到测试视频: {video_path}")
                break
                
        if not video_path:
            video_path = input("请输入视频文件路径: ").strip()
            if not video_path or not os.path.exists(video_path):
                print("错误: 未找到可用的测试视频文件")
                print("请确保项目目录中有测试视频文件，或手动指定视频路径")
                return
        
        print(f"\n使用视频文件: {video_path}")
        
        # 视频文件检测
        if True:  # 始终进入视频检测模式
            
            # 询问是否保存输出视频
            save_output = input("是否保存输出视频? (y/n): ").strip().lower()
            output_path = None
            if save_output == 'y':
                timestamp = int(time.time())
                output_path = f"temp/realtime_output_{timestamp}.mp4"
                print(f"输出视频将保存到: {output_path}")
            
            print(f"\n开始处理视频: {video_path}")
            print("按 'q' 退出, 'p' 暂停, 's' 保存当前帧")
            
            # 处理视频
            start_time = time.time()
            statistics = detector.process_video(
                video_path=video_path,
                output_path=output_path,
                display=True
            )
            
            # 显示处理结果
            processing_time = time.time() - start_time
            print("\n" + "="*50)
            print("=== 处理完成 ===")
            print(f"总处理时间: {processing_time:.2f}秒")
            print(f"总帧数: {statistics['total_frames']}")
            print(f"检测帧数: {statistics['detection_count']}")
            print(f"检测率: {statistics['detection_rate']*100:.1f}%")
            print(f"平均FPS: {statistics['avg_fps']:.1f}")
            print("\n行为统计:")
            for behavior, count in statistics['behavior_counts'].items():
                percentage = (count / max(statistics['total_frames'], 1)) * 100
                print(f"  {behavior}: {count} ({percentage:.1f}%)")
            
            if output_path and os.path.exists(output_path):
                print(f"\n输出视频已保存: {output_path}")
            
            # 保存详细统计
            stats_file = f"temp/detection_stats_{int(time.time())}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                # 移除results字段以减少文件大小
                stats_copy = statistics.copy()
                stats_copy.pop('results', None)
                json.dump(stats_copy, f, indent=2, ensure_ascii=False)
            print(f"详细统计已保存: {stats_file}")
            
        else:
            print("无效选择")
            return
            
    except KeyboardInterrupt:
        print("\n用户中断检测")
        logger.info("用户中断检测")
        
    except Exception as e:
        print(f"\n检测过程中发生错误: {e}")
        logger.error(f"检测错误: {e}", exc_info=True)
        
    finally:
        # 清理资源
        try:
            detector.cleanup()
            print("\n资源清理完成")
        except Exception as e:
            print(f"资源清理失败: {e}")
        
        print("\n感谢使用实时手部行为检测系统!")


if __name__ == '__main__':
    main()