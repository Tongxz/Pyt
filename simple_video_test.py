#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的视频显示测试脚本
用于测试OpenCV窗口显示功能
"""

import cv2
import numpy as np
from pathlib import Path
import time

def find_test_video():
    """查找测试视频文件"""
    possible_paths = [
        "tests/fixtures/videos/20250724072708.mp4",
        "tests/fixtures/videos/test_video.mp4",
        "data/test_video.mp4"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"找到测试视频: {path}")
            return str(Path(path).absolute())
    
    print("未找到测试视频文件")
    return None

def main():
    """主函数"""
    print("=== 简单视频显示测试 ===")
    
    # 查找测试视频
    video_path = find_test_video()
    if not video_path:
        print("错误: 无法找到测试视频")
        return
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    print(f"成功打开视频: {video_path}")
    print("控制键:")
    print("  空格键: 暂停/继续")
    print("  ESC键: 退出")
    print("  'r'键: 重新开始")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕，重新开始...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
        
        # 在帧上添加信息
        if 'frame' in locals():
            display_frame = frame.copy()
            cv2.putText(display_frame, f'Frame: {frame_count}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, 'Press ESC to exit, SPACE to pause', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if paused:
                cv2.putText(display_frame, 'PAUSED', 
                           (display_frame.shape[1]//2-50, display_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 显示帧
            cv2.imshow('Video Test', display_frame)
        
        # 处理按键
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC键
            print("用户退出")
            break
        elif key == ord(' '):  # 空格键
            paused = not paused
            print(f"视频 {'暂停' if paused else '继续'}")
        elif key == ord('r'):  # r键
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            paused = False
            print("重新开始播放")
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("测试完成")

if __name__ == "__main__":
    main()