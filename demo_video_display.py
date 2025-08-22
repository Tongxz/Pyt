#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频实时显示检测演示

专门用于显示视频检测过程的简化版本，确保能看到实时的检测可视化效果。

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
import logging
from typing import Dict, List, Optional, Tuple, Any

# 本地模块导入
try:
    from src.core.enhanced_hand_detector import EnhancedHandDetector, DetectionMode
    from src.core.enhanced_motion_analyzer import EnhancedMotionAnalyzer
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖模块都已正确安装")
    sys.exit(1)


class VideoDisplayDetector:
    """视频显示检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.logger = get_logger(__name__)
        
        # 初始化核心组件
        try:
            self.hand_detector = EnhancedHandDetector(
                detection_mode=DetectionMode.WITH_FALLBACK,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                quality_threshold=0.3
            )
            self.motion_analyzer = EnhancedMotionAnalyzer()
            self.logger.info("✓ 检测器初始化完成")
        except Exception as e:
            self.logger.error(f"检测器初始化失败: {e}")
            raise
        
        # 统计信息
        self.frame_count = 0
        self.detection_count = 0
        self.behavior_counts = {'handwash': 0, 'sanitize': 0, 'none': 0}
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.paused = False
        
        # 颜色定义
        self.colors = {
            'hand_bbox': (0, 255, 0),      # 绿色
            'landmarks': (255, 0, 0),       # 蓝色
            'handwash': (0, 255, 0),        # 绿色
            'sanitize': (255, 0, 255),      # 紫色
            'none': (128, 128, 128),        # 灰色
            'text': (255, 255, 255),        # 白色
            'background': (0, 0, 0),        # 黑色
            'paused': (0, 255, 255)         # 黄色
        }
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理单帧图像"""
        if self.paused:
            # 暂停时只显示暂停信息
            vis_frame = self._draw_paused_info(frame)
            return vis_frame, {'behavior': 'paused', 'confidence': 0.0}
            
        self.frame_count += 1
        start_time = time.time()
        
        try:
            # 手部检测
            detection_results = self.hand_detector.detect_hands_robust(frame)
            
            if not detection_results:
                return self._draw_no_detection_info(frame), {'behavior': 'none', 'confidence': 0.0}
            
            self.detection_count += 1
            
            # 转换检测结果
            track_id = 1
            hands_data = [{
                'label': result.hand_label,
                'landmarks': result.landmarks,
                'bbox': result.bbox,
                'confidence': result.confidence
            } for result in detection_results]
            
            # 更新运动分析
            self.motion_analyzer.update_hand_motion(track_id, hands_data)
            
            # 行为分析
            handwash_confidence = self.motion_analyzer.analyze_handwashing_enhanced(track_id)
            sanitize_confidence = self.motion_analyzer.analyze_sanitizing_enhanced(track_id)
            
            # 确定行为
            if handwash_confidence > 0.6:
                behavior = 'handwash'
                confidence = handwash_confidence
            elif sanitize_confidence > 0.6:
                behavior = 'sanitize'
                confidence = sanitize_confidence
            else:
                behavior = 'none'
                confidence = max(handwash_confidence, sanitize_confidence)
            
            self.behavior_counts[behavior] += 1
            
            # 可视化
            vis_frame = self._visualize_detection(
                frame, detection_results, behavior, confidence,
                handwash_confidence, sanitize_confidence
            )
            
            result = {
                'behavior': behavior,
                'confidence': confidence,
                'handwash_confidence': handwash_confidence,
                'sanitize_confidence': sanitize_confidence,
                'processing_time': time.time() - start_time
            }
            
            return vis_frame, result
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            return self._draw_error_info(frame, str(e)), {'behavior': 'error', 'confidence': 0.0}
            
    def _visualize_detection(self, 
                           frame: np.ndarray,
                           detection_results: List[Any],
                           behavior: str,
                           confidence: float,
                           handwash_conf: float,
                           sanitize_conf: float) -> np.ndarray:
        """可视化检测结果"""
        vis_frame = frame.copy()
        
        # 绘制手部检测结果
        for result in detection_results:
            # 绘制边界框
            if result.bbox:
                x1, y1, x2, y2 = result.bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), 
                            self.colors['hand_bbox'], 2)
                
                # 绘制置信度
                conf_text = f"{result.hand_label}: {result.confidence:.2f}"
                cv2.putText(vis_frame, conf_text, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          self.colors['text'], 1)
            
            # 绘制关键点
            if result.landmarks:
                self._draw_hand_landmarks(vis_frame, result.landmarks)
        
        # 绘制行为识别结果
        self._draw_behavior_info(vis_frame, behavior, confidence, 
                               handwash_conf, sanitize_conf)
        
        # 绘制统计信息
        self._draw_statistics(vis_frame)
        
        # 绘制控制提示
        self._draw_controls(vis_frame)
        
        return vis_frame
        
    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float]]) -> None:
        """绘制手部关键点"""
        h, w = frame.shape[:2]
        
        # 绘制关键点
        for landmark in landmarks:
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(frame, (x, y), 3, self.colors['landmarks'], -1)
            
    def _draw_behavior_info(self, 
                          frame: np.ndarray,
                          behavior: str,
                          confidence: float,
                          handwash_conf: float,
                          sanitize_conf: float) -> None:
        """绘制行为识别信息"""
        # 主要行为结果
        behavior_text = f"行为: {behavior} ({confidence:.2f})"
        behavior_color = self.colors.get(behavior, self.colors['text'])
        
        # 绘制背景
        cv2.rectangle(frame, (10, 10), (400, 120), self.colors['background'], -1)
        cv2.rectangle(frame, (10, 10), (400, 120), self.colors['text'], 1)
        
        # 绘制文本
        y_offset = 35
        cv2.putText(frame, behavior_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   behavior_color, 2)
        
        y_offset += 30
        cv2.putText(frame, f"洗手置信度: {handwash_conf:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.colors['handwash'], 1)
        
        y_offset += 25
        cv2.putText(frame, f"消毒置信度: {sanitize_conf:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.colors['sanitize'], 1)
                   
    def _draw_statistics(self, frame: np.ndarray) -> None:
        """绘制统计信息"""
        h, w = frame.shape[:2]
        
        # 计算FPS
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        # 绘制统计背景
        stats_x = w - 220
        cv2.rectangle(frame, (stats_x, 10), (w - 10, 140), self.colors['background'], -1)
        cv2.rectangle(frame, (stats_x, 10), (w - 10, 140), self.colors['text'], 1)
        
        # 绘制统计文本
        y_offset = 30
        stats_info = [
            f"帧数: {self.frame_count}",
            f"检测数: {self.detection_count}",
            f"FPS: {self.current_fps:.1f}",
            f"洗手: {self.behavior_counts['handwash']}",
            f"消毒: {self.behavior_counts['sanitize']}",
            f"检测率: {self.detection_count/max(self.frame_count,1)*100:.1f}%"
        ]
        
        for info in stats_info:
            cv2.putText(frame, info, (stats_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       self.colors['text'], 1)
            y_offset += 18
            
    def _draw_controls(self, frame: np.ndarray) -> None:
        """绘制控制提示"""
        h, w = frame.shape[:2]
        
        # 绘制控制提示背景
        controls_y = h - 80
        cv2.rectangle(frame, (10, controls_y), (w - 10, h - 10), self.colors['background'], -1)
        cv2.rectangle(frame, (10, controls_y), (w - 10, h - 10), self.colors['text'], 1)
        
        # 绘制控制提示文本
        controls = "控制: 'q'=退出 | 空格=暂停/继续 | 's'=保存帧 | 'r'=重置统计"
        cv2.putText(frame, controls, (20, controls_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.colors['text'], 1)
                   
    def _draw_no_detection_info(self, frame: np.ndarray) -> np.ndarray:
        """绘制无检测信息"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # 绘制"无检测"信息
        text = "无手部检测"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(vis_frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # 绘制统计信息和控制提示
        self._draw_statistics(vis_frame)
        self._draw_controls(vis_frame)
        
        return vis_frame
        
    def _draw_paused_info(self, frame: np.ndarray) -> np.ndarray:
        """绘制暂停信息"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # 绘制暂停信息
        text = "已暂停 - 按空格继续"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(vis_frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['paused'], 2)
        
        # 绘制统计信息和控制提示
        self._draw_statistics(vis_frame)
        self._draw_controls(vis_frame)
        
        return vis_frame
        
    def _draw_error_info(self, frame: np.ndarray, error_msg: str) -> np.ndarray:
        """绘制错误信息"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # 绘制错误信息
        text = f"处理错误: {error_msg}"
        cv2.putText(vis_frame, text, (20, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 255), 2)  # 红色
        
        return vis_frame
        
    def process_video_file(self, video_path: str) -> None:
        """处理视频文件并实时显示"""
        print(f"\n开始处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return
            
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height}, {fps:.1f}FPS, {total_frames}帧")
        
        # 创建窗口
        window_name = '实时手部行为检测 - 视频显示'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))
        
        frame_delay = max(1, int(1000 / fps))  # 毫秒
        
        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\n视频播放完成")
                        break
                        
                    # 处理帧
                    vis_frame, result = self.process_frame(frame)
                else:
                    # 暂停时重复显示当前帧
                    if 'current_frame' in locals():
                        vis_frame = self._draw_paused_info(current_frame)
                    else:
                        continue
                
                current_frame = frame if not self.paused else current_frame
                
                # 显示帧
                cv2.imshow(window_name, vis_frame)
                
                # 按键控制
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):  # 退出
                    break
                elif key == ord(' '):  # 暂停/继续
                    self.paused = not self.paused
                    print(f"{'暂停' if self.paused else '继续'}播放")
                elif key == ord('s'):  # 保存当前帧
                    timestamp = int(time.time())
                    filename = f'video_frame_{timestamp}.jpg'
                    cv2.imwrite(filename, vis_frame)
                    print(f"保存帧: {filename}")
                elif key == ord('r'):  # 重置统计
                    self._reset_statistics()
                    print("统计信息已重置")
                        
        except KeyboardInterrupt:
            print("\n用户中断检测")
            
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            
            # 显示最终统计
            print("\n=== 检测统计 ===")
            print(f"总帧数: {self.frame_count}")
            print(f"检测帧数: {self.detection_count}")
            print(f"检测率: {self.detection_count/max(self.frame_count,1)*100:.1f}%")
            print(f"平均FPS: {self.current_fps:.1f}")
            print("\n行为统计:")
            for behavior, count in self.behavior_counts.items():
                percentage = (count / max(self.frame_count, 1)) * 100
                print(f"  {behavior}: {count} ({percentage:.1f}%)")
                
    def _reset_statistics(self) -> None:
        """重置统计信息"""
        self.frame_count = 0
        self.detection_count = 0
        self.behavior_counts = {'handwash': 0, 'sanitize': 0, 'none': 0}
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0


def main():
    """主函数"""
    print("=== 视频实时显示检测演示 ===")
    print("专门用于显示视频检测过程的可视化效果")
    print("确保您能看到实时的手部检测和行为识别结果")
    
    # 设置日志
    logging.basicConfig(
        level=logging.WARNING,  # 减少日志输出
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建检测器
        detector = VideoDisplayDetector()
        
        # 查找可用的测试视频
        test_videos = [
            "test_video.mp4",
            "demo_video.mp4", 
            "sample_video.mp4",
            "handwash_demo.mp4",
            "tests/fixtures/videos/20250724072708.mp4",
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
                print("请确保项目目录中有测试视频文件")
                return
        
        print(f"\n使用视频文件: {video_path}")
        print("\n控制键:")
        print("- 'q': 退出")
        print("- 空格: 暂停/继续")
        print("- 's': 保存当前帧")
        print("- 'r': 重置统计")
        print("\n按任意键开始...")
        input()
        
        # 处理视频
        detector.process_video_file(video_path)
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n感谢使用视频显示检测系统!")


if __name__ == '__main__':
    main()