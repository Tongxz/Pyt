#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头实时检测演示

专门用于摄像头实时显示的简化版本，确保在Windows环境下能正常显示视频窗口。

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


class SimpleCameraDetector:
    """简化的摄像头检测器"""
    
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
        
        # 颜色定义
        self.colors = {
            'hand_bbox': (0, 255, 0),      # 绿色
            'landmarks': (255, 0, 0),       # 蓝色
            'handwash': (0, 255, 0),        # 绿色
            'sanitize': (255, 0, 255),      # 紫色
            'none': (128, 128, 128),        # 灰色
            'text': (255, 255, 255),        # 白色
            'background': (0, 0, 0)         # 黑色
        }
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理单帧图像"""
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
        cv2.rectangle(frame, (10, 10), (350, 100), self.colors['background'], -1)
        cv2.rectangle(frame, (10, 10), (350, 100), self.colors['text'], 1)
        
        # 绘制文本
        y_offset = 30
        cv2.putText(frame, behavior_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   behavior_color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"洗手: {handwash_conf:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.colors['handwash'], 1)
        
        y_offset += 20
        cv2.putText(frame, f"消毒: {sanitize_conf:.2f}", (20, y_offset),
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
        stats_x = w - 200
        cv2.rectangle(frame, (stats_x, 10), (w - 10, 120), self.colors['background'], -1)
        cv2.rectangle(frame, (stats_x, 10), (w - 10, 120), self.colors['text'], 1)
        
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
            y_offset += 15
            
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
        
        # 绘制统计信息
        self._draw_statistics(vis_frame)
        
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
        
    def run_camera(self, camera_id: int = 0) -> None:
        """运行摄像头检测"""
        print(f"\n开始摄像头实时检测 (ID: {camera_id})...")
        print("控制键:")
        print("- 'q': 退出")
        print("- 's': 保存当前帧")
        print("- 'r': 重置统计")
        print("- 'f': 切换全屏")
        print("\n按任意键开始...")
        input()
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {camera_id}")
            return
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 获取实际参数
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"摄像头参数: {actual_width}x{actual_height}, {actual_fps:.1f}FPS")
        
        # 创建窗口
        window_name = '实时手部行为检测'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        fullscreen = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("警告: 无法读取摄像头帧")
                    continue
                    
                # 处理帧
                vis_frame, result = self.process_frame(frame)
                
                # 显示帧
                cv2.imshow(window_name, vis_frame)
                
                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 退出
                    break
                elif key == ord('s'):  # 保存当前帧
                    timestamp = int(time.time())
                    filename = f'camera_frame_{timestamp}.jpg'
                    cv2.imwrite(filename, vis_frame)
                    print(f"保存帧: {filename}")
                elif key == ord('r'):  # 重置统计
                    self._reset_statistics()
                    print("统计信息已重置")
                elif key == ord('f'):  # 切换全屏
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        
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
    print("=== 摄像头实时手部行为检测 ===")
    print("这是一个简化版本，专门用于摄像头实时显示")
    print("确保您的摄像头已连接并可用")
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建检测器
        detector = SimpleCameraDetector()
        
        # 选择摄像头
        camera_id = input("\n请输入摄像头ID (默认0): ").strip()
        camera_id = int(camera_id) if camera_id.isdigit() else 0
        
        # 运行检测
        detector.run_camera(camera_id)
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n感谢使用摄像头实时检测系统!")


if __name__ == '__main__':
    main()