#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化手部检测演示脚本
专门用于展示实时手部检测和行为识别的可视化效果
"""

import cv2
import numpy as np
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Tuple

# 导入检测模块
try:
    from src.core.pose_detector import PoseDetectorFactory
    from src.core.motion_analyzer import MotionAnalyzer
    from src.utils.visualization import VisualizationManager
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("将使用基础OpenCV功能")
    PoseDetectorFactory = None
    MotionAnalyzer = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualDetectionDemo:
    """可视化检测演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        self.pose_detector = None
        self.motion_analyzer = None
        
        # 初始化检测器
        if PoseDetectorFactory:
            try:
                self.pose_detector = PoseDetectorFactory.create(backend='mediapipe')
                logger.info("✓ 姿态检测器初始化成功")
            except Exception as e:
                logger.error(f"姿态检测器初始化失败: {e}")
        
        if MotionAnalyzer:
            try:
                self.motion_analyzer = MotionAnalyzer()
                logger.info("✓ 运动分析器初始化成功")
            except Exception as e:
                logger.error(f"运动分析器初始化失败: {e}")
        
        # 初始化可视化管理器
        try:
            self.visualization_manager = VisualizationManager({
                'show_landmarks': True,
                'show_bbox': True,
                'show_behavior_text': True,
                'show_statistics': True,
                'show_trajectory': True,
                'font_size': 32
            })
            logger.info("✓ 可视化管理器初始化成功")
        except Exception as e:
            logger.error(f"可视化管理器初始化失败: {e}")
            self.visualization_manager = None
        
        # 可视化配置
        self.colors = {
            'hand_box': (0, 255, 0),      # 绿色边界框
            'keypoints': (255, 0, 0),     # 蓝色关键点
            'behavior': (0, 255, 255),    # 黄色行为文本
            'stats': (255, 255, 255),     # 白色统计信息
            'warning': (0, 0, 255)        # 红色警告
        }
        
        # 统计信息
        self.stats = {
            'frames_processed': 0,
            'hands_detected': 0,
            'behaviors_detected': 0,
            'processing_time': 0.0
        }
        
        # 行为历史（用于平滑显示）
        self.behavior_history = []
        self.max_history = 10
    
    def find_test_video(self) -> Optional[str]:
        """查找测试视频文件"""
        possible_paths = [
            "tests/fixtures/videos/20250724072708.mp4",
            "tests/fixtures/videos/test_video.mp4",
            "data/test_video.mp4"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"找到测试视频: {path}")
                return str(Path(path).absolute())
        
        logger.warning("未找到测试视频文件")
        return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """处理单帧图像"""
        start_time = time.time()
        self.stats['frames_processed'] += 1
        
        # 初始化结果
        hand_results = []
        motion_results = {}
        
        # 手部检测
        if self.pose_detector:
            try:
                hand_results = self.pose_detector.detect_hands(frame)
                if hand_results:
                    self.stats['hands_detected'] += len(hand_results)
            except Exception as e:
                logger.error(f"手部检测失败: {e}")
        
        # 运动分析
        if self.motion_analyzer and hand_results:
            try:
                # 更新运动数据
                for hand in hand_results:
                    if 'landmarks' in hand and hand['landmarks']:
                        self.motion_analyzer.update_hand_motion(
                            hand['landmarks'], 
                            hand.get('label', 'unknown')
                        )
                
                # 分析行为
                handwash_conf = self.motion_analyzer.analyze_handwashing()
                sanitize_conf = self.motion_analyzer.analyze_sanitizing()
                
                motion_results = {
                    'handwash_confidence': handwash_conf,
                    'sanitize_confidence': sanitize_conf,
                    'behavior_type': 'handwashing' if handwash_conf > sanitize_conf else 'sanitizing',
                    'behavior_confidence': max(handwash_conf, sanitize_conf)
                }
                
                # 更新行为历史
                if motion_results['behavior_confidence'] > 0.3:
                    self.behavior_history.append(motion_results['behavior_type'])
                    if len(self.behavior_history) > self.max_history:
                        self.behavior_history.pop(0)
                    self.stats['behaviors_detected'] += 1
                
            except Exception as e:
                logger.error(f"运动分析失败: {e}")
        
        # 计算处理时间
        processing_time = time.time() - start_time
        self.stats['processing_time'] += processing_time
        
        # 绘制可视化
        vis_frame = self.draw_visualizations(frame.copy(), hand_results, motion_results)
        
        return vis_frame, {
            'hand_results': hand_results,
            'motion_results': motion_results,
            'processing_time': processing_time
        }
    
    def draw_visualizations(self, frame: np.ndarray, hand_results: List, motion_results: Dict) -> np.ndarray:
        """使用统一可视化管理器绘制可视化信息"""
        if self.visualization_manager:
            # 准备结果数据
            results = {
                'hand_results': hand_results,
                'motion_results': motion_results,
                'processing_time': 0.0
            }
            
            # 使用统一可视化管理器
            vis_frame = self.visualization_manager.visualize_frame(frame, results, self.stats['frames_processed'])
            
            # 添加控制提示
            vis_frame = self.visualization_manager.draw_control_hints(vis_frame)
            
            return vis_frame
        else:
            # 回退到原始可视化方法
            return self._draw_fallback_visualizations(frame, hand_results, motion_results)
    
    def _draw_fallback_visualizations(self, frame: np.ndarray, hand_results: List, motion_results: Dict) -> np.ndarray:
        """回退可视化方法（当可视化管理器不可用时）"""
        height, width = frame.shape[:2]
        
        # 绘制手部检测结果
        if hand_results:
            for i, hand in enumerate(hand_results):
                # 绘制边界框
                if 'bbox' in hand and hand['bbox']:
                    x1, y1, x2, y2 = hand['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['hand_box'], 2)
                    
                    # 绘制置信度
                    confidence = hand.get('confidence', 0.0)
                    label = hand.get('label', f'Hand_{i}')
                    text = f'{label}: {confidence:.2f}'
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, self.colors['hand_box'], 2)
                
                # 绘制关键点
                if 'landmarks' in hand and hand['landmarks']:
                    landmarks = hand['landmarks']
                    for point in landmarks:
                        if len(point) >= 2:
                            x, y = int(point[0] * width), int(point[1] * height)
                            cv2.circle(frame, (x, y), 3, self.colors['keypoints'], -1)
        
        # 绘制行为识别结果
        if motion_results and motion_results.get('behavior_confidence', 0) > 0.3:
            behavior = motion_results.get('behavior_type', 'unknown')
            confidence = motion_results.get('behavior_confidence', 0.0)
            
            # 主要行为显示
            text = f'Behavior: {behavior} ({confidence:.2f})'
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8, self.colors['behavior'], 2)
            
            # 详细置信度
            handwash_conf = motion_results.get('handwash_confidence', 0.0)
            sanitize_conf = motion_results.get('sanitize_confidence', 0.0)
            
            cv2.putText(frame, f'Handwashing: {handwash_conf:.2f}', 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['stats'], 1)
            cv2.putText(frame, f'Sanitizing: {sanitize_conf:.2f}', 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['stats'], 1)
        
        # 绘制统计信息
        self.draw_statistics(frame)
        
        # 绘制行为历史
        if self.behavior_history:
            recent_behavior = max(set(self.behavior_history), key=self.behavior_history.count)
            cv2.putText(frame, f'Recent: {recent_behavior}', 
                      (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['behavior'], 2)
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray):
        """绘制统计信息"""
        height, width = frame.shape[:2]
        
        # 统计信息位置（右上角）
        x_offset = width - 250
        y_start = 30
        line_height = 25
        
        stats_text = [
            f"Frames: {self.stats['frames_processed']}",
            f"Hands: {self.stats['hands_detected']}",
            f"Behaviors: {self.stats['behaviors_detected']}",
            f"Avg FPS: {self.stats['frames_processed']/(self.stats['processing_time']+0.001):.1f}"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = y_start + i * line_height
            cv2.putText(frame, text, (x_offset, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, self.colors['stats'], 1)
    
    def run_demo(self):
        """运行演示"""
        logger.info("=== 可视化手部检测演示 ===")
        
        # 查找测试视频
        video_path = self.find_test_video()
        if not video_path:
            logger.error("无法找到测试视频")
            return
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return
        
        logger.info(f"开始处理视频: {video_path}")
        logger.info("控制键:")
        logger.info("  空格键: 暂停/继续")
        logger.info("  ESC键: 退出")
        logger.info("  'r'键: 重置统计")
        logger.info("  's'键: 保存当前帧")
        
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("视频播放完毕，重新开始...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    frame_count += 1
                    
                    # 处理帧
                    vis_frame, results = self.process_frame(frame)
                else:
                    # 暂停时显示最后一帧
                    if 'vis_frame' in locals():
                        # 添加暂停标识
                        temp_frame = vis_frame.copy()
                        cv2.putText(temp_frame, 'PAUSED', 
                                  (temp_frame.shape[1]//2-60, temp_frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.colors['warning'], 3)
                        cv2.imshow('Hand Detection Demo', temp_frame)
                
                # 显示结果
                if not paused and 'vis_frame' in locals():
                    cv2.imshow('Hand Detection Demo', vis_frame)
                
                # 处理按键
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC键
                    logger.info("用户退出演示")
                    break
                elif key == ord(' '):  # 空格键
                    paused = not paused
                    logger.info(f"演示 {'暂停' if paused else '继续'}")
                elif key == ord('r'):  # r键
                    self.reset_statistics()
                    logger.info("统计信息已重置")
                elif key == ord('s'):  # s键
                    if 'vis_frame' in locals():
                        filename = f"detection_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, vis_frame)
                        logger.info(f"当前帧已保存: {filename}")
        
        except KeyboardInterrupt:
            logger.info("检测到键盘中断")
        
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            
            # 显示最终统计
            logger.info("=== 演示统计 ===")
            logger.info(f"处理帧数: {self.stats['frames_processed']}")
            logger.info(f"检测到的手部: {self.stats['hands_detected']}")
            logger.info(f"识别的行为: {self.stats['behaviors_detected']}")
            if self.stats['processing_time'] > 0:
                avg_fps = self.stats['frames_processed'] / self.stats['processing_time']
                logger.info(f"平均FPS: {avg_fps:.2f}")
            logger.info("演示完成")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'frames_processed': 0,
            'hands_detected': 0,
            'behaviors_detected': 0,
            'processing_time': 0.0
        }
        self.behavior_history = []
        if self.visualization_manager:
            self.visualization_manager.reset_trajectories()

def main():
    """主函数"""
    demo = VisualDetectionDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()