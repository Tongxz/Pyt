#!/usr/bin/env python3
"""
自动启动视频检测演示
专门用于自动显示视频检测过程的可视化效果
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# 导入项目模块
from src.core.enhanced_hand_detector import EnhancedHandDetector, DetectionMode
from src.core.motion_analyzer import MotionAnalyzer
from src.utils.logger import get_logger

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

class AutoVideoDisplayDetector:
    """自动视频显示检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.hand_detector = EnhancedHandDetector(detection_mode=DetectionMode.WITH_FALLBACK)
        self.motion_analyzer = MotionAnalyzer()
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'hands_detected': 0,
            'behaviors_detected': 0,
            'processing_time': 0.0
        }
        
        # 可视化设置
        self.colors = {
            'hand_box': (0, 255, 0),      # 绿色
            'keypoints': (255, 0, 0),     # 蓝色
            'behavior': (0, 0, 255),      # 红色
            'text': (255, 255, 255),      # 白色
            'background': (0, 0, 0)       # 黑色
        }
        
        logger.info("✓ 自动视频检测器初始化完成")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """处理单帧图像"""
        start_time = time.time()
        
        # 手部检测
        hand_results = self.hand_detector.detect_hands_robust(frame)
        
        # 运动分析
        track_id = 1  # 使用固定的track_id
        motion_results = {}
        
        if hand_results:
            for hand in hand_results:
                if hasattr(hand, 'landmarks') and hand.landmarks:
                    # 更新运动跟踪
                    self.motion_analyzer.update_hand_motion(
                        hand.landmarks, 
                        hand.hand_label if hasattr(hand, 'hand_label') else 'unknown'
                    )
                    
                    # 分析洗手行为
                    handwash_confidence = self.motion_analyzer.analyze_handwashing()
                    
                    # 分析消毒行为
                    sanitize_confidence = self.motion_analyzer.analyze_sanitizing()
                    
                    # 确定行为类型和置信度
                    if handwash_confidence > sanitize_confidence:
                        behavior_type = "handwashing"
                        confidence = handwash_confidence
                    else:
                        behavior_type = "sanitizing"
                        confidence = sanitize_confidence
                    
                    motion_results = {
                        'behavior_type': behavior_type,
                        'behavior_confidence': confidence,
                        'handwash_confidence': handwash_confidence,
                        'sanitize_confidence': sanitize_confidence
                    }
                    break  # 只处理第一个检测到的手部
        
        # 更新统计
        self.stats['total_frames'] += 1
        if hand_results:
            self.stats['hands_detected'] += 1
        if motion_results and motion_results.get('behavior_confidence', 0) > 0.5:
            self.stats['behaviors_detected'] += 1
        
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
        """绘制可视化信息"""
        height, width = frame.shape[:2]
        
        # 绘制手部检测结果
        if hand_results:
            for hand in hand_results:
                # 绘制边界框
                if hasattr(hand, 'bbox') and hand.bbox:
                    x1, y1, x2, y2 = hand.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['hand_box'], 2)
                    
                    # 绘制置信度
                    confidence = hand.confidence if hasattr(hand, 'confidence') else 0.0
                    cv2.putText(frame, f'Hand: {confidence:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              self.colors['hand_box'], 2)
                
                # 绘制关键点
                if hasattr(hand, 'landmarks') and hand.landmarks:
                    # MediaPipe landmarks需要特殊处理
                    if hasattr(hand.landmarks, 'landmark'):
                        for landmark in hand.landmarks.landmark:
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            cv2.circle(frame, (x, y), 3, self.colors['keypoints'], -1)
        
        # 绘制行为识别结果
        if motion_results:
            behavior = motion_results.get('behavior_type', 'unknown')
            confidence = motion_results.get('behavior_confidence', 0.0)
            
            if confidence > 0.3:  # 只显示置信度较高的结果
                text = f'Behavior: {behavior} ({confidence:.2f})'
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, self.colors['behavior'], 2)
        
        # 绘制统计信息
        self.draw_statistics(frame)
        
        # 绘制FPS
        if self.stats['total_frames'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['total_frames']
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f'FPS: {fps:.1f}', (width-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray):
        """绘制统计信息"""
        height, width = frame.shape[:2]
        
        # 统计信息文本
        stats_text = [
            f"Frames: {self.stats['total_frames']}",
            f"Hands: {self.stats['hands_detected']}",
            f"Behaviors: {self.stats['behaviors_detected']}"
        ]
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height-100), (200, height-10), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制统计文本
        for i, text in enumerate(stats_text):
            y_pos = height - 80 + i * 25
            cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, self.colors['text'], 1)

def find_test_video() -> Optional[str]:
    """查找测试视频文件"""
    video_paths = [
        "tests/fixtures/videos/20250724072708.mp4",
        "tests/fixtures/videos/test_video.mp4",
        "data/test_video.mp4",
        "test_video.mp4"
    ]
    
    for path in video_paths:
        if Path(path).exists():
            logger.info(f"找到测试视频: {path}")
            return path
    
    logger.warning("未找到测试视频文件")
    return None

def main():
    """主函数"""
    print("\n=== 自动视频检测演示 ===")
    print("自动显示视频检测过程的可视化效果")
    print("按 'q' 键退出，空格键暂停/继续\n")
    
    # 查找测试视频
    video_path = find_test_video()
    if not video_path:
        print("错误: 未找到测试视频文件")
        return
    
    print(f"使用视频文件: {video_path}")
    
    # 初始化检测器
    try:
        detector = AutoVideoDisplayDetector()
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        return
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
    
    # 创建窗口
    window_name = "手部行为检测 - 实时可视化"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    paused = False
    frame_delay = max(1, int(1000 / fps))  # 毫秒
    
    print("\n开始处理视频...")
    print("控制键: 'q'=退出, 空格=暂停/继续, 's'=保存帧")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logger.info("视频播放完毕，重新开始")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重新开始
                    continue
                
                # 处理帧
                vis_frame, results = detector.process_frame(frame)
                
                # 显示结果
                cv2.imshow(window_name, vis_frame)
            
            # 处理按键
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键
                paused = not paused
                status = "暂停" if paused else "继续"
                logger.info(f"视频 {status}")
            elif key == ord('s'):  # 保存当前帧
                if 'vis_frame' in locals():
                    timestamp = int(time.time())
                    filename = f"saved_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    logger.info(f"保存帧: {filename}")
    
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 显示最终统计
        print("\n=== 处理统计 ===")
        print(f"总帧数: {detector.stats['total_frames']}")
        print(f"检测到手部: {detector.stats['hands_detected']}")
        print(f"识别到行为: {detector.stats['behaviors_detected']}")
        if detector.stats['total_frames'] > 0:
            avg_time = detector.stats['processing_time'] / detector.stats['total_frames']
            print(f"平均处理时间: {avg_time:.3f}秒/帧")
            print(f"平均FPS: {1.0/avg_time:.1f}")
        
        logger.info("视频检测演示结束")

if __name__ == "__main__":
    main()