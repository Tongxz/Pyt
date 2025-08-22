#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的可视化检测测试脚本
直接运行检测器测试，无需用户交互
"""

import cv2
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Any, Optional

try:
    from src.core.enhanced_hand_detector import EnhancedHandDetector
    from src.core.pose_detector import PoseDetector
    from src.core.motion_analyzer import MotionAnalyzer
    from src.utils.visualization import VisualizationManager
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖模块都已正确安装")
    exit(1)


class SimpleDetectionTest:
    """
    简单的检测测试类
    """
    
    def __init__(self):
        """
        初始化测试系统
        """
        self.logger = get_logger(__name__)
        
        # 初始化检测器
        try:
            self.hand_detector = EnhancedHandDetector()
            self.logger.info("✓ 增强手部检测器初始化成功")
        except Exception as e:
            self.logger.error(f"手部检测器初始化失败: {e}")
            self.hand_detector = None
        
        try:
            self.motion_analyzer = MotionAnalyzer()
            self.logger.info("✓ 运动分析器初始化成功")
        except Exception as e:
            self.logger.error(f"运动分析器初始化失败: {e}")
            self.motion_analyzer = None
        
        # 初始化可视化管理器
        try:
            self.visualization_manager = VisualizationManager({
                'show_landmarks': True,
                'show_bbox': True,
                'show_behavior_text': True,
                'show_statistics': True,
                'show_trajectory': True,
                'font_size': 24,
                'line_thickness': 2,
                'trajectory_length': 30
            })
            self.logger.info("✓ 可视化管理器初始化成功")
        except Exception as e:
            self.logger.error(f"可视化管理器初始化失败: {e}")
            self.visualization_manager = None
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'handwash_detections': 0,
            'sanitize_detections': 0,
            'processing_times': [],
            'hand_counts': []
        }
        
        self.logger.info("简单检测测试系统初始化完成")
    
    def find_test_video(self) -> Optional[Path]:
        """
        查找测试视频文件
        
        Returns:
            视频文件路径或None
        """
        video_dirs = [
            Path("tests/fixtures/videos"),
            Path("test_videos"),
            Path("videos"),
            Path("data/videos")
        ]
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        for video_dir in video_dirs:
            if video_dir.exists():
                for ext in video_extensions:
                    video_files = list(video_dir.glob(f'*{ext}'))
                    if video_files:
                        self.logger.info(f"找到测试视频: {video_files[0]}")
                        return video_files[0]
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        results = {
            'hand_results': [],
            'motion_results': {},
            'processing_time': 0.0
        }
        
        try:
            # 手部检测
            if self.hand_detector:
                hand_results = self.hand_detector.detect_hands_robust(frame)
                results['hand_results'] = hand_results
                
                # 运动分析
                if self.motion_analyzer and hand_results:
                    # 使用固定的track_id进行运动分析
                    track_id = 1
                    
                    # 转换HandDetectionResult对象为字典格式
                    hands_data = []
                    for hand in hand_results:
                        if hasattr(hand, 'bbox'):
                            hand_dict = {
                                'bbox': hand.bbox,
                                'confidence': getattr(hand, 'confidence', 0.0),
                                'label': getattr(hand, 'hand_label', 'unknown'),
                                'landmarks': getattr(hand, 'landmarks', None)
                            }
                            hands_data.append(hand_dict)
                    
                    # 更新手部运动数据
                    if hands_data:
                        self.motion_analyzer.update_hand_motion(track_id, hands_data)
                        
                        # 分析洗手行为
                        handwash_conf = self.motion_analyzer.analyze_handwashing(track_id)
                        
                        # 分析消毒行为
                        sanitize_conf = self.motion_analyzer.analyze_sanitizing(track_id)
                    else:
                        handwash_conf = 0.0
                        sanitize_conf = 0.0
                    
                    # 确定主要行为
                    if handwash_conf > sanitize_conf and handwash_conf > 0.3:
                        behavior_type = 'handwashing'
                        behavior_confidence = handwash_conf
                        self.stats['handwash_detections'] += 1
                    elif sanitize_conf > 0.3:
                        behavior_type = 'sanitizing'
                        behavior_confidence = sanitize_conf
                        self.stats['sanitize_detections'] += 1
                    else:
                        behavior_type = 'none'
                        behavior_confidence = 0.0
                    
                    results['motion_results'] = {
                        'behavior_type': behavior_type,
                        'behavior_confidence': behavior_confidence,
                        'handwash_confidence': handwash_conf,
                        'sanitize_confidence': sanitize_conf
                    }
        
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
        
        # 记录处理时间
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.stats['processing_times'].append(processing_time)
        self.stats['hand_counts'].append(len(results['hand_results']))
        
        return results
    
    def run_test(self):
        """
        运行简单测试
        """
        print("=== 简单检测测试 ===")
        
        # 查找测试视频
        video_path = self.find_test_video()
        
        if video_path is None:
            self.logger.error("未找到测试视频文件")
            print("未找到测试视频文件，请确保 tests/fixtures/videos/ 目录下有视频文件")
            return
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"无法打开视频文件: {video_path}")
            return
        
        print(f"正在处理视频: {video_path.name}")
        print("按 'q' 键退出，按 's' 键保存当前帧")
        
        frame_count = 0
        max_frames = 300  # 限制处理帧数
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            
            frame_count += 1
            self.stats['total_frames'] = frame_count
            
            # 处理帧
            results = self.process_frame(frame)
            
            # 可视化
            if self.visualization_manager:
                try:
                    # 准备可视化数据
                    viz_data = {
                        'hand_results': results['hand_results'],
                        'motion_results': results['motion_results'],
                        'frame_info': {
                            'frame_number': frame_count,
                            'processing_time': results['processing_time']
                        },
                        'statistics': self.stats
                    }
                    
                    # 绘制可视化
                    vis_frame = self.visualization_manager.visualize_frame(
                        frame.copy(), viz_data, frame_count
                    )
                    
                    # 显示结果
                    cv2.imshow('检测结果', vis_frame)
                    
                except Exception as e:
                    self.logger.error(f"可视化失败: {e}")
                    cv2.imshow('原始视频', frame)
            else:
                cv2.imshow('原始视频', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                save_path = f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"保存帧: {save_path}")
            
            # 每50帧输出一次统计信息
            if frame_count % 50 == 0:
                avg_time = np.mean(self.stats['processing_times'][-50:]) if self.stats['processing_times'] else 0
                print(f"帧 {frame_count}: 平均处理时间 {avg_time:.3f}s, "
                      f"洗手检测 {self.stats['handwash_detections']}, "
                      f"消毒检测 {self.stats['sanitize_detections']}")
        
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 输出最终统计
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """
        输出最终统计信息
        """
        print("\n=== 最终统计信息 ===")
        print(f"总处理帧数: {self.stats['total_frames']}")
        print(f"洗手行为检测次数: {self.stats['handwash_detections']}")
        print(f"消毒行为检测次数: {self.stats['sanitize_detections']}")
        
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            max_time = np.max(self.stats['processing_times'])
            min_time = np.min(self.stats['processing_times'])
            print(f"平均处理时间: {avg_time:.3f}s")
            print(f"最大处理时间: {max_time:.3f}s")
            print(f"最小处理时间: {min_time:.3f}s")
            print(f"平均FPS: {1.0/avg_time:.1f}")
        
        if self.stats['hand_counts']:
            avg_hands = np.mean(self.stats['hand_counts'])
            print(f"平均检测到的手部数量: {avg_hands:.1f}")


def main():
    """
    主函数
    """
    try:
        test = SimpleDetectionTest()
        test.run_test()
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()