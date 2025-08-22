#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一可视化演示脚本
展示新的可视化管理器功能，包括中文字体支持
"""

import cv2
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Any, Optional

try:
    from src.core.pose_detector import PoseDetectorFactory
    from src.core.motion_analyzer import MotionAnalyzer
    from src.utils.visualization import VisualizationManager
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖模块都已正确安装")
    exit(1)


class UnifiedVisualizationDemo:
    """
    统一可视化演示类
    展示新的可视化管理器的完整功能
    """
    
    def __init__(self):
        """
        初始化演示系统
        """
        self.logger = get_logger(__name__)
        
        # 初始化检测器
        try:
            # 使用工厂模式创建检测器，并指定yolov8后端
            self.hand_detector = PoseDetectorFactory.create(backend='yolov8')
            self.logger.info("✓ YOLOv8 姿态检测器初始化成功")
        except Exception as e:
            self.logger.error(f"手部检测器初始化失败: {e}")
            self.hand_detector = None
        
        try:
            self.motion_analyzer = MotionAnalyzer()
            self.logger.info("✓ 运动分析器初始化成功")
        except Exception as e:
            self.logger.error(f"运动分析器初始化失败: {e}")
            self.motion_analyzer = None
        
        # 初始化可视化管理器（支持中文）
        try:
            self.visualization_manager = VisualizationManager({
                'show_landmarks': True,
                'show_bbox': True,
                'show_behavior_text': True,
                'show_statistics': True,
                'show_trajectory': True,
                'font_size': 32,
                'line_thickness': 2,
                'trajectory_length': 50
            })
            self.logger.info("✓ 统一可视化管理器初始化成功（支持中文字体）")
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
        
        # 行为历史记录
        self.behavior_history = []
        
        self.logger.info("统一可视化演示系统初始化完成")
    
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
                hand_results = self.hand_detector.detect(frame)
                results['hand_results'] = hand_results
                
                # 运动分析
                if self.motion_analyzer and hand_results:
                    # 使用固定的track_id进行运动分析
                    track_id = 1
                    
                    # 更新手部运动数据
                    self.motion_analyzer.update_hand_motion(track_id, hand_results)
                    
                    # 分析洗手行为
                    handwash_conf = self.motion_analyzer.analyze_handwashing(track_id)
                    
                    # 分析消毒行为
                    sanitize_conf = self.motion_analyzer.analyze_sanitizing(track_id)
                    
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
                    
                    # 记录行为历史
                    if behavior_confidence > 0.3:
                        self.behavior_history.append({
                            'frame': self.stats['total_frames'],
                            'behavior': behavior_type,
                            'confidence': behavior_confidence
                        })
        
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
        
        # 记录处理时间
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.stats['processing_times'].append(processing_time)
        self.stats['hand_counts'].append(len(results['hand_results']))
        
        return results
    
    def run_video_demo(self, video_path: Optional[Path] = None):
        """
        运行视频演示
        
        Args:
            video_path: 视频文件路径，如果为None则自动查找
        """
        if video_path is None:
            video_path = self.find_test_video()
        
        if video_path is None or not video_path.exists():
            self.logger.error("未找到测试视频文件")
            return
        
        self.logger.info(f"开始处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"视频信息: {width}x{height}, {fps:.2f}FPS, {total_frames}帧")
        
        # 创建窗口
        window_name = "统一可视化演示 - 支持中文字体"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.info("视频播放完成")
                        break
                    
                    frame_count += 1
                    self.stats['total_frames'] = frame_count
                    
                    # 处理帧
                    results = self.process_frame(frame)
                    
                    # 可视化
                    if self.visualization_manager:
                        vis_frame = self.visualization_manager.visualize_frame(
                            frame, results, frame_count, total_frames
                        )
                        vis_frame = self.visualization_manager.draw_control_hints(vis_frame)
                    else:
                        vis_frame = frame.copy()
                        cv2.putText(vis_frame, "可视化管理器不可用", (20, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
                    # 显示帧
                    cv2.imshow(window_name, vis_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC键退出
                    self.logger.info("用户退出演示")
                    break
                elif key == ord(' '):  # 空格键暂停/继续
                    paused = not paused
                    self.logger.info(f"演示 {'暂停' if paused else '继续'}")
                elif key == ord('r') or key == ord('R'):  # R键重置统计
                    self.reset_statistics()
                    self.logger.info("统计信息已重置")
                elif key == ord('s') or key == ord('S'):  # S键保存当前帧
                    if 'vis_frame' in locals():
                        save_path = f"saved_frame_{frame_count:06d}.jpg"
                        cv2.imwrite(save_path, vis_frame)
                        self.logger.info(f"帧已保存: {save_path}")
                elif key == ord('t') or key == ord('T'):  # T键切换轨迹显示
                    if self.visualization_manager:
                        current_config = self.visualization_manager.config
                        current_config['show_trajectory'] = not current_config['show_trajectory']
                        self.visualization_manager.update_config(current_config)
                        self.logger.info(f"轨迹显示: {'开启' if current_config['show_trajectory'] else '关闭'}")
                elif key == ord('l') or key == ord('L'):  # L键切换关键点显示
                    if self.visualization_manager:
                        current_config = self.visualization_manager.config
                        current_config['show_landmarks'] = not current_config['show_landmarks']
                        self.visualization_manager.update_config(current_config)
                        self.logger.info(f"关键点显示: {'开启' if current_config['show_landmarks'] else '关闭'}")
        
        except KeyboardInterrupt:
            self.logger.info("用户中断演示")
        
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            
            # 输出最终统计
            self.print_final_statistics()
    
    def reset_statistics(self):
        """
        重置统计信息
        """
        self.stats = {
            'total_frames': 0,
            'handwash_detections': 0,
            'sanitize_detections': 0,
            'processing_times': [],
            'hand_counts': []
        }
        self.behavior_history = []
        
        if self.visualization_manager:
            self.visualization_manager.reset_trajectories()
        
        self.logger.info("统计信息和轨迹已重置")
    
    def print_final_statistics(self):
        """
        打印最终统计信息
        """
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            self.logger.info("=== 最终统计信息 ===")
            self.logger.info(f"总处理帧数: {self.stats['total_frames']}")
            self.logger.info(f"洗手检测次数: {self.stats['handwash_detections']}")
            self.logger.info(f"消毒检测次数: {self.stats['sanitize_detections']}")
            self.logger.info(f"平均处理时间: {avg_time*1000:.2f}ms")
            self.logger.info(f"平均FPS: {avg_fps:.2f}")
            
            if self.stats['hand_counts']:
                avg_hands = np.mean(self.stats['hand_counts'])
                self.logger.info(f"平均手部检测数量: {avg_hands:.2f}")
            
            if self.behavior_history:
                self.logger.info(f"行为检测历史记录: {len(self.behavior_history)}条")
                for record in self.behavior_history[-5:]:  # 显示最后5条记录
                    self.logger.info(f"  帧{record['frame']}: {record['behavior']} ({record['confidence']:.2f})")
    
    def run_camera_demo(self, camera_id: int = 0):
        """
        运行摄像头实时演示
        
        Args:
            camera_id: 摄像头ID
        """
        self.logger.info(f"启动摄像头演示 (ID: {camera_id})")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error(f"无法打开摄像头 (ID: {camera_id})")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 创建窗口
        window_name = "统一可视化演示 - 实时摄像头"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("无法读取摄像头帧")
                    break
                
                frame_count += 1
                self.stats['total_frames'] = frame_count
                
                # 处理帧
                results = self.process_frame(frame)
                
                # 可视化
                if self.visualization_manager:
                    vis_frame = self.visualization_manager.visualize_frame(
                        frame, results, frame_count
                    )
                    vis_frame = self.visualization_manager.draw_control_hints(vis_frame)
                else:
                    vis_frame = frame.copy()
                    cv2.putText(vis_frame, "可视化管理器不可用", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # 显示帧
                cv2.imshow(window_name, vis_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC键退出
                    break
                elif key == ord('r') or key == ord('R'):  # R键重置统计
                    self.reset_statistics()
                elif key == ord('s') or key == ord('S'):  # S键保存当前帧
                    save_path = f"camera_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(save_path, vis_frame)
                    self.logger.info(f"帧已保存: {save_path}")
        
        except KeyboardInterrupt:
            self.logger.info("用户中断摄像头演示")
        
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_statistics()


def main():
    """
    主函数
    """
    print("=== 统一可视化演示系统 ===")
    print("支持中文字体显示和完整的可视化功能")
    print()
    
    # 创建演示实例
    demo = UnifiedVisualizationDemo()
    
    # 选择演示模式
    print("请选择演示模式:")
    print("1. 视频文件演示")
    print("2. 实时摄像头演示")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == '1':
            # 视频演示
            video_path = input("请输入视频文件路径 (留空自动查找): ").strip()
            if video_path:
                video_path = Path(video_path)
                if not video_path.exists():
                    print(f"视频文件不存在: {video_path}")
                    return
            else:
                video_path = None
            
            demo.run_video_demo(video_path)
        
        elif choice == '2':
            # 摄像头演示
            camera_id = input("请输入摄像头ID (默认0): ").strip()
            camera_id = int(camera_id) if camera_id.isdigit() else 0
            
            demo.run_camera_demo(camera_id)
        
        else:
            print("无效选择，默认使用视频演示")
            demo.run_video_demo()
    
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序运行错误: {e}")
    
    print("演示结束")


if __name__ == "__main__":
    main()