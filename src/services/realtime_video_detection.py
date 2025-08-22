#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时视频检测服务

提供实时视频检测和可视化功能，支持：
- 实时手部检测和关键点显示
- 行为识别结果可视化
- 运动轨迹显示
- 检测统计信息实时更新

Author: Trae AI Assistant
Date: 2025-01-21
"""

import cv2
import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass

# 本地模块导入
from src.core.enhanced_hand_detector import EnhancedHandDetector, DetectionMode
from src.core.enhanced_motion_analyzer import EnhancedMotionAnalyzer
from src.core.deep_behavior_recognizer import DeepBehaviorRecognizer
from src.core.personalization_engine import PersonalizationEngine
from src.core.performance_optimizer import PerformanceOptimizer
from src.utils.logger import get_logger


@dataclass
class VisualizationConfig:
    """可视化配置"""
    show_landmarks: bool = True
    show_bbox: bool = True
    show_trajectory: bool = True
    show_behavior_text: bool = True
    show_statistics: bool = True
    trajectory_length: int = 30
    font_scale: float = 0.6
    line_thickness: int = 2
    

class RealtimeVideoDetector:
    """实时视频检测器"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 visualization_config: Optional[VisualizationConfig] = None):
        """
        初始化实时视频检测器
        
        Args:
            config_path: 配置文件路径
            visualization_config: 可视化配置
        """
        self.logger = get_logger(__name__)
        self.config = visualization_config or VisualizationConfig()
        
        # 初始化核心组件
        self._initialize_components()
        
        # 可视化相关
        self.trajectory_points: Dict[int, List[Tuple[int, int]]] = {}
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
            'trajectory': (0, 255, 255),    # 黄色
            'handwash': (0, 255, 0),        # 绿色
            'sanitize': (255, 0, 255),      # 紫色
            'none': (128, 128, 128),        # 灰色
            'text': (255, 255, 255),        # 白色
            'background': (0, 0, 0)         # 黑色
        }
        
    def _initialize_components(self) -> None:
        """初始化核心组件"""
        try:
            # 增强手部检测器
            self.hand_detector = EnhancedHandDetector(
                detection_mode=DetectionMode.WITH_FALLBACK,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                quality_threshold=0.3
            )
            self.logger.info("✓ 增强手部检测器初始化完成")
            
            # 增强运动分析器
            self.motion_analyzer = EnhancedMotionAnalyzer()
            self.logger.info("✓ 增强运动分析器初始化完成")
            
            # 深度学习行为识别器（可选）
            try:
                self.deep_recognizer = DeepBehaviorRecognizer()
                self.logger.info("✓ 深度学习行为识别器初始化完成")
            except Exception as e:
                self.logger.warning(f"深度学习行为识别器初始化失败: {e}")
                self.deep_recognizer = None
                
            # 个性化引擎（可选）
            try:
                self.personalization = PersonalizationEngine()
                self.logger.info("✓ 个性化引擎初始化完成")
            except Exception as e:
                self.logger.warning(f"个性化引擎初始化失败: {e}")
                self.personalization = None
                
            # 性能优化器
            self.performance_optimizer = PerformanceOptimizer()
            self.logger.info("✓ 性能优化器初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Tuple[np.ndarray, Dict]: 处理后的可视化帧和检测结果
        """
        self.frame_count += 1
        start_time = time.time()
        
        try:
            # 1. 手部检测
            detection_results = self.hand_detector.detect_hands_robust(frame)
            
            if not detection_results:
                return self._draw_no_detection_info(frame), self._create_empty_result()
            
            self.detection_count += 1
            
            # 2. 转换检测结果
            track_id = 1  # 简化处理，使用固定track_id
            hands_data = [{
                'label': result.hand_label,
                'landmarks': result.landmarks,
                'bbox': result.bbox,
                'confidence': result.confidence
            } for result in detection_results]
            
            # 3. 更新运动分析
            self.motion_analyzer.update_hand_motion(track_id, hands_data)
            
            # 4. 行为分析
            handwash_confidence = self.motion_analyzer.analyze_handwashing_enhanced(track_id)
            sanitize_confidence = self.motion_analyzer.analyze_sanitizing_enhanced(track_id)
            
            # 5. 深度学习行为识别（可选）
            deep_predictions = {'handwash': 0.0, 'sanitize': 0.0, 'none': 1.0}
            if self.deep_recognizer:
                motion_summary = self.motion_analyzer.get_enhanced_motion_summary(track_id)
                if motion_summary:
                    self.deep_recognizer.update_features(motion_summary)
                    deep_predictions = self.deep_recognizer.predict_behavior()
            
            # 6. 确定最终行为
            final_behavior, final_confidence = self._determine_final_behavior(
                handwash_confidence, sanitize_confidence, deep_predictions
            )
            
            # 7. 更新统计
            self.behavior_counts[final_behavior] += 1
            
            # 8. 可视化
            vis_frame = self._visualize_detection(
                frame, detection_results, final_behavior, final_confidence,
                handwash_confidence, sanitize_confidence, deep_predictions
            )
            
            # 9. 创建结果
            result = {
                'frame_id': self.frame_count,
                'detection_count': len(detection_results),
                'behavior': final_behavior,
                'confidence': final_confidence,
                'handwash_confidence': handwash_confidence,
                'sanitize_confidence': sanitize_confidence,
                'deep_predictions': deep_predictions,
                'processing_time': time.time() - start_time,
                'hands_data': hands_data
            }
            
            return vis_frame, result
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            return self._draw_error_info(frame, str(e)), self._create_empty_result()
            
    def _determine_final_behavior(self, 
                                handwash_conf: float, 
                                sanitize_conf: float,
                                deep_preds: Dict[str, float]) -> Tuple[str, float]:
        """确定最终行为"""
        # 传统方法权重
        traditional_weight = 0.6
        deep_weight = 0.4
        
        # 融合预测
        final_handwash = traditional_weight * handwash_conf + deep_weight * deep_preds.get('handwash', 0.0)
        final_sanitize = traditional_weight * sanitize_conf + deep_weight * deep_preds.get('sanitize', 0.0)
        
        if final_handwash > 0.6:
            return 'handwash', final_handwash
        elif final_sanitize > 0.6:
            return 'sanitize', final_sanitize
        else:
            return 'none', max(final_handwash, final_sanitize)
            
    def _visualize_detection(self, 
                           frame: np.ndarray,
                           detection_results: List[Any],
                           behavior: str,
                           confidence: float,
                           handwash_conf: float,
                           sanitize_conf: float,
                           deep_preds: Dict[str, float]) -> np.ndarray:
        """可视化检测结果"""
        vis_frame = frame.copy()
        
        for i, result in enumerate(detection_results):
            # 绘制边界框
            if self.config.show_bbox and result.bbox:
                x1, y1, x2, y2 = result.bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), 
                            self.colors['hand_bbox'], self.config.line_thickness)
                
                # 绘制置信度
                conf_text = f"{result.hand_label}: {result.confidence:.2f}"
                cv2.putText(vis_frame, conf_text, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                          self.colors['text'], 1)
            
            # 绘制关键点
            if self.config.show_landmarks and result.landmarks:
                self._draw_hand_landmarks(vis_frame, result.landmarks)
                
            # 更新轨迹
            if self.config.show_trajectory and result.bbox:
                self._update_trajectory(i, result.bbox)
                self._draw_trajectory(vis_frame, i)
        
        # 绘制行为识别结果
        if self.config.show_behavior_text:
            self._draw_behavior_info(vis_frame, behavior, confidence, 
                                   handwash_conf, sanitize_conf, deep_preds)
        
        # 绘制统计信息
        if self.config.show_statistics:
            self._draw_statistics(vis_frame)
            
        return vis_frame
        
    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float]]) -> None:
        """绘制手部关键点"""
        h, w = frame.shape[:2]
        
        # 绘制关键点
        for landmark in landmarks:
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(frame, (x, y), 3, self.colors['landmarks'], -1)
            
        # 绘制连接线（简化版）
        if len(landmarks) >= 21:  # MediaPipe手部有21个关键点
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
                (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
                (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
                (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            ]
            
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (int(landmarks[start_idx][0] * w), 
                                 int(landmarks[start_idx][1] * h))
                    end_point = (int(landmarks[end_idx][0] * w), 
                               int(landmarks[end_idx][1] * h))
                    cv2.line(frame, start_point, end_point, 
                           self.colors['landmarks'], 1)
                           
    def _update_trajectory(self, track_id: int, bbox: List[int]) -> None:
        """更新轨迹点"""
        if track_id not in self.trajectory_points:
            self.trajectory_points[track_id] = []
            
        # 计算中心点
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # 添加新点
        self.trajectory_points[track_id].append(center)
        
        # 限制轨迹长度
        if len(self.trajectory_points[track_id]) > self.config.trajectory_length:
            self.trajectory_points[track_id].pop(0)
            
    def _draw_trajectory(self, frame: np.ndarray, track_id: int) -> None:
        """绘制轨迹"""
        if track_id not in self.trajectory_points:
            return
            
        points = self.trajectory_points[track_id]
        if len(points) < 2:
            return
            
        # 绘制轨迹线
        for i in range(1, len(points)):
            # 渐变透明度
            alpha = i / len(points)
            color = tuple(int(c * alpha) for c in self.colors['trajectory'])
            cv2.line(frame, points[i-1], points[i], color, 2)
            
    def _draw_behavior_info(self, 
                          frame: np.ndarray,
                          behavior: str,
                          confidence: float,
                          handwash_conf: float,
                          sanitize_conf: float,
                          deep_preds: Dict[str, float]) -> None:
        """绘制行为识别信息"""
        h, w = frame.shape[:2]
        
        # 主要行为结果
        behavior_text = f"行为: {behavior} ({confidence:.2f})"
        behavior_color = self.colors.get(behavior, self.colors['text'])
        
        # 绘制背景
        cv2.rectangle(frame, (10, 10), (400, 120), self.colors['background'], -1)
        cv2.rectangle(frame, (10, 10), (400, 120), self.colors['text'], 1)
        
        # 绘制文本
        y_offset = 30
        cv2.putText(frame, behavior_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   behavior_color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"洗手: {handwash_conf:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   self.colors['handwash'], 1)
        
        y_offset += 20
        cv2.putText(frame, f"消毒: {sanitize_conf:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   self.colors['sanitize'], 1)
        
        y_offset += 20
        if deep_preds:
            deep_text = f"深度学习: H:{deep_preds.get('handwash', 0):.2f} S:{deep_preds.get('sanitize', 0):.2f}"
            cv2.putText(frame, deep_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale * 0.8,
                       self.colors['text'], 1)
                       
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
        stats_x = w - 250
        cv2.rectangle(frame, (stats_x, 10), (w - 10, 150), self.colors['background'], -1)
        cv2.rectangle(frame, (stats_x, 10), (w - 10, 150), self.colors['text'], 1)
        
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
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale * 0.8,
                       self.colors['text'], 1)
            y_offset += 20
            
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
        if self.config.show_statistics:
            self._draw_statistics(vis_frame)
            
        return vis_frame
        
    def _draw_error_info(self, frame: np.ndarray, error_msg: str) -> np.ndarray:
        """绘制错误信息"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # 绘制错误信息
        text = f"处理错误: {error_msg}"
        cv2.putText(vis_frame, text, (20, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   (0, 0, 255), 2)  # 红色
        
        return vis_frame
        
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'frame_id': self.frame_count,
            'detection_count': 0,
            'behavior': 'none',
            'confidence': 0.0,
            'handwash_confidence': 0.0,
            'sanitize_confidence': 0.0,
            'deep_predictions': {'handwash': 0.0, 'sanitize': 0.0, 'none': 1.0},
            'processing_time': 0.0,
            'hands_data': []
        }
        
    def process_video(self, 
                     video_path: str,
                     output_path: Optional[str] = None,
                     display: bool = True) -> Dict[str, Any]:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            display: 是否显示实时画面
            
        Returns:
            Dict: 处理统计结果
        """
        self.logger.info(f"开始处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # 处理统计
        results = []
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 处理帧
                vis_frame, result = self.process_frame(frame)
                results.append(result)
                
                # 保存帧
                if writer:
                    writer.write(vis_frame)
                    
                # 显示帧
                if display:
                    cv2.imshow('实时手部行为检测', vis_frame)
                    
                    # 按键控制
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # 退出
                        break
                    elif key == ord('p'):  # 暂停
                        cv2.waitKey(0)
                    elif key == ord('s'):  # 保存当前帧
                        cv2.imwrite(f'frame_{self.frame_count}.jpg', vis_frame)
                        self.logger.info(f"保存帧: frame_{self.frame_count}.jpg")
                        
                # 进度显示
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    self.logger.info(f"处理进度: {progress:.1f}% ({self.frame_count}/{total_frames})")
                    
        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
            
        finally:
            # 清理资源
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
                
        # 计算统计结果
        processing_time = time.time() - start_time
        avg_fps = self.frame_count / processing_time if processing_time > 0 else 0
        
        statistics = {
            'total_frames': self.frame_count,
            'detection_count': self.detection_count,
            'behavior_counts': self.behavior_counts.copy(),
            'processing_time': processing_time,
            'avg_fps': avg_fps,
            'detection_rate': self.detection_count / max(self.frame_count, 1),
            'results': results
        }
        
        self.logger.info(f"视频处理完成: {self.frame_count}帧, 平均FPS: {avg_fps:.1f}")
        return statistics
        
    def process_camera(self, camera_id: int = 0) -> None:
        """
        处理摄像头实时流
        
        Args:
            camera_id: 摄像头ID
        """
        self.logger.info(f"开始摄像头实时检测: {camera_id}")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("无法读取摄像头帧")
                    continue
                    
                # 处理帧
                vis_frame, result = self.process_frame(frame)
                
                # 显示帧
                cv2.imshow('实时手部行为检测', vis_frame)
                
                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 退出
                    break
                elif key == ord('s'):  # 保存当前帧
                    timestamp = int(time.time())
                    filename = f'camera_frame_{timestamp}.jpg'
                    cv2.imwrite(filename, vis_frame)
                    self.logger.info(f"保存帧: {filename}")
                elif key == ord('r'):  # 重置统计
                    self._reset_statistics()
                    self.logger.info("统计信息已重置")
                    
        except KeyboardInterrupt:
            self.logger.info("用户中断摄像头检测")
            
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            
    def _reset_statistics(self) -> None:
        """重置统计信息"""
        self.frame_count = 0
        self.detection_count = 0
        self.behavior_counts = {'handwash': 0, 'sanitize': 0, 'none': 0}
        self.trajectory_points.clear()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if hasattr(self, 'performance_optimizer'):
                self.performance_optimizer.cleanup()
            if hasattr(self, 'motion_analyzer'):
                self.motion_analyzer.cleanup()
            cv2.destroyAllWindows()
            self.logger.info("实时检测器资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实时视频手部行为检测')
    parser.add_argument('--input', '-i', type=str, help='输入视频路径（不指定则使用摄像头）')
    parser.add_argument('--output', '-o', type=str, help='输出视频路径')
    parser.add_argument('--camera', '-c', type=int, default=0, help='摄像头ID')
    parser.add_argument('--no-display', action='store_true', help='不显示实时画面')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = RealtimeVideoDetector()
    
    try:
        if args.input:
            # 处理视频文件
            statistics = detector.process_video(
                args.input, 
                args.output, 
                display=not args.no_display
            )
            print("\n=== 处理统计 ===")
            print(json.dumps(statistics, indent=2, ensure_ascii=False))
        else:
            # 处理摄像头
            detector.process_camera(args.camera)
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        detector.cleanup()


if __name__ == '__main__':
    main()