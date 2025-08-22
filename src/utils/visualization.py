#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化管理器模块
统一管理所有可视化相关功能，支持中文字体显示
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class VisualizationManager:
    """
    统一的可视化管理器
    负责处理所有检测结果的可视化显示，支持中文字体
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化可视化管理器
        
        Args:
            config: 可视化配置字典
        """
        self.logger = get_logger(__name__)
        
        # 默认配置
        default_config = {
            'show_landmarks': True,
            'show_bbox': True,
            'show_behavior_text': True,
            'show_statistics': True,
            'show_trajectory': False,
            'font_size': 32,
            'line_thickness': 2,
            'trajectory_length': 30
        }
        
        self.config = {**default_config, **(config or {})}
        
        # 颜色配置
        self.colors = {
            'hand_bbox': (0, 255, 0),      # 绿色边界框
            'landmarks': (255, 0, 0),       # 蓝色关键点
            'trajectory': (0, 255, 255),    # 黄色轨迹
            'handwash': (0, 255, 0),        # 绿色洗手
            'sanitize': (255, 0, 255),      # 紫色消毒
            'none': (128, 128, 128),        # 灰色无行为
            'text': (255, 255, 255),        # 白色文字
            'background': (0, 0, 0),        # 黑色背景
            'warning': (0, 0, 255),         # 红色警告
            'success': (0, 255, 0),         # 绿色成功
            'info': (255, 255, 0)           # 黄色信息
        }
        
        # 轨迹点存储
        self.trajectory_points: Dict[int, List[Tuple[int, int]]] = {}
        
        # 中文字体缓存
        self._font_cache = {}
        
        self.logger.info("可视化管理器初始化完成")
    
    def get_chinese_font(self, font_size: int = 32) -> Optional[Any]:
        """
        获取中文字体，支持多种字体回退
        
        Args:
            font_size: 字体大小
            
        Returns:
            PIL字体对象或None
        """
        if not PIL_AVAILABLE:
            return None
            
        # 检查缓存
        if font_size in self._font_cache:
            return self._font_cache[font_size]
        
        font_paths = [
            # macOS 系统字体
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            # Linux 系统字体
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            # Windows 系统字体
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            # 通用字体
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/arial.ttf"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    self._font_cache[font_size] = font
                    return font
                except Exception:
                    continue
        
        # 如果都找不到，使用默认字体
        try:
            font = ImageFont.load_default()
            self._font_cache[font_size] = font
            return font
        except Exception:
            return None
    
    def cv2_add_chinese_text(self, img: np.ndarray, text: str, position: Tuple[int, int],
                           font_size: int = 32, color: Tuple[int, int, int] = (255, 255, 255),
                           thickness: int = 2) -> np.ndarray:
        """
        在OpenCV图像上添加中文文字
        
        Args:
            img: OpenCV图像
            text: 要添加的文字
            position: 文字位置 (x, y)
            font_size: 字体大小
            color: 文字颜色 (B, G, R)
            thickness: 线条粗细
            
        Returns:
            添加文字后的图像
        """
        if not PIL_AVAILABLE:
            # 回退到OpenCV英文显示
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       font_size/32, color, thickness)
            return img
        
        try:
            # 转换OpenCV图像为PIL图像
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 获取中文字体
            font = self.get_chinese_font(font_size)
            if font is None:
                # 如果找不到中文字体，使用英文
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                           font_size/32, color, thickness)
                return img
            
            # 绘制中文文字 (PIL使用RGB，OpenCV使用BGR)
            draw.text(position, text, font=font, fill=color[::-1])
            
            # 转换回OpenCV格式
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
            
        except Exception as e:
            self.logger.warning(f"中文字体绘制失败，回退到英文: {e}")
            # 如果PIL绘制失败，回退到英文
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                       font_size/32, color, thickness)
            return img
    
    def visualize_frame(self, frame: np.ndarray, results: Dict[str, Any],
                       frame_count: int, total_frames: Optional[int] = None) -> np.ndarray:
        """
        统一的帧可视化接口
        
        Args:
            frame: 输入图像帧
            results: 检测结果字典
            frame_count: 当前帧数
            total_frames: 总帧数（可选）
            
        Returns:
            可视化后的图像帧
        """
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        try:
            # 绘制手部检测结果
            if 'hand_results' in results and self.config['show_bbox']:
                vis_frame = self._draw_hand_detections(vis_frame, results['hand_results'])
            
            # 绘制关键点
            if 'hand_results' in results and self.config['show_landmarks']:
                vis_frame = self._draw_landmarks(vis_frame, results['hand_results'])
            
            # 绘制轨迹
            if 'hand_results' in results and self.config['show_trajectory']:
                vis_frame = self._draw_trajectories(vis_frame, results['hand_results'])
            
            # 绘制行为识别结果
            if 'motion_results' in results and self.config['show_behavior_text']:
                vis_frame = self._draw_behavior_info(vis_frame, results['motion_results'])
            
            # 绘制统计信息
            if self.config['show_statistics']:
                vis_frame = self._draw_statistics(vis_frame, results, frame_count, total_frames)
            
            # 绘制帧信息
            vis_frame = self._draw_frame_info(vis_frame, frame_count, total_frames)
            
        except Exception as e:
            self.logger.error(f"可视化处理失败: {e}")
            # 在图像上显示错误信息
            error_text = f"可视化错误: {str(e)[:50]}"
            vis_frame = self.cv2_add_chinese_text(vis_frame, error_text, (20, height - 30),
                                                font_size=24, color=self.colors['warning'])
        
        return vis_frame
    
    def _draw_hand_detections(self, frame: np.ndarray, hand_results: List[Any]) -> np.ndarray:
        """
        绘制手部检测边界框
        
        Args:
            frame: 输入图像
            hand_results: 手部检测结果列表
            
        Returns:
            绘制后的图像
        """
        for i, hand in enumerate(hand_results):
            # 处理不同格式的手部检测结果
            bbox = None
            confidence = 0.0
            label = f'Hand_{i}'
            
            # 兼容不同的数据格式
            if hasattr(hand, 'bbox'):
                bbox = hand.bbox
                confidence = getattr(hand, 'confidence', 0.0)
                label = getattr(hand, 'hand_label', label)
            elif isinstance(hand, dict):
                bbox = hand.get('bbox')
                confidence = hand.get('confidence', 0.0)
                label = hand.get('label', label)
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['hand_bbox'], 
                            self.config['line_thickness'])
                
                # 绘制置信度标签
                text = f'{label}: {confidence:.2f}'
                frame = self.cv2_add_chinese_text(frame, text, (x1, y1-10),
                                                font_size=self.config['font_size']//2,
                                                color=self.colors['hand_bbox'])
        
        return frame
    
    def _draw_landmarks(self, frame: np.ndarray, hand_results: List[Any]) -> np.ndarray:
        """
        绘制手部关键点
        
        Args:
            frame: 输入图像
            hand_results: 手部检测结果列表
            
        Returns:
            绘制后的图像
        """
        height, width = frame.shape[:2]
        
        for hand in hand_results:
            landmarks = None
            
            # 兼容不同的数据格式
            if hasattr(hand, 'landmarks'):
                landmarks = hand.landmarks
            elif isinstance(hand, dict):
                landmarks = hand.get('landmarks')
            
            if landmarks:
                # 处理MediaPipe格式的关键点
                if hasattr(landmarks, 'landmark'):
                    for landmark in landmarks.landmark:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        cv2.circle(frame, (x, y), 3, self.colors['landmarks'], -1)
                # 处理列表格式的关键点
                elif isinstance(landmarks, list):
                    for point in landmarks:
                        if len(point) >= 2:
                            x, y = int(point[0] * width), int(point[1] * height)
                            cv2.circle(frame, (x, y), 3, self.colors['landmarks'], -1)
        
        return frame
    
    def _draw_trajectories(self, frame: np.ndarray, hand_results: List[Any]) -> np.ndarray:
        """
        绘制手部运动轨迹
        
        Args:
            frame: 输入图像
            hand_results: 手部检测结果列表
            
        Returns:
            绘制后的图像
        """
        for i, hand in enumerate(hand_results):
            bbox = None
            if hasattr(hand, 'bbox'):
                bbox = hand.bbox
            elif isinstance(hand, dict):
                bbox = hand.get('bbox')
            
            if bbox:
                # 计算边界框中心点
                x1, y1, x2, y2 = map(int, bbox)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # 更新轨迹点
                if i not in self.trajectory_points:
                    self.trajectory_points[i] = []
                
                self.trajectory_points[i].append(center)
                
                # 限制轨迹长度
                if len(self.trajectory_points[i]) > self.config['trajectory_length']:
                    self.trajectory_points[i].pop(0)
                
                # 绘制轨迹
                points = self.trajectory_points[i]
                for j in range(1, len(points)):
                    cv2.line(frame, points[j-1], points[j], self.colors['trajectory'], 2)
        
        return frame
    
    def _draw_behavior_info(self, frame: np.ndarray, motion_results: Dict[str, Any]) -> np.ndarray:
        """
        绘制行为识别信息
        
        Args:
            frame: 输入图像
            motion_results: 运动分析结果
            
        Returns:
            绘制后的图像
        """
        height, width = frame.shape[:2]
        
        # 获取行为信息
        behavior_type = motion_results.get('behavior_type', 'unknown')
        confidence = motion_results.get('behavior_confidence', 0.0)
        
        if confidence > 0.3:  # 只显示置信度较高的结果
            # 中文行为标签映射
            behavior_labels = {
                'handwashing': '洗手',
                'handwash': '洗手',
                'sanitizing': '消毒',
                'sanitize': '消毒',
                'none': '无行为',
                'unknown': '未知'
            }
            
            chinese_label = behavior_labels.get(behavior_type, behavior_type)
            behavior_text = f'行为: {chinese_label} ({confidence:.2f})'
            
            # 选择颜色
            color = self.colors.get(behavior_type, self.colors['text'])
            
            # 绘制行为信息
            frame = self.cv2_add_chinese_text(frame, behavior_text, (20, 50),
                                            font_size=self.config['font_size'],
                                            color=color)
            
            # 绘制详细置信度信息
            if 'handwash_confidence' in motion_results:
                handwash_conf = motion_results['handwash_confidence']
                sanitize_conf = motion_results.get('sanitize_confidence', 0.0)
                
                detail_text = f'洗手: {handwash_conf:.2f} | 消毒: {sanitize_conf:.2f}'
                frame = self.cv2_add_chinese_text(frame, detail_text, (20, 90),
                                                font_size=self.config['font_size']//2,
                                                color=self.colors['info'])
        
        return frame
    
    def _draw_statistics(self, frame: np.ndarray, results: Dict[str, Any],
                        frame_count: int, total_frames: Optional[int] = None) -> np.ndarray:
        """
        绘制统计信息
        
        Args:
            frame: 输入图像
            results: 检测结果
            frame_count: 当前帧数
            total_frames: 总帧数
            
        Returns:
            绘制后的图像
        """
        height, width = frame.shape[:2]
        
        # 统计信息
        stats_lines = []
        
        # 手部检测统计
        if 'hand_results' in results:
            hand_count = len(results['hand_results'])
            stats_lines.append(f'检测到手部: {hand_count}')
        
        # 处理时间统计
        if 'processing_time' in results:
            processing_time = results['processing_time']
            fps = 1.0 / processing_time if processing_time > 0 else 0
            stats_lines.append(f'处理时间: {processing_time*1000:.1f}ms')
            stats_lines.append(f'FPS: {fps:.1f}')
        
        # 绘制统计信息
        y_offset = height - 120
        for i, line in enumerate(stats_lines):
            frame = self.cv2_add_chinese_text(frame, line, (20, y_offset + i * 30),
                                            font_size=self.config['font_size']//2,
                                            color=self.colors['text'])
        
        return frame
    
    def _draw_frame_info(self, frame: np.ndarray, frame_count: int,
                        total_frames: Optional[int] = None) -> np.ndarray:
        """
        绘制帧信息
        
        Args:
            frame: 输入图像
            frame_count: 当前帧数
            total_frames: 总帧数
            
        Returns:
            绘制后的图像
        """
        height, width = frame.shape[:2]
        
        # 帧数信息
        if total_frames:
            frame_text = f'帧数: {frame_count}/{total_frames}'
            progress = frame_count / total_frames
            progress_text = f'进度: {progress*100:.1f}%'
        else:
            frame_text = f'帧数: {frame_count}'
            progress_text = ''
        
        # 绘制帧数信息
        frame = self.cv2_add_chinese_text(frame, frame_text, (width - 200, 30),
                                        font_size=self.config['font_size']//2,
                                        color=self.colors['text'])
        
        if progress_text:
            frame = self.cv2_add_chinese_text(frame, progress_text, (width - 200, 60),
                                            font_size=self.config['font_size']//2,
                                            color=self.colors['info'])
        
        return frame
    
    def draw_control_hints(self, frame: np.ndarray) -> np.ndarray:
        """
        绘制控制提示信息
        
        Args:
            frame: 输入图像
            
        Returns:
            绘制后的图像
        """
        height, width = frame.shape[:2]
        
        hints = [
            '控制键:',
            '空格键: 暂停/继续',
            'ESC键: 退出',
            'S键: 保存帧',
            'R键: 重置统计'
        ]
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 250, height - 150), (width - 10, height - 10),
                     self.colors['background'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 绘制提示文字
        for i, hint in enumerate(hints):
            frame = self.cv2_add_chinese_text(frame, hint, (width - 240, height - 140 + i * 25),
                                            font_size=self.config['font_size']//3,
                                            color=self.colors['text'])
        
        return frame
    
    def draw_error_message(self, frame: np.ndarray, error_msg: str) -> np.ndarray:
        """
        绘制错误信息
        
        Args:
            frame: 输入图像
            error_msg: 错误消息
            
        Returns:
            绘制后的图像
        """
        height, width = frame.shape[:2]
        
        error_text = f'错误: {error_msg}'
        frame = self.cv2_add_chinese_text(frame, error_text, (20, height - 30),
                                        font_size=self.config['font_size'],
                                        color=self.colors['warning'])
        
        return frame
    
    def reset_trajectories(self):
        """
        重置所有轨迹点
        """
        self.trajectory_points.clear()
        self.logger.info("轨迹点已重置")
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        更新可视化配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        self.logger.info(f"可视化配置已更新: {new_config}")