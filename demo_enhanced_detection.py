#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强手部行为检测演示脚本

这个脚本演示了所有改进功能的集成使用：
1. 增强手部检测（质量评估、多模型融合）
2. 高级运动分析（特征提取、自适应阈值）
3. 深度学习行为识别（Transformer模型）
4. 个性化适配（用户画像、自适应学习）
5. 性能优化（多线程、模型量化）

Author: Trae AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# 导入我们的增强模块
try:
    from src.core.enhanced_hand_detector import EnhancedHandDetector
    from src.core.enhanced_motion_analyzer import EnhancedMotionAnalyzer
    from src.core.deep_behavior_recognizer import DeepBehaviorRecognizer
    from src.core.personalization_engine import PersonalizationEngine
    from src.core.performance_optimizer import PerformanceOptimizer
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖模块都已正确安装")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/demo_enhanced_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedBehaviorDetectionDemo:
    """增强行为检测演示类"""
    
    def __init__(self, 
                 user_id: str = "demo_user",
                 enable_optimization: bool = True,
                 enable_deep_learning: bool = True,
                 enable_personalization: bool = True):
        """
        初始化演示系统
        
        Args:
            user_id: 用户ID
            enable_optimization: 是否启用性能优化
            enable_deep_learning: 是否启用深度学习
            enable_personalization: 是否启用个性化
        """
        self.user_id = user_id
        self.enable_optimization = enable_optimization
        self.enable_deep_learning = enable_deep_learning
        self.enable_personalization = enable_personalization
        
        # 初始化组件
        self._initialize_components()
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'detection_count': 0,
            'handwash_detections': 0,
            'sanitize_detections': 0,
            'avg_processing_time': 0.0,
            'quality_scores': [],
            'confidence_scores': []
        }
        
        logger.info(f"增强行为检测演示系统初始化完成 - 用户: {user_id}")
    
    def _initialize_components(self) -> None:
        """初始化所有组件"""
        try:
            # 1. 增强手部检测器
            from src.core.enhanced_hand_detector import DetectionMode
            self.hand_detector = EnhancedHandDetector(
                detection_mode=DetectionMode.WITH_FALLBACK,
                max_num_hands=4,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                quality_threshold=0.3
            )
            logger.info("✓ 增强手部检测器初始化完成")
            
            # 2. 增强运动分析器
            self.motion_analyzer = EnhancedMotionAnalyzer()
            logger.info("✓ 增强运动分析器初始化完成")
            
            # 3. 深度学习行为识别器（可选）
            if self.enable_deep_learning:
                self.deep_recognizer = DeepBehaviorRecognizer(
                    device='cpu',  # 演示使用CPU
                    sequence_length=30,
                    feature_dim=50
                )
                logger.info("✓ 深度学习行为识别器初始化完成")
            else:
                self.deep_recognizer = None
                logger.info("○ 深度学习行为识别器已禁用")
            
            # 4. 个性化引擎（可选）
            if self.enable_personalization:
                self.personalization = PersonalizationEngine(
                    storage_path="user_profiles"
                )
                self.personalization.set_current_user(self.user_id)
                logger.info("✓ 个性化引擎初始化完成")
            else:
                self.personalization = None
                logger.info("○ 个性化引擎已禁用")
            
            # 5. 性能优化器（可选）
            if self.enable_optimization:
                self.optimizer = PerformanceOptimizer(
                    enable_multithreading=True,
                    enable_quantization=False,  # 演示时禁用量化
                    enable_frame_buffering=True,
                    max_buffer_size=30
                )
                logger.info("✓ 性能优化器初始化完成")
            else:
                self.optimizer = None
                logger.info("○ 性能优化器已禁用")
                
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Dict: 处理结果
        """
        start_time = time.time()
        
        try:
            # 1. 手部检测
            detection_results = self.hand_detector.detect_hands_robust(frame)
            
            if not detection_results:
                return self._create_empty_result()
            
            # 转换为兼容格式
            detection_result = {
                'hands': [{
                    'landmarks': result.landmarks,
                    'bbox': result.bbox,
                    'confidence': result.confidence,
                    'label': result.hand_label
                } for result in detection_results]
            }
            
            # 2. 运动分析
            track_id = 1  # 演示使用固定track_id
            hands_data = [{
                'label': hand['label'],
                'landmarks': hand['landmarks'],
                'bbox': hand['bbox'],
                'confidence': hand['confidence']
            } for hand in detection_result['hands']]
            
            self.motion_analyzer.update_hand_motion(track_id, hands_data)
            
            # 3. 传统行为分析
            handwash_confidence = self.motion_analyzer.analyze_handwashing_enhanced(track_id)
            sanitize_confidence = self.motion_analyzer.analyze_sanitizing_enhanced(track_id)
            
            # 4. 深度学习行为识别（可选）
            deep_predictions = {'handwash': 0.0, 'sanitize': 0.0, 'none': 1.0}
            if self.deep_recognizer:
                motion_summary = self.motion_analyzer.get_enhanced_motion_summary(track_id)
                if motion_summary:
                    self.deep_recognizer.update_features(motion_summary)
                    deep_predictions = self.deep_recognizer.predict_behavior()
            
            # 5. 个性化处理（可选）
            personalized_thresholds = {}
            if self.personalization:
                personalized_thresholds = self.personalization.get_personalized_thresholds()
                
                # 更新用户行为数据
                if handwash_confidence > 0.6:
                    motion_summary = self.motion_analyzer.get_enhanced_motion_summary(track_id)
                    self.personalization.update_user_behavior(
                        'handwash', 15.0, handwash_confidence, motion_summary
                    )
                elif sanitize_confidence > 0.6:
                    motion_summary = self.motion_analyzer.get_enhanced_motion_summary(track_id)
                    self.personalization.update_user_behavior(
                        'sanitize', 5.0, sanitize_confidence, motion_summary
                    )
            
            # 6. 结果整合
            processing_time = time.time() - start_time
            
            result = {
                'timestamp': time.time(),
                'processing_time': processing_time,
                'detection_result': detection_result,
                'behavior_analysis': {
                    'handwash_confidence': handwash_confidence,
                    'sanitize_confidence': sanitize_confidence,
                    'deep_predictions': deep_predictions,
                    'personalized_thresholds': personalized_thresholds
                },
                'quality_metrics': detection_result.get('quality_metrics', {}),
                'hand_count': len(detection_result.get('hands', [])),
                'detection_source': detection_result.get('detection_source', 'unknown')
            }
            
            # 更新统计信息
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"帧处理失败: {e}")
            return self._create_empty_result()
    
    def _convert_detection_to_motion_data(self, detection_result: Dict) -> Dict:
        """将检测结果转换为运动数据格式"""
        hand_data = {}
        
        for hand in detection_result.get('hands', []):
            label = hand.get('label', 'unknown').lower()
            hand_key = f"{label}_hand"
            
            hand_data[hand_key] = {
                'landmarks': np.array(hand.get('landmarks', [])),
                'bbox': hand.get('bbox', [0, 0, 100, 100]),
                'confidence': hand.get('confidence', 0.5)
            }
        
        return hand_data
    
    def _create_empty_result(self) -> Dict:
        """创建空结果"""
        return {
            'timestamp': time.time(),
            'processing_time': 0.0,
            'detection_result': None,
            'behavior_analysis': {
                'handwash_confidence': 0.0,
                'sanitize_confidence': 0.0,
                'deep_predictions': {'handwash': 0.0, 'sanitize': 0.0, 'none': 1.0},
                'personalized_thresholds': {}
            },
            'quality_metrics': {},
            'hand_count': 0,
            'detection_source': 'none'
        }
    
    def _update_stats(self, result: Dict) -> None:
        """更新统计信息"""
        self.stats['total_frames'] += 1
        
        if result['hand_count'] > 0:
            self.stats['detection_count'] += 1
        
        behavior = result['behavior_analysis']
        if behavior['handwash_confidence'] > 0.5:
            self.stats['handwash_detections'] += 1
        if behavior['sanitize_confidence'] > 0.5:
            self.stats['sanitize_detections'] += 1
        
        # 更新平均处理时间
        current_avg = self.stats['avg_processing_time']
        new_time = result['processing_time']
        total_frames = self.stats['total_frames']
        self.stats['avg_processing_time'] = (current_avg * (total_frames - 1) + new_time) / total_frames
        
        # 记录质量分数
        quality_score = result['quality_metrics'].get('overall_score', 0.0)
        self.stats['quality_scores'].append(quality_score)
        
        # 记录置信度分数
        max_confidence = max(behavior['handwash_confidence'], behavior['sanitize_confidence'])
        self.stats['confidence_scores'].append(max_confidence)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            
        Returns:
            Dict: 处理统计结果
        """
        logger.info(f"开始处理视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 初始化输出视频（如果需要）
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                result = self.process_frame(frame)
                results.append(result)
                
                # 绘制检测结果（如果需要输出视频）
                if out:
                    annotated_frame = self._annotate_frame(frame, result)
                    out.write(annotated_frame)
                
                frame_count += 1
                
                # 进度显示
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if out:
                out.release()
        
        logger.info(f"视频处理完成: {frame_count}帧")
        
        return {
            'total_frames_processed': frame_count,
            'results': results,
            'statistics': self.get_statistics()
        }
    
    def _annotate_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """在帧上绘制检测结果"""
        annotated = frame.copy()
        
        # 绘制手部检测结果
        detection_result = result.get('detection_result')
        if detection_result and 'hands' in detection_result:
            for hand in detection_result['hands']:
                bbox = hand.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 显示标签和置信度
                    label = hand.get('label', 'Hand')
                    confidence = hand.get('confidence', 0.0)
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(annotated, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 显示行为分析结果
        behavior = result['behavior_analysis']
        y_offset = 30
        
        # 洗手置信度
        handwash_text = f"Handwash: {behavior['handwash_confidence']:.2f}"
        cv2.putText(annotated, handwash_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += 25
        
        # 消毒置信度
        sanitize_text = f"Sanitize: {behavior['sanitize_confidence']:.2f}"
        cv2.putText(annotated, sanitize_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        
        # 深度学习预测（如果可用）
        if self.deep_recognizer:
            deep_pred = behavior['deep_predictions']
            deep_text = f"Deep: H:{deep_pred['handwash']:.2f} S:{deep_pred['sanitize']:.2f}"
            cv2.putText(annotated, deep_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
        
        # 质量分数
        quality_score = result['quality_metrics'].get('overall_score', 0.0)
        quality_text = f"Quality: {quality_score:.2f}"
        cv2.putText(annotated, quality_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 检测源
        source_text = f"Source: {result['detection_source']}"
        cv2.putText(annotated, source_text, (10, annotated.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return annotated
    
    def process_webcam(self, camera_id: int = 0, duration: Optional[float] = None) -> None:
        """
        处理摄像头实时视频
        
        Args:
            camera_id: 摄像头ID
            duration: 处理时长（秒），None表示无限制
        """
        logger.info(f"开始摄像头实时处理: 摄像头{camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("无法读取摄像头帧")
                    break
                
                # 处理帧
                result = self.process_frame(frame)
                
                # 绘制结果
                annotated_frame = self._annotate_frame(frame, result)
                
                # 显示
                cv2.imshow('Enhanced Behavior Detection Demo', annotated_frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("用户请求退出")
                    break
                
                if duration and (time.time() - start_time) > duration:
                    logger.info(f"达到指定时长: {duration}秒")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info("摄像头处理结束")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算额外统计信息
        if stats['quality_scores']:
            stats['avg_quality_score'] = np.mean(stats['quality_scores'])
            stats['min_quality_score'] = np.min(stats['quality_scores'])
            stats['max_quality_score'] = np.max(stats['quality_scores'])
        
        if stats['confidence_scores']:
            stats['avg_confidence_score'] = np.mean(stats['confidence_scores'])
            stats['min_confidence_score'] = np.min(stats['confidence_scores'])
            stats['max_confidence_score'] = np.max(stats['confidence_scores'])
        
        # 检测率
        if stats['total_frames'] > 0:
            stats['detection_rate'] = stats['detection_count'] / stats['total_frames']
            stats['handwash_rate'] = stats['handwash_detections'] / stats['total_frames']
            stats['sanitize_rate'] = stats['sanitize_detections'] / stats['total_frames']
        
        return stats
    
    def get_user_profile_summary(self) -> Optional[Dict]:
        """获取用户画像摘要"""
        if not self.personalization:
            return None
        
        profile = self.personalization.get_current_user_profile()
        if not profile:
            return None
        
        return profile.get_profile_summary()
    
    def get_recommendations(self) -> Optional[Dict]:
        """获取用户推荐"""
        if not self.personalization:
            return None
        
        return self.personalization.get_user_recommendations()
    
    def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理系统资源...")
        
        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.cleanup()
        
        if hasattr(self, 'motion_analyzer') and self.motion_analyzer:
            self.motion_analyzer.cleanup()
        
        logger.info("资源清理完成")


def check_test_video(video_path: str) -> bool:
    """
    检查测试视频是否存在
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        bool: 视频文件是否存在
    """
    if Path(video_path).exists():
        logger.info(f"使用现有测试视频: {video_path}")
        return True
    else:
        logger.warning(f"测试视频不存在: {video_path}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强手部行为检测演示')
    parser.add_argument('--mode', choices=['video', 'webcam', 'test'], default='test',
                       help='运行模式: video(视频文件), webcam(摄像头), test(创建测试视频)')
    parser.add_argument('--input', type=str, help='输入视频文件路径')
    parser.add_argument('--output', type=str, help='输出视频文件路径')
    parser.add_argument('--user-id', type=str, default='demo_user', help='用户ID')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    parser.add_argument('--duration', type=float, help='处理时长（秒）')
    parser.add_argument('--disable-optimization', action='store_true', help='禁用性能优化')
    parser.add_argument('--disable-deep-learning', action='store_true', help='禁用深度学习')
    parser.add_argument('--disable-personalization', action='store_true', help='禁用个性化')
    
    args = parser.parse_args()
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)
    
    try:
        if args.mode == 'test':
            # 使用现有测试视频
            test_video_path = 'tests/fixtures/videos/20250724072708.mp4'
            Path(test_video_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 检查测试视频是否存在
            if not check_test_video(test_video_path):
                logger.error("测试视频不存在，请确保视频文件已放置在正确位置")
                return
            
            # 处理测试视频
            demo = EnhancedBehaviorDetectionDemo(
                user_id=args.user_id,
                enable_optimization=not args.disable_optimization,
                enable_deep_learning=not args.disable_deep_learning,
                enable_personalization=not args.disable_personalization
            )
            
            output_path = args.output or 'tests/fixtures/videos/test_output.mp4'
            result = demo.process_video(test_video_path, output_path)
            
            # 显示统计信息
            stats = demo.get_statistics()
            print("\n=== 处理统计 ===")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
            # 显示用户画像（如果启用）
            if not args.disable_personalization:
                profile = demo.get_user_profile_summary()
                if profile:
                    print("\n=== 用户画像 ===")
                    print(json.dumps(profile, indent=2, ensure_ascii=False))
                
                recommendations = demo.get_recommendations()
                if recommendations:
                    print("\n=== 用户推荐 ===")
                    print(json.dumps(recommendations, indent=2, ensure_ascii=False))
            
            demo.cleanup()
            
        elif args.mode == 'video':
            if not args.input:
                raise ValueError("视频模式需要指定输入文件 (--input)")
            
            demo = EnhancedBehaviorDetectionDemo(
                user_id=args.user_id,
                enable_optimization=not args.disable_optimization,
                enable_deep_learning=not args.disable_deep_learning,
                enable_personalization=not args.disable_personalization
            )
            
            result = demo.process_video(args.input, args.output)
            
            # 显示结果
            stats = demo.get_statistics()
            print("\n=== 处理统计 ===")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
            demo.cleanup()
            
        elif args.mode == 'webcam':
            demo = EnhancedBehaviorDetectionDemo(
                user_id=args.user_id,
                enable_optimization=not args.disable_optimization,
                enable_deep_learning=not args.disable_deep_learning,
                enable_personalization=not args.disable_personalization
            )
            
            print("开始摄像头实时检测，按 'q' 键退出...")
            demo.process_webcam(args.camera_id, args.duration)
            
            # 显示统计信息
            stats = demo.get_statistics()
            print("\n=== 处理统计 ===")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
            demo.cleanup()
    
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        raise


if __name__ == '__main__':
    main()