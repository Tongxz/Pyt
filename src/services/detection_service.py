import base64
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import Depends, Request

from core.optimized_detection_pipeline import (
    DetectionResult,
    OptimizedDetectionPipeline,
)
from core.yolo_hairnet_detector import YOLOHairnetDetector

logger = logging.getLogger(__name__)

# 在这里我们将实例化我们的服务，或者在app启动时进行
# 为了简单起见，我们先在这里设置为None
optimized_pipeline: Optional[OptimizedDetectionPipeline] = None
hairnet_pipeline: Optional[YOLOHairnetDetector] = None


def get_optimized_pipeline(request: Request) -> Optional[OptimizedDetectionPipeline]:
    return getattr(request.app.state, "optimized_pipeline", None)


def get_hairnet_pipeline(request: Request) -> Optional[YOLOHairnetDetector]:
    return getattr(request.app.state, "hairnet_pipeline", None)


def comprehensive_detection_logic(
    contents: bytes,
    filename: str,
    optimized_pipeline: Optional[OptimizedDetectionPipeline],
    hairnet_pipeline: Optional[YOLOHairnetDetector],
) -> dict:
    """
    执行综合检测并返回统一格式的结果。
    这个函数会被 comprehensive.py 中的API端点调用。
    支持图像和视频文件。
    """
    import cv2
    import numpy as np
    import tempfile
    import os
    from pathlib import Path

    # 检查文件类型
    file_ext = Path(filename).suffix.lower()
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    image = None
    
    if file_ext in video_extensions:
        # 处理视频文件
        logger.info(f"检测到视频文件: {filename}")
        
        # 将视频内容写入临时文件
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_file.write(contents)
            temp_video_path = temp_file.name
        
        try:
            # 打开视频并提取第一帧
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError("无法从视频中读取帧")
            
            image = frame
            cap.release()
            
            logger.info(f"成功从视频中提取第一帧，尺寸: {image.shape}")
            
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_video_path)
            except Exception as e:
                logger.warning(f"清理临时视频文件失败: {e}")
                
    elif file_ext in image_extensions:
        # 处理图像文件
        logger.info(f"检测到图像文件: {filename}")
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像")
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}。支持的格式: {image_extensions | video_extensions}")
    
    if image is None:
        raise ValueError("无法获取有效的图像数据")

    if optimized_pipeline:
        logger.info("使用优化检测管道进行综合检测")
        result = optimized_pipeline.detect_comprehensive(image)
        
        # 转换为前端期望的格式
        total_persons = len(result.person_detections)
        persons_with_hairnet = len([h for h in result.hairnet_results if h.get('has_hairnet', False)])
        persons_handwashing = len([h for h in result.handwash_results if h.get('is_handwashing', False)])
        persons_sanitizing = len([s for s in result.sanitize_results if s.get('is_sanitizing', False)])
        
        # 构建统计信息
        statistics = {
            "persons_with_hairnet": persons_with_hairnet,
            "persons_handwashing": persons_handwashing,
            "persons_sanitizing": persons_sanitizing
        }
        
        # 构建检测详情
        detections = []
        for detection in result.person_detections:
            detections.append({
                "class": "person",
                "confidence": detection.get("confidence", 0.0),
                "bbox": detection.get("bbox", [0, 0, 0, 0])
            })
        
        for detection in result.hairnet_results:
            if detection.get('has_hairnet', False):
                detections.append({
                    "class": "hairnet",
                    "confidence": detection.get("hairnet_confidence", 0.0),
                    "bbox": detection.get("hairnet_bbox", [0, 0, 0, 0])
                })
        
        # 处理标注图像
        annotated_image_b64 = None
        if result.annotated_image is not None:
            _, buffer = cv2.imencode(".jpg", result.annotated_image)
            annotated_image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        
        return {
            "total_persons": total_persons,
            "statistics": statistics,
            "detections": detections,
            "annotated_image": annotated_image_b64,
            "processing_times": result.processing_times
        }
        
    elif hairnet_pipeline:
        logger.warning("优化检测管道不可用，回退到原有发网检测逻辑")
        result = hairnet_pipeline.detect_hairnet_compliance(image)
        
        # 转换为前端期望的格式
        total_persons = result.get("total_persons", 0)
        persons_with_hairnet = result.get("persons_with_hairnet", 0)
        
        # 构建统计信息（回退模式只有发网检测）
        statistics = {
            "persons_with_hairnet": persons_with_hairnet,
            "persons_handwashing": 0,  # 回退模式不支持
            "persons_sanitizing": 0   # 回退模式不支持
        }
        
        # 构建检测详情
        detections = []
        for detection in result.get("detections", []):
            detections.append({
                "class": "hairnet" if detection.get("has_hairnet", False) else "person",
                "confidence": detection.get("confidence", 0.0),
                "bbox": detection.get("bbox", [0, 0, 0, 0])
            })
        
        # 处理可视化图像
        annotated_image_b64 = None
        if "visualization" in result and isinstance(result.get("visualization"), np.ndarray):
            img = result["visualization"]
            _, buffer = cv2.imencode(".jpg", img)
            annotated_image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        elif "visualization" in result and isinstance(result.get("visualization"), str):
            # 如果已经是base64字符串
            annotated_image_b64 = result["visualization"]
        
        return {
            "total_persons": total_persons,
            "statistics": statistics,
            "detections": detections,
            "annotated_image": annotated_image_b64,
            "compliance_rate": result.get("compliance_rate", 0.0),
            "average_confidence": result.get("average_confidence", 0.0)
        }
    else:
        logger.error("所有检测管道都不可用")
        raise RuntimeError("检测服务未初始化")


def initialize_detection_services():
    """初始化所有检测服务和模型."""
    global optimized_pipeline, hairnet_pipeline
    logger.info("正在初始化检测服务...")
    try:
        # 这里的初始化逻辑需要从 app.py 的 startup 事件中迁移过来
        # 为了演示，我们先使用None
        from src.core.behavior import BehaviorRecognizer
        from src.core.data_manager import DetectionDataManager
        from src.core.detector import HumanDetector
        from src.core.region import RegionManager
        from src.core.rule_engine import RuleEngine

        detector = HumanDetector()
        behavior_recognizer = BehaviorRecognizer()
        data_manager = DetectionDataManager()
        region_manager = RegionManager()
        rule_engine = RuleEngine()

        optimized_pipeline = OptimizedDetectionPipeline(
            human_detector=detector,
            hairnet_detector=YOLOHairnetDetector(),
            behavior_recognizer=behavior_recognizer,
        )
        hairnet_pipeline = YOLOHairnetDetector()
        logger.info("检测服务初始化完成。")
    except Exception as e:
        logger.exception(f"初始化检测服务失败: {e}")