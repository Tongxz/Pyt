import logging
import base64
from typing import Optional, Any, Dict

from fastapi import Depends, Request

from dataclasses import asdict

from core.optimized_detection_pipeline import OptimizedDetectionPipeline, DetectionResult
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
    hairnet_pipeline: Optional[YOLOHairnetDetector]
) -> dict:
    """
    这里应该是包含所有检测逻辑的地方。
    这个函数会被 comprehensive.py 中的API端点调用。
    """
    # 这个实现只是一个例子，你需要将 app.py 中的逻辑移到这里
    if optimized_pipeline:
        logger.info("使用优化检测管道进行综合检测")
        import cv2
        import numpy as np
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像")
        result = optimized_pipeline.detect_comprehensive(image)
        result_dict = asdict(result)
        if 'annotated_image' in result_dict and result_dict['annotated_image'] is not None:
            img = result_dict['annotated_image']
            _, buffer = cv2.imencode('.jpg', img)
            result_dict['annotated_image'] = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return {"source": "optimized_pipeline", "result": result_dict}
    elif hairnet_pipeline:
        logger.warning("优化检测管道不可用，回退到原有发网检测逻辑")
        # YOLOHairnetDetector 需要一个图像帧, 而不是字节
        # 我们需要先解码
        import cv2
        import numpy as np
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码图像")
        result = hairnet_pipeline.detect_hairnet_compliance(image)
        if 'visualization' in result and isinstance(result.get('visualization'), np.ndarray):
             img = result['visualization']
             _, buffer = cv2.imencode('.jpg', img)
             result['visualization'] = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return {"source": "fallback_hairnet_pipeline", "result": result}
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
        from src.core.detector import HumanDetector
        from src.core.behavior import BehaviorRecognizer
        from src.core.data_manager import DetectionDataManager
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
            behavior_recognizer=behavior_recognizer
        )
        hairnet_pipeline = YOLOHairnetDetector()
        logger.info("检测服务初始化完成。")
    except Exception as e:
        logger.exception(f"初始化检测服务失败: {e}")