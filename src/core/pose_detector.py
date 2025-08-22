"""姿态检测模块.

提供基于MediaPipe的人体姿态和手部关键点检测功能，
集成增强的手部检测器和质量评估功能。
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# 智能GPU检测和配置
def _configure_mediapipe_gpu():
    """智能配置MediaPipe GPU使用策略"""
    try:
        import torch
        
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # 检查GPU显存是否足够（至少需要2GB可用显存）
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
            available_memory_gb = gpu_memory_gb - allocated_memory_gb
            
            # 检查是否为支持的GPU架构（计算能力>=6.0）
            compute_capability = torch.cuda.get_device_capability(0)
            compute_version = compute_capability[0] + compute_capability[1] * 0.1
            
            # GPU使用条件：
            # 1. CUDA可用
            # 2. 可用显存>=2GB
            # 3. 计算能力>=6.0
            # 4. 没有手动禁用GPU的环境变量
            manual_disable = os.environ.get('MEDIAPIPE_DISABLE_GPU', '').lower() in ['1', 'true', 'yes']
            
            if not manual_disable and available_memory_gb >= 2.0 and compute_version >= 6.0:
                # 启用GPU加速
                if 'MEDIAPIPE_DISABLE_GPU' in os.environ:
                    del os.environ['MEDIAPIPE_DISABLE_GPU']
                logger = logging.getLogger(__name__)
                logger.info(f"MediaPipe GPU加速已启用 - GPU: {torch.cuda.get_device_name(0)}, "
                           f"可用显存: {available_memory_gb:.1f}GB, 计算能力: {compute_version:.1f}")
                return True
            else:
                # 禁用GPU，使用CPU模式
                os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
                logger = logging.getLogger(__name__)
                reasons = []
                if manual_disable:
                    reasons.append("手动禁用")
                if available_memory_gb < 2.0:
                    reasons.append(f"显存不足({available_memory_gb:.1f}GB<2.0GB)")
                if compute_version < 6.0:
                    reasons.append(f"计算能力不足({compute_version:.1f}<6.0)")
                logger.info(f"MediaPipe使用CPU模式 - 原因: {', '.join(reasons)}")
                return False
        else:
            # CUDA不可用，使用CPU模式
            os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
            logger = logging.getLogger(__name__)
            logger.info("MediaPipe使用CPU模式 - CUDA不可用")
            return False
            
    except ImportError:
        # PyTorch不可用，默认使用CPU模式
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        logger = logging.getLogger(__name__)
        logger.info("MediaPipe使用CPU模式 - PyTorch不可用")
        return False
    except Exception as e:
        # 其他异常，默认使用CPU模式
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        logger = logging.getLogger(__name__)
        logger.warning(f"MediaPipe GPU检测失败，使用CPU模式: {e}")
        return False

# 配置MediaPipe GPU使用策略
_gpu_enabled = _configure_mediapipe_gpu()

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

from .enhanced_hand_detector import EnhancedHandDetector, DetectionMode, HandDetectionResult
from ..utils.logger import get_logger
from .detector import BaseDetector
from ultralytics import YOLO
from ..config.unified_params import get_unified_params


logger = get_logger(__name__)


class PoseDetectorFactory:
    """姿态检测器工厂

    根据配置动态创建和管理人体姿态和手部关键点检测器。
    """

    @staticmethod
    def create(backend: str = "yolov8", **kwargs) -> BaseDetector:
        """
        创建姿态检测器实例

        Args:
            backend: 要使用的后端 ('yolov8' or 'mediapipe')
            **kwargs: 传递给检测器构造函数的其他参数

        Returns:
            一个实现了 BaseDetector 接口的检测器实例

        Raises:
            ValueError: 如果选择了无效的后端
        """
        if backend == "yolov8":
            logger.info("创建 YOLOv8PoseDetector 实例")
            return YOLOv8PoseDetector(**kwargs)
        elif backend == "mediapipe":
            logger.info("创建 MediaPipePoseDetector 实例")
            return MediaPipePoseDetector(**kwargs)
        else:
            raise ValueError(f"无效的姿态检测后端: {backend}")


class YOLOv8PoseDetector(BaseDetector):
    """YOLOv8姿态检测器

    基于YOLOv8-pose模型进行人体姿态关键点检测。
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        初始化YOLOv8姿态检测器

        Args:
            model_path: YOLO模型路径，如果为None则使用统一配置
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.params = get_unified_params().pose_detection
        model_path = model_path if model_path is not None else self.params.model_path
        device = device if device != "auto" else self.params.device

        super().__init__(model_path, device)

        self.confidence_threshold = self.params.confidence_threshold
        self.iou_threshold = self.params.iou_threshold

        logger.info(
            f"YOLOv8PoseDetector initialized on {self.device} with params: "
            f"conf={self.confidence_threshold}, iou={self.iou_threshold}"
        )

    def _load_model(self, model_path: str) -> YOLO:
        """加载YOLO模型"""
        try:
            model = YOLO(model_path)
            if hasattr(model, "to"):
                model.to(self.device)
            logger.info(f"成功加载YOLOv8姿态模型: {model_path} 到设备: {self.device}")
            return model
        except Exception as e:
            logger.error(f"YOLOv8姿态模型加载失败: {e}")
            raise RuntimeError(f"无法加载YOLOv8姿态模型: {model_path}") from e

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测图像中的人体姿态关键点

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表，每个元素是一个字典，包含:
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float
            - 'keypoints': 包含所有关键点信息的字典
                - 'xy': (N, 2) 的numpy数组，包含x,y坐标
                - 'conf': (N,) 的numpy数组，包含每个关键点的置信度
        """
        if self.model is None:
            raise RuntimeError("YOLOv8姿态模型未正确加载，无法进行检测")

        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            detections = []
            for result in results:
                if result.boxes is None or result.keypoints is None:
                    continue

                for box, keypoints in zip(result.boxes, result.keypoints):
                    # 使用 .item() 可以安全地从0维张量中提取数值，避免DeprecationWarning
                    if box.cls[0].item() != 0:  # 只处理 'person' 类别
                        continue
                    
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].item()
                    
                    kpts_xy = keypoints.xy[0].cpu().numpy()
                    kpts_conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(len(kpts_xy))

                    detection = {
                        "bbox": bbox.tolist(),
                        "confidence": conf,
                        "keypoints": {
                            "xy": kpts_xy,
                            "conf": kpts_conf,
                        },
                        "class_id": 0,
                        "class_name": "person",
                    }
                    detections.append(detection)
            
            return detections

        except Exception as e:
            logger.error(f"YOLOv8姿态检测过程中发生错误: {e}", exc_info=True)
            raise RuntimeError("YOLOv8姿态检测失败") from e

    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上可视化姿态检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表

        Returns:
            带有检测框和关键点的图像
        """
        vis_image = image.copy()
        
        for detection in detections:
            # 绘制边界框
            bbox = detection.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                label = self._get_label(detection)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # 绘制关键点
            keypoints = detection.get("keypoints")
            if keypoints:
                kpts_xy = keypoints.get("xy")
                kpts_conf = keypoints.get("conf")
                
                # COCO keypoints connection pairs
                skeleton = [
                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                ]
                
                for i, (x, y) in enumerate(kpts_xy):
                    if kpts_conf[i] > 0.5: # 只绘制高置信度的点
                        cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)

                for pair in skeleton:
                    idx1, idx2 = pair[0] - 1, pair[1] - 1
                    if kpts_conf[idx1] > 0.5 and kpts_conf[idx2] > 0.5:
                        pt1 = (int(kpts_xy[idx1][0]), int(kpts_xy[idx1][1]))
                        pt2 = (int(kpts_xy[idx2][0]), int(kpts_xy[idx2][1]))
                        cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)

        return vis_image

    def cleanup(self):
        """清理资源."""
        # YOLOv8模型会自动管理资源，这里主要用于接口一致性
        logger.info("YOLOv8PoseDetector cleaned up")


class MediaPipePoseDetector(BaseDetector):
    """MediaPipe姿态检测器.

    使用MediaPipe检测人体关键点，特别是手部关键点。
    """

    def __init__(self, 
                 use_enhanced_hand_detection: bool = True,
                 detection_mode: DetectionMode = DetectionMode.WITH_FALLBACK,
                 **kwargs):
        """初始化MediaPipe姿态检测器.

        Args:
            use_enhanced_hand_detection: 是否使用增强的手部检测器
            detection_mode: 手部检测模式
        """
        # MediaPipe不需要模型路径，但为了兼容BaseDetector，传入一个虚拟路径
        super().__init__(model_path="mediapipe", device="auto")

        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not available. Please install it.")

        self.use_enhanced_hand_detection = use_enhanced_hand_detection
        
        # 初始化增强手部检测器
        if self.use_enhanced_hand_detection:
            try:
                self.enhanced_hand_detector = EnhancedHandDetector(
                    detection_mode=detection_mode,
                    max_num_hands=4,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    quality_threshold=0.7
                )
                logger.info(f"Enhanced hand detector initialized with mode: {detection_mode.value}")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced hand detector: {e}")
                self.use_enhanced_hand_detection = False
                self.enhanced_hand_detector = None
        else:
            self.enhanced_hand_detector = None

        try:
            mp_pose = mp.solutions.pose # type: ignore
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            device_mode = "GPU模式" if _gpu_enabled else "CPU模式"
            logger.info(f"MediaPipePoseDetector initialized ({device_mode})")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise RuntimeError("Could not initialize MediaPipe Pose.") from e

    def _load_model(self, model_path: str):
        """MediaPipe不需要加载外部模型文件，此方法为空."""
        pass

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测人体姿态和手部.

        为了与YOLOv8的输出格式保持一致，这里将MediaPipe的结果进行转换。
        """
        pose_results = self._detect_pose(image)
        hand_results = self._detect_hands(image)

        # 在此简化实现中，我们将姿态和手部检测分开处理
        # 一个更完整的实现可能会将手部关键点附加到对应的人体上
        
        detections = []
        if pose_results and pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            h, w, _ = image.shape

            # 提取关键点坐标和置信度
            kpts_xy = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
            kpts_conf = np.array([lm.visibility for lm in landmarks])

            # 计算边界框
            x_coords = kpts_xy[:, 0]
            y_coords = kpts_xy[:, 1]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            detection = {
                "bbox": [int(c) for c in bbox],
                "confidence": np.mean(kpts_conf[kpts_conf > 0]), # 平均可见度作为置信度
                "keypoints": {
                    "xy": kpts_xy,
                    "conf": kpts_conf,
                },
                "class_id": 0,
                "class_name": "person",
            }
            detections.append(detection)

        # TODO: 添加手部检测结果的转换

        return detections

    def _detect_pose(self, image: np.ndarray):
        """使用MediaPipe检测姿态."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.pose.process(rgb_image)
        except Exception as e:
            logger.error(f"MediaPipe pose detection failed: {e}")
            return None

    def _detect_hands(self, image: np.ndarray) -> List[Dict]:
        """检测手部关键点."""
        hand_detections = []
        
        if self.use_enhanced_hand_detection and self.enhanced_hand_detector:
            try:
                enhanced_results = self.enhanced_hand_detector.detect_hands_robust(image)
                
                # 转换增强手部检测结果为统一格式
                for result in enhanced_results:
                    if hasattr(result, 'bbox') and hasattr(result, 'confidence'):
                        hand_detection = {
                            "bbox": result.bbox,
                            "confidence": result.confidence,
                            "class_id": 1,  # 手部类别ID
                            "class_name": "hand",
                            "keypoints": getattr(result, 'keypoints', None)
                        }
                        hand_detections.append(hand_detection)
                        
            except Exception as e:
                logger.error(f"Enhanced hand detection failed: {e}")
                
        return hand_detections

    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """可视化MediaPipe检测结果."""
        vis_image = image.copy()
        mp_drawing = mp.solutions.drawing_utils # type: ignore
        mp_pose = mp.solutions.pose # type: ignore

        for detection in detections:
            # 这是一个简化的可视化，实际应使用MediaPipe的landmark对象
            # 这里我们只绘制边界框和关键点
            bbox = detection.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            keypoints = detection.get("keypoints")
            if keypoints:
                kpts_xy = keypoints.get("xy")
                for (x, y) in kpts_xy:
                    cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)

        return vis_image

    def cleanup(self):
        """清理资源."""
        if hasattr(self, "pose") and self.pose:
            self.pose.close()
        logger.info("MediaPipePoseDetector cleaned up")


# 移除旧的 PoseDetector 类
