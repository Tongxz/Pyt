import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

from .motion_analyzer import MotionAnalyzer
from .pose_detector import PoseDetector

# 导入统一参数配置
try:
    from src.config.unified_params import get_unified_params
except ImportError:
    # 兼容性处理
    import os
    import sys

    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from src.config.unified_params import get_unified_params

logger = logging.getLogger(__name__)


class BehaviorState:
    """行为状态类"""

    def __init__(self, behavior_type: str, confidence: float = 0.0):
        self.behavior_type = behavior_type
        self.confidence = confidence
        self.start_time = time.time()
        self.duration = 0.0
        self.is_active = True

    def update(self, confidence: float):
        """更新行为状态"""
        self.confidence = confidence
        self.duration = time.time() - self.start_time

    def end(self):
        """结束行为"""
        self.is_active = False
        self.duration = time.time() - self.start_time


class BehaviorRecognizer:
    """行为识别器

    识别人体的特定行为，包括：
    - 发网佩戴检测
    - 洗手行为识别
    - 手部消毒识别
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        use_advanced_detection: Optional[bool] = None,
        use_mediapipe: Optional[bool] = None,
    ):
        """
        初始化行为识别器

        Args:
            confidence_threshold: 行为识别置信度阈值，如果为None则使用统一配置
            use_advanced_detection: 是否使用高级检测，如果为None则使用统一配置
            use_mediapipe: 是否使用 MediaPipe，如果为None则使用统一配置
        """
        # 获取统一参数配置
        self.params = get_unified_params().behavior_recognition

        # 使用统一配置或传入参数
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.params.confidence_threshold
        )
        self.use_advanced_detection = (
            use_advanced_detection
            if use_advanced_detection is not None
            else self.params.use_advanced_detection
        )
        self.use_mediapipe = (
            use_mediapipe if use_mediapipe is not None else self.params.use_mediapipe
        ) and MEDIAPIPE_AVAILABLE

        # 使用统一配置的历史记录长度
        self.behavior_history = defaultdict(
            lambda: deque(maxlen=self.params.history_maxlen)
        )
        self.active_behaviors = {}

        # 初始化 MediaPipe 手部检测器（使用统一参数）
        self.mp_hands = None
        self.hands_detector = None
        if self.use_mediapipe and mp is not None:
            try:
                self.mp_hands = mp.solutions.hands  # type: ignore
                self.hands_detector = self.mp_hands.Hands(  # type: ignore
                    static_image_mode=False,
                    max_num_hands=self.params.max_num_hands,
                    min_detection_confidence=self.params.min_detection_confidence,
                    min_tracking_confidence=self.params.min_tracking_confidence,
                )
                logger.info(
                    "MediaPipe hands detector initialized successfully with unified params"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe hands detector: {e}")
                self.use_mediapipe = False

        # 初始化高级检测模块
        if self.use_advanced_detection:
            try:
                self.pose_detector = PoseDetector()
                self.motion_analyzer = MotionAnalyzer()
                logger.info("Advanced detection modules initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced detection: {e}")
                self.use_advanced_detection = False
                self.pose_detector = None
                self.motion_analyzer = None
        else:
            self.pose_detector = None
            self.motion_analyzer = None

        # 行为识别规则配置（使用统一参数）
        self.behavior_rules = {
            "hairnet": {
                "min_duration": self.params.hairnet_min_duration,
                "stability_frames": self.params.hairnet_stability_frames,
            },
            "handwashing": {
                "min_duration": self.params.handwashing_min_duration,
                "max_duration": self.params.handwashing_max_duration,
                "stability_frames": self.params.handwashing_stability_frames,
            },
            "sanitizing": {
                "min_duration": self.params.sanitizing_min_duration,
                "max_duration": self.params.sanitizing_max_duration,
                "stability_frames": self.params.sanitizing_stability_frames,
            },
        }

        logger.info(
            f"BehaviorRecognizer initialized with unified params: "
            f"threshold={self.confidence_threshold}, "
            f"advanced_detection={self.use_advanced_detection}, "
            f"mediapipe={self.use_mediapipe}, "
            f"handwash_min_duration={self.params.handwashing_min_duration}s"
        )

    def detect_hairnet(
        self, person_bbox: List[int], head_region: Optional[Dict] = None
    ) -> float:
        """
        检测发网佩戴

        Args:
            person_bbox: 人体边界框
            head_region: 头部区域信息（可选）

        Returns:
            发网检测置信度
        """
        # 简化的发网检测逻辑
        # 实际实现中会使用专门的CNN模型

        if head_region is None:
            # 估算头部区域（人体上部1/4区域）
            x1, y1, x2, y2 = person_bbox
            head_height = (y2 - y1) // 4
            head_region = {
                "bbox": [x1, y1, x2, y1 + head_height],
                "features": {},  # 这里会包含头部特征
            }

        # 模拟发网检测逻辑
        # 实际会基于头部纹理、颜色、形状等特征
        confidence = 0.0

        # 检查头部区域特征
        head_bbox = head_region["bbox"]
        head_area = (head_bbox[2] - head_bbox[0]) * (head_bbox[3] - head_bbox[1])

        # 简单的启发式规则（实际会用深度学习模型）
        if head_area > 100:  # 头部区域足够大
            # 模拟特征检测
            confidence = 0.8  # 临时固定值，实际会动态计算

        return confidence

    def _basic_sanitizing_detection(
        self, person_bbox: List[int], hand_regions: List[Dict]
    ) -> float:
        """
        基础消毒检测方法

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表

        Returns:
            消毒行为置信度
        """
        confidence = 0.0

        # 检查双手是否靠近（消毒时通常双手互相摩擦）
        if len(hand_regions) >= 2:
            hand1_center = self._get_bbox_center(hand_regions[0]["bbox"])
            hand2_center = self._get_bbox_center(hand_regions[1]["bbox"])

            # 计算双手距离
            distance = (
                (hand1_center[0] - hand2_center[0]) ** 2
                + (hand1_center[1] - hand2_center[1]) ** 2
            ) ** 0.5

            # 如果双手距离较近，可能在进行消毒
            if distance < 100:  # 像素距离阈值
                # 基于距离计算置信度
                distance_score = max(0.0, (100 - distance) / 100) * 0.4
                confidence += distance_score

                # 检查手部位置是否在合理范围内
                person_height = person_bbox[3] - person_bbox[1]
                if person_height > 0:
                    # 消毒时手部通常在身体中上部
                    avg_hand_y = (hand1_center[1] + hand2_center[1]) / 2
                    relative_y = (avg_hand_y - person_bbox[1]) / person_height

                    if 0.2 <= relative_y <= 0.7:  # 合理的手部位置
                        confidence += 0.2
            else:
                confidence = 0.0
        else:
            # 只有一只手或没有手，不太可能在消毒
            confidence = 0.0

        return confidence

    def _analyze_sanitizing_pose(
        self, pose_data: Dict, hands_data: List[Dict], person_bbox: List[int]
    ) -> float:
        """
        基于姿态数据分析消毒行为

        Args:
            pose_data: 姿态检测数据
            hands_data: 手部检测数据
            person_bbox: 人体边界框

        Returns:
            姿态分析置信度
        """
        if not pose_data or not hands_data:
            return 0.0

        confidence = 0.0

        try:
            # 检查双手是否都检测到
            if len(hands_data) >= 2:
                confidence += 0.3
            elif len(hands_data) == 1:
                confidence += 0.1

            # 分析手部位置相对于身体的合理性
            landmarks = pose_data.get("landmarks", [])
            if len(landmarks) >= 16:  # 确保有足够的关键点
                # 获取肩膀和手腕的关键点
                left_shoulder = landmarks[11] if len(landmarks) > 11 else None
                right_shoulder = landmarks[12] if len(landmarks) > 12 else None
                left_wrist = landmarks[15] if len(landmarks) > 15 else None
                right_wrist = landmarks[16] if len(landmarks) > 16 else None

                # 检查手部位置是否在合理范围内（消毒时手部通常在胸前）
                if left_wrist and right_wrist and left_shoulder and right_shoulder:
                    # 计算手部相对于肩膀的位置
                    shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
                    wrist_y = (left_wrist["y"] + right_wrist["y"]) / 2

                    # 消毒时手部应该在肩膀附近或稍下方
                    if abs(wrist_y - shoulder_y) < 0.2:  # 归一化坐标
                        confidence += 0.2

                    # 检查双手是否在身体中线附近（消毒时双手通常靠近）
                    hand_distance = abs(left_wrist["x"] - right_wrist["x"])
                    if hand_distance < 0.2:  # 归一化坐标
                        confidence += 0.3

            # 检查手部运动模式（基于手部检测数据）
            for hand_data in hands_data:
                if "landmarks" in hand_data and hand_data["landmarks"]:
                    # 分析手部姿态（简化版本）
                    landmarks = hand_data["landmarks"]
                    if len(landmarks) >= 21:  # MediaPipe手部有21个关键点
                        # 检查手部是否呈现消毒动作（手掌相对）
                        palm_orientation = self._analyze_palm_orientation(landmarks)
                        confidence += palm_orientation * 0.2

        except Exception as e:
            logger.error(f"Error in sanitizing pose analysis: {e}")

        return min(1.0, confidence)

    def _analyze_palm_orientation(self, hand_landmarks: List[Dict]) -> float:
        """
        分析手掌朝向（用于消毒检测）

        Args:
            hand_landmarks: 手部关键点

        Returns:
            手掌朝向评分 (0.0-1.0)
        """
        try:
            # 简化的手掌朝向分析
            if len(hand_landmarks) < 21:
                return 0.0

            # 使用手腕(0)和中指根部(9)来判断手掌朝向
            wrist = hand_landmarks[0]
            middle_mcp = hand_landmarks[9]

            # 计算手掌法向量的简化版本
            palm_vector_y = middle_mcp["y"] - wrist["y"]

            # 消毒时手掌通常朝向身体中心或向上
            if palm_vector_y < 0:  # 手掌向上
                return 0.8
            elif abs(palm_vector_y) < 0.1:  # 手掌水平
                return 0.6
            else:  # 手掌向下
                return 0.2

        except Exception as e:
            logger.error(f"Error in palm orientation analysis: {e}")

        return 0.0

    def detect_handwashing(
        self,
        person_bbox: List[int],
        hand_regions: List[Dict],
        track_id: Optional[int] = None,
        frame: Optional[Any] = None,
    ) -> float:
        """
        检测洗手行为（集成 MediaPipe 增强）

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表
            track_id: 追踪目标ID（用于运动分析）
            frame: 当前帧（用于姿态检测和 MediaPipe 增强）

        Returns:
            洗手行为置信度
        """
        if not hand_regions:
            return 0.0

        # 使用 MediaPipe 增强手部检测
        enhanced_hand_regions = hand_regions
        if frame is not None and isinstance(frame, np.ndarray):
            enhanced_hand_regions = self._enhance_hand_detection_with_mediapipe(
                frame, hand_regions
            )
            logger.debug(
                f"Enhanced hand regions: {len(enhanced_hand_regions)} hands detected"
            )

        confidence = 0.0

        # 使用高级检测方法
        if self.use_advanced_detection and track_id is not None:
            try:
                # 更新运动分析
                if self.motion_analyzer:
                    self.motion_analyzer.update_hand_motion(
                        track_id, enhanced_hand_regions
                    )
                    motion_confidence = self.motion_analyzer.analyze_handwashing(
                        track_id
                    )
                    confidence = max(confidence, motion_confidence)

                # 姿态检测增强
                if self.pose_detector and frame is not None:
                    pose_data = self.pose_detector.detect_pose(frame)
                    hands_data = self.pose_detector.detect_hands(frame)

                    if pose_data and hands_data:
                        pose_confidence = self._analyze_handwash_pose(
                            pose_data, hands_data, person_bbox
                        )
                        confidence = max(confidence, pose_confidence)

                logger.debug(
                    f"Advanced handwashing detection for track {track_id}: {confidence:.3f}"
                )

            except Exception as e:
                logger.error(f"Error in advanced handwashing detection: {e}")
                # 回退到基础检测
                confidence = self._basic_handwashing_detection(
                    person_bbox, enhanced_hand_regions
                )
        else:
            # 基础检测方法（使用增强后的手部区域）
            confidence = self._basic_handwashing_detection(
                person_bbox, enhanced_hand_regions
            )

        return confidence

    def detect_sanitizing(
        self,
        person_bbox: List[int],
        hand_regions: List[Dict],
        track_id: Optional[int] = None,
        frame: Optional[Any] = None,
    ) -> float:
        """
        检测手部消毒行为

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表
            track_id: 追踪目标ID（用于运动分析）
            frame: 当前帧（用于姿态检测）

        Returns:
            消毒行为置信度
        """
        if not hand_regions:
            return 0.0

        confidence = 0.0

        # 使用高级检测方法
        if self.use_advanced_detection and track_id is not None:
            try:
                # 更新运动分析
                if self.motion_analyzer:
                    self.motion_analyzer.update_hand_motion(track_id, hand_regions)
                    motion_confidence = self.motion_analyzer.analyze_sanitizing(
                        track_id
                    )
                    confidence = max(confidence, motion_confidence)

                # 姿态检测增强
                if self.pose_detector and frame is not None:
                    pose_data = self.pose_detector.detect_pose(frame)
                    hands_data = self.pose_detector.detect_hands(frame)

                    if pose_data and hands_data:
                        pose_confidence = self._analyze_sanitizing_pose(
                            pose_data, hands_data, person_bbox
                        )
                        confidence = max(confidence, pose_confidence)

                logger.debug(
                    f"Advanced sanitizing detection for track {track_id}: {confidence:.3f}"
                )

            except Exception as e:
                logger.error(f"Error in advanced sanitizing detection: {e}")
                # 回退到基础检测
                confidence = self._basic_sanitizing_detection(person_bbox, hand_regions)
        else:
            # 基础检测方法
            confidence = self._basic_sanitizing_detection(person_bbox, hand_regions)

        return confidence

    def _basic_handwashing_detection(
        self, person_bbox: List[int], hand_regions: List[Dict]
    ) -> float:
        """
        基础洗手检测方法

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表

        Returns:
            洗手行为置信度
        """
        if not hand_regions:
            return 0.0

        confidence = 0.0
        valid_hands = 0

        # 检查手部位置和动作
        for hand_region in hand_regions:
            if "bbox" not in hand_region:
                continue

            hand_bbox = hand_region["bbox"]
            if len(hand_bbox) < 4:
                continue

            # 检查手部是否在合适的位置（身体中下部）
            person_height = person_bbox[3] - person_bbox[1]
            person_width = person_bbox[2] - person_bbox[0]

            if person_height <= 0 or person_width <= 0:
                continue

            hand_center_x = (hand_bbox[0] + hand_bbox[2]) / 2
            hand_center_y = (hand_bbox[1] + hand_bbox[3]) / 2

            # 相对于人体的位置
            relative_x = (hand_center_x - person_bbox[0]) / person_width
            relative_y = (hand_center_y - person_bbox[1]) / person_height

            # 洗手时手部通常在身体中部偏下，且在身体范围内
            position_score = 0.0
            if 0.3 <= relative_y <= 0.8 and 0.1 <= relative_x <= 0.9:
                position_score = 0.4

            # 检查手部大小（洗手时手部通常比较明显）
            hand_area = (hand_bbox[2] - hand_bbox[0]) * (hand_bbox[3] - hand_bbox[1])
            person_area = person_height * person_width
            area_ratio = hand_area / person_area if person_area > 0 else 0

            size_score = 0.0
            if 0.005 <= area_ratio <= 0.1:  # 手部占人体面积的合理比例
                size_score = 0.3

            # 检查手部动作特征
            motion_score = self._analyze_hand_motion(hand_region)

            # 综合评分
            hand_confidence = position_score + size_score + motion_score
            confidence = max(confidence, hand_confidence)
            valid_hands += 1

        # 如果检测到多只手，稍微提高置信度
        if valid_hands >= 2:
            confidence = min(1.0, confidence * 1.2)

        return confidence

    def _analyze_handwash_pose(
        self, pose_data: Dict, hands_data: List[Dict], person_bbox: List[int]
    ) -> float:
        """
        基于姿态数据分析洗手行为

        Args:
            pose_data: 姿态检测数据
            hands_data: 手部检测数据
            person_bbox: 人体边界框

        Returns:
            姿态分析置信度
        """
        if not pose_data or not hands_data:
            return 0.0

        confidence = 0.0

        try:
            # 检查双手是否都检测到
            if len(hands_data) >= 2:
                confidence += 0.3
            elif len(hands_data) == 1:
                confidence += 0.1

            # 分析手部位置相对于身体的合理性
            landmarks = pose_data.get("landmarks", [])
            if len(landmarks) >= 16:  # 确保有足够的关键点
                # 获取肩膀和手腕的关键点（MediaPipe Pose的索引）
                left_shoulder = landmarks[11] if len(landmarks) > 11 else None
                right_shoulder = landmarks[12] if len(landmarks) > 12 else None
                left_wrist = landmarks[15] if len(landmarks) > 15 else None
                right_wrist = landmarks[16] if len(landmarks) > 16 else None

                # 检查手部位置是否在合理范围内
                if left_wrist and right_wrist and left_shoulder and right_shoulder:
                    # 计算手部相对于肩膀的位置
                    shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
                    wrist_y = (left_wrist["y"] + right_wrist["y"]) / 2

                    # 洗手时手部应该在肩膀下方
                    if wrist_y > shoulder_y:
                        confidence += 0.2

                    # 检查双手是否在身体中线附近（洗手时双手通常靠近）
                    hand_distance = abs(left_wrist["x"] - right_wrist["x"])
                    if hand_distance < 0.3:  # 归一化坐标
                        confidence += 0.2

            # 检查手部运动模式（基于手部检测数据）
            for hand_data in hands_data:
                if "landmarks" in hand_data and hand_data["landmarks"]:
                    # 分析手部姿态（简化版本）
                    landmarks = hand_data["landmarks"]
                    if len(landmarks) >= 21:  # MediaPipe手部有21个关键点
                        # 检查手指是否展开（洗手时通常手指展开）
                        finger_spread = self._analyze_finger_spread(landmarks)
                        confidence += finger_spread * 0.3

        except Exception as e:
            logger.error(f"Error in pose analysis: {e}")

        return min(1.0, confidence)

    def _analyze_finger_spread(self, hand_landmarks: List[Dict]) -> float:
        """
        分析手指展开程度

        Args:
            hand_landmarks: 手部关键点

        Returns:
            手指展开程度 (0.0-1.0)
        """
        try:
            # 简化的手指展开分析
            # 计算手指尖端之间的距离
            if len(hand_landmarks) < 21:
                return 0.0

            # MediaPipe手部关键点索引：拇指尖(4), 食指尖(8), 中指尖(12), 无名指尖(16), 小指尖(20)
            fingertips = [4, 8, 12, 16, 20]

            distances = []
            for i in range(len(fingertips)):
                for j in range(i + 1, len(fingertips)):
                    if fingertips[i] < len(hand_landmarks) and fingertips[j] < len(
                        hand_landmarks
                    ):
                        p1 = hand_landmarks[fingertips[i]]
                        p2 = hand_landmarks[fingertips[j]]
                        dist = (
                            (p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2
                        ) ** 0.5
                        distances.append(dist)

            if distances:
                avg_distance = sum(distances) / len(distances)
                # 归一化到0-1范围（经验值）
                spread_factor = min(1.0, avg_distance / 0.3)
                return spread_factor

        except Exception as e:
            logger.error(f"Error in finger spread analysis: {e}")

        return 0.0

    def _enhance_hand_detection_with_mediapipe(
        self, frame: np.ndarray, hand_regions: List[Dict]
    ) -> List[Dict]:
        """
        使用 MediaPipe 增强手部检测

        Args:
            frame: 输入图像帧
            hand_regions: 原始手部检测结果

        Returns:
            增强后的手部检测结果
        """
        if not self.use_mediapipe or self.hands_detector is None or frame is None:
            return hand_regions

        try:
            # 转换图像格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(rgb_frame)

            enhanced_regions = []

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 计算手部边界框
                    h, w, _ = frame.shape
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]

                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # 创建增强的手部区域信息
                    enhanced_region = {
                        "bbox": [x_min, y_min, x_max, y_max],
                        "landmarks": [
                            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                            for landmark in hand_landmarks.landmark
                        ],
                        "confidence": 0.8,  # MediaPipe 检测的置信度通常较高
                        "source": "mediapipe",
                    }

                    # 如果有对应的原始检测结果，合并信息
                    if idx < len(hand_regions):
                        original_region = hand_regions[idx]
                        enhanced_region.update(
                            {
                                "original_bbox": original_region.get("bbox"),
                                "original_confidence": original_region.get(
                                    "confidence", 0.0
                                ),
                            }
                        )

                    enhanced_regions.append(enhanced_region)

            # 如果 MediaPipe 没有检测到手部，返回原始结果
            return enhanced_regions if enhanced_regions else hand_regions

        except Exception as e:
            logger.error(f"Error in MediaPipe hand detection enhancement: {e}")
            return hand_regions

    def _analyze_hand_motion(self, hand_region: Dict) -> float:
        """
        分析手部运动模式（增强版本）

        Args:
            hand_region: 手部区域信息

        Returns:
            运动模式置信度
        """
        # 基于手部区域特征进行运动分析
        motion_score = 0.0

        bbox = hand_region.get("bbox", [0, 0, 0, 0])
        hand_width = bbox[2] - bbox[0]
        hand_height = bbox[3] - bbox[1]

        # 检查手部区域大小的合理性
        if hand_width > 20 and hand_height > 20:
            # 基于手部尺寸给出基础分数
            motion_score += 0.3

            # 检查手部宽高比（洗手时手部通常呈现一定形状）
            if hand_width > 0 and hand_height > 0:
                aspect_ratio = hand_width / hand_height
                # 合理的手部宽高比范围
                if 0.5 <= aspect_ratio <= 2.0:
                    motion_score += 0.2

        # 检查是否有 MediaPipe 关键点信息（更准确）
        if "landmarks" in hand_region and hand_region["landmarks"]:
            landmarks = hand_region["landmarks"]
            if (
                isinstance(landmarks, list) and len(landmarks) >= 21
            ):  # MediaPipe 手部有21个关键点
                # MediaPipe 关键点信息更准确，给予更高分数
                motion_score += 0.3

                # 分析手指展开程度（洗手时手指通常展开）
                finger_spread_score = self._analyze_mediapipe_finger_spread(landmarks)
                motion_score += finger_spread_score * 0.2
            elif isinstance(landmarks, list) and len(landmarks) > 0:
                # 有关键点信息说明手部检测质量较好
                motion_score += 0.1

        # 检查置信度信息
        if "confidence" in hand_region:
            conf = hand_region["confidence"]
            if isinstance(conf, (int, float)) and conf > 0.5:
                motion_score += 0.1

        # MediaPipe 检测的结果给予额外加分
        if hand_region.get("source") == "mediapipe":
            motion_score += 0.1

        return min(motion_score, 0.8)  # 提高最大置信度限制

    def _analyze_mediapipe_finger_spread(self, landmarks: List[Dict]) -> float:
        """
        分析 MediaPipe 手部关键点的手指展开程度

        Args:
            landmarks: MediaPipe 手部关键点

        Returns:
            手指展开程度 (0.0-1.0)
        """
        try:
            if len(landmarks) < 21:
                return 0.0

            # MediaPipe 手部关键点索引：拇指尖(4), 食指尖(8), 中指尖(12), 无名指尖(16), 小指尖(20)
            fingertips_indices = [4, 8, 12, 16, 20]

            distances = []
            for i in range(len(fingertips_indices)):
                for j in range(i + 1, len(fingertips_indices)):
                    idx1, idx2 = fingertips_indices[i], fingertips_indices[j]
                    if idx1 < len(landmarks) and idx2 < len(landmarks):
                        p1 = landmarks[idx1]
                        p2 = landmarks[idx2]
                        dist = (
                            (p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2
                        ) ** 0.5
                        distances.append(dist)

            if distances:
                avg_distance = sum(distances) / len(distances)
                # 归一化到0-1范围（经验值，基于 MediaPipe 归一化坐标）
                spread_factor = min(1.0, avg_distance / 0.2)
                return spread_factor

        except Exception as e:
            logger.error(f"Error in MediaPipe finger spread analysis: {e}")

        return 0.0

    def _get_bbox_center(self, bbox: List[int]) -> Tuple[float, float]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update_behavior(
        self,
        track_id: int,
        person_bbox: List[int],
        additional_info: Optional[Dict] = None,
    ) -> Dict[str, BehaviorState]:
        """
        更新指定追踪目标的行为状态

        Args:
            track_id: 追踪目标ID
            person_bbox: 人体边界框
            additional_info: 额外信息（头部、手部区域等）

        Returns:
            当前行为状态字典
        """
        current_behaviors = {}

        # 检测发网
        head_region = additional_info.get("head_region") if additional_info else None
        hairnet_conf = self.detect_hairnet(person_bbox, head_region)
        if hairnet_conf > self.confidence_threshold:
            current_behaviors["hairnet"] = BehaviorState("hairnet", hairnet_conf)

        # 检测洗手
        hand_regions = (
            additional_info.get("hand_regions", []) if additional_info else []
        )
        handwashing_conf = self.detect_handwashing(person_bbox, hand_regions)
        if handwashing_conf > self.confidence_threshold:
            current_behaviors["handwashing"] = BehaviorState(
                "handwashing", handwashing_conf
            )

        # 检测消毒
        sanitizing_conf = self.detect_sanitizing(person_bbox, hand_regions)
        if sanitizing_conf > self.confidence_threshold:
            current_behaviors["sanitizing"] = BehaviorState(
                "sanitizing", sanitizing_conf
            )

        # 更新行为历史
        self.behavior_history[track_id].append(current_behaviors)

        # 更新活跃行为
        self._update_active_behaviors(track_id, current_behaviors)

        return current_behaviors

    def _update_active_behaviors(
        self, track_id: int, current_behaviors: Dict[str, BehaviorState]
    ):
        """
        更新活跃行为状态

        Args:
            track_id: 追踪目标ID
            current_behaviors: 当前检测到的行为
        """
        if track_id not in self.active_behaviors:
            self.active_behaviors[track_id] = {}

        # 更新或添加新行为
        for behavior_type, behavior_state in current_behaviors.items():
            if behavior_type in self.active_behaviors[track_id]:
                # 更新现有行为
                self.active_behaviors[track_id][behavior_type].update(
                    behavior_state.confidence
                )
            else:
                # 添加新行为
                self.active_behaviors[track_id][behavior_type] = behavior_state

        # 检查行为是否结束
        to_remove = []
        for behavior_type, behavior_state in self.active_behaviors[track_id].items():
            if behavior_type not in current_behaviors:
                # 行为不再检测到，标记为结束
                behavior_state.end()

                # 检查是否满足最小持续时间
                rules = self.behavior_rules.get(behavior_type, {})
                min_duration = rules.get("min_duration", 0.0)

                if behavior_state.duration >= min_duration:
                    logger.info(
                        f"Behavior {behavior_type} completed for track {track_id}, "
                        f"duration: {behavior_state.duration:.2f}s"
                    )

                to_remove.append(behavior_type)

        # 移除已结束的行为
        for behavior_type in to_remove:
            del self.active_behaviors[track_id][behavior_type]

    def get_behavior_summary(self, track_id: int) -> Dict:
        """
        获取指定追踪目标的行为摘要

        Args:
            track_id: 追踪目标ID

        Returns:
            行为摘要信息
        """
        summary = {"track_id": track_id, "active_behaviors": {}, "behavior_history": []}

        # 活跃行为
        if track_id in self.active_behaviors:
            for behavior_type, behavior_state in self.active_behaviors[
                track_id
            ].items():
                summary["active_behaviors"][behavior_type] = {
                    "confidence": behavior_state.confidence,
                    "duration": behavior_state.duration,
                    "is_active": behavior_state.is_active,
                }

        # 行为历史
        if track_id in self.behavior_history:
            summary["behavior_history"] = list(self.behavior_history[track_id])

        return summary

    def reset_track(self, track_id: int):
        """重置指定追踪目标的行为状态"""
        if track_id in self.active_behaviors:
            del self.active_behaviors[track_id]
        if track_id in self.behavior_history:
            self.behavior_history[track_id].clear()

        logger.info(f"Behavior state reset for track {track_id}")

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Behavior confidence threshold set to {self.confidence_threshold}")
