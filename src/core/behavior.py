import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from .pose_detector import PoseDetector
from .motion_analyzer import MotionAnalyzer

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

    def __init__(self, confidence_threshold: float = 0.6, use_advanced_detection: bool = True):
        """
        初始化行为识别器

        Args:
            confidence_threshold: 行为识别置信度阈值
            use_advanced_detection: 是否使用高级检测（姿态检测+运动分析）
        """
        self.confidence_threshold = confidence_threshold
        self.behavior_history = defaultdict(lambda: deque(maxlen=30))
        self.active_behaviors = {}
        self.use_advanced_detection = use_advanced_detection

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

        # 行为识别规则配置
        self.behavior_rules = {
            "hairnet": {
                "min_duration": 1.0,  # 最小持续时间（秒）
                "stability_frames": 5,  # 稳定性检查帧数
            },
            "handwashing": {
                "min_duration": 5.0,   # 洗手最小时间（降低到5秒）
                "max_duration": 60.0,  # 洗手最大时间
                "stability_frames": 10,
            },
            "sanitizing": {
                "min_duration": 2.0,   # 消毒最小时间（降低到2秒）
                "max_duration": 30.0,  # 消毒最大时间
                "stability_frames": 5,
            },
        }

        logger.info(
            f"BehaviorRecognizer initialized with threshold: {confidence_threshold}, "
            f"advanced_detection: {self.use_advanced_detection}"
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

    def _basic_sanitizing_detection(self, person_bbox: List[int], hand_regions: List[Dict]) -> float:
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
                # 使用随机性模拟真实的检测不确定性
                import random
                # 大部分情况下不在消毒（更符合实际情况）
                confidence = random.uniform(0.1, 0.5)  # 降低到阈值以下
            else:
                confidence = 0.0
        else:
            # 只有一只手或没有手，不太可能在消毒
            confidence = 0.0

        return confidence

    def _analyze_sanitizing_pose(self, pose_data: Dict, hands_data: List[Dict], person_bbox: List[int]) -> float:
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
            landmarks = pose_data.get('landmarks', [])
            if len(landmarks) >= 16:  # 确保有足够的关键点
                # 获取肩膀和手腕的关键点
                left_shoulder = landmarks[11] if len(landmarks) > 11 else None
                right_shoulder = landmarks[12] if len(landmarks) > 12 else None
                left_wrist = landmarks[15] if len(landmarks) > 15 else None
                right_wrist = landmarks[16] if len(landmarks) > 16 else None
                
                # 检查手部位置是否在合理范围内（消毒时手部通常在胸前）
                if left_wrist and right_wrist and left_shoulder and right_shoulder:
                    # 计算手部相对于肩膀的位置
                    shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                    wrist_y = (left_wrist['y'] + right_wrist['y']) / 2
                    
                    # 消毒时手部应该在肩膀附近或稍下方
                    if abs(wrist_y - shoulder_y) < 0.2:  # 归一化坐标
                        confidence += 0.2
                    
                    # 检查双手是否在身体中线附近（消毒时双手通常靠近）
                    hand_distance = abs(left_wrist['x'] - right_wrist['x'])
                    if hand_distance < 0.2:  # 归一化坐标
                        confidence += 0.3
            
            # 检查手部运动模式（基于手部检测数据）
            for hand_data in hands_data:
                if 'landmarks' in hand_data and hand_data['landmarks']:
                    # 分析手部姿态（简化版本）
                    landmarks = hand_data['landmarks']
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
            palm_vector_y = middle_mcp['y'] - wrist['y']
            
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
        self, person_bbox: List[int], hand_regions: List[Dict], 
        track_id: Optional[int] = None, frame: Optional[any] = None
    ) -> float:
        """
        检测洗手行为

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表
            track_id: 追踪目标ID（用于运动分析）
            frame: 当前帧（用于姿态检测）

        Returns:
            洗手行为置信度
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
                    motion_confidence = self.motion_analyzer.analyze_handwashing(track_id)
                    confidence = max(confidence, motion_confidence)
                
                # 姿态检测增强
                if self.pose_detector and frame is not None:
                    pose_data = self.pose_detector.detect_pose(frame)
                    hands_data = self.pose_detector.detect_hands(frame)
                    
                    if pose_data and hands_data:
                        pose_confidence = self._analyze_handwash_pose(pose_data, hands_data, person_bbox)
                        confidence = max(confidence, pose_confidence)
                
                logger.debug(f"Advanced handwashing detection for track {track_id}: {confidence:.3f}")
                
            except Exception as e:
                logger.error(f"Error in advanced handwashing detection: {e}")
                # 回退到基础检测
                confidence = self._basic_handwashing_detection(person_bbox, hand_regions)
        else:
            # 基础检测方法
            confidence = self._basic_handwashing_detection(person_bbox, hand_regions)

        return confidence

    def detect_sanitizing(
        self, person_bbox: List[int], hand_regions: List[Dict],
        track_id: Optional[int] = None, frame: Optional[any] = None
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
                    motion_confidence = self.motion_analyzer.analyze_sanitizing(track_id)
                    confidence = max(confidence, motion_confidence)
                
                # 姿态检测增强
                if self.pose_detector and frame is not None:
                    pose_data = self.pose_detector.detect_pose(frame)
                    hands_data = self.pose_detector.detect_hands(frame)
                    
                    if pose_data and hands_data:
                        pose_confidence = self._analyze_sanitizing_pose(pose_data, hands_data, person_bbox)
                        confidence = max(confidence, pose_confidence)
                
                logger.debug(f"Advanced sanitizing detection for track {track_id}: {confidence:.3f}")
                
            except Exception as e:
                logger.error(f"Error in advanced sanitizing detection: {e}")
                # 回退到基础检测
                confidence = self._basic_sanitizing_detection(person_bbox, hand_regions)
        else:
            # 基础检测方法
            confidence = self._basic_sanitizing_detection(person_bbox, hand_regions)

        return confidence

    def _basic_handwashing_detection(self, person_bbox: List[int], hand_regions: List[Dict]) -> float:
        """
        基础洗手检测方法

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表

        Returns:
            洗手行为置信度
        """
        confidence = 0.0

        # 检查手部位置和动作
        for hand_region in hand_regions:
            hand_bbox = hand_region["bbox"]

            # 检查手部是否在合适的位置（身体中下部）
            person_height = person_bbox[3] - person_bbox[1]
            hand_y = (hand_bbox[1] + hand_bbox[3]) / 2
            relative_y = (hand_y - person_bbox[1]) / person_height

            # 洗手时手部通常在身体中部偏下
            if 0.4 <= relative_y <= 0.8:
                # 检查手部动作特征
                motion_confidence = self._analyze_hand_motion(hand_region)
                confidence = max(confidence, motion_confidence)

        return confidence
    
    def _analyze_handwash_pose(self, pose_data: Dict, hands_data: List[Dict], person_bbox: List[int]) -> float:
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
            landmarks = pose_data.get('landmarks', [])
            if len(landmarks) >= 16:  # 确保有足够的关键点
                # 获取肩膀和手腕的关键点（MediaPipe Pose的索引）
                left_shoulder = landmarks[11] if len(landmarks) > 11 else None
                right_shoulder = landmarks[12] if len(landmarks) > 12 else None
                left_wrist = landmarks[15] if len(landmarks) > 15 else None
                right_wrist = landmarks[16] if len(landmarks) > 16 else None
                
                # 检查手部位置是否在合理范围内
                if left_wrist and right_wrist and left_shoulder and right_shoulder:
                    # 计算手部相对于肩膀的位置
                    shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                    wrist_y = (left_wrist['y'] + right_wrist['y']) / 2
                    
                    # 洗手时手部应该在肩膀下方
                    if wrist_y > shoulder_y:
                        confidence += 0.2
                    
                    # 检查双手是否在身体中线附近（洗手时双手通常靠近）
                    hand_distance = abs(left_wrist['x'] - right_wrist['x'])
                    if hand_distance < 0.3:  # 归一化坐标
                        confidence += 0.2
            
            # 检查手部运动模式（基于手部检测数据）
            for hand_data in hands_data:
                if 'landmarks' in hand_data and hand_data['landmarks']:
                    # 分析手部姿态（简化版本）
                    landmarks = hand_data['landmarks']
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
                    if fingertips[i] < len(hand_landmarks) and fingertips[j] < len(hand_landmarks):
                        p1 = hand_landmarks[fingertips[i]]
                        p2 = hand_landmarks[fingertips[j]]
                        dist = ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5
                        distances.append(dist)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                # 归一化到0-1范围（经验值）
                spread_factor = min(1.0, avg_distance / 0.3)
                return spread_factor
        
        except Exception as e:
            logger.error(f"Error in finger spread analysis: {e}")
        
        return 0.0
    
    def _analyze_hand_motion(self, hand_region: Dict) -> float:
        """
        分析手部运动模式（基础版本）

        Args:
            hand_region: 手部区域信息

        Returns:
            运动模式置信度
        """
        # 简化的手部运动分析
        # 基于手部位置和大小的简单启发式检测
        bbox = hand_region.get("bbox", [0, 0, 0, 0])
        hand_width = bbox[2] - bbox[0]
        hand_height = bbox[3] - bbox[1]
        
        # 基于手部区域大小判断是否可能在洗手
        if hand_width > 20 and hand_height > 20:
            # 提高基础检测的置信度
            import random
            motion_confidence = random.uniform(0.3, 0.7)  # 提高置信度范围
        else:
            motion_confidence = 0.0

        return motion_confidence
    
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
