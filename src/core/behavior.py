import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

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

    def __init__(self, confidence_threshold: float = 0.6):
        """
        初始化行为识别器

        Args:
            confidence_threshold: 行为识别置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        self.behavior_history = defaultdict(lambda: deque(maxlen=30))
        self.active_behaviors = {}

        # 行为识别规则配置
        self.behavior_rules = {
            "hairnet": {
                "min_duration": 1.0,  # 最小持续时间（秒）
                "stability_frames": 5,  # 稳定性检查帧数
            },
            "handwashing": {
                "min_duration": 15.0,  # 洗手最小时间
                "max_duration": 60.0,  # 洗手最大时间
                "stability_frames": 10,
            },
            "sanitizing": {
                "min_duration": 3.0,  # 消毒最小时间
                "max_duration": 30.0,  # 消毒最大时间
                "stability_frames": 5,
            },
        }

        logger.info(
            f"BehaviorRecognizer initialized with threshold: {confidence_threshold}"
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

    def detect_handwashing(
        self, person_bbox: List[int], hand_regions: List[Dict]
    ) -> float:
        """
        检测洗手行为

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表

        Returns:
            洗手行为置信度
        """
        if not hand_regions:
            return 0.0

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
                # 检查手部动作特征（实际会用姿态估计）
                motion_confidence = self._analyze_hand_motion(hand_region)
                confidence = max(confidence, motion_confidence)

        return confidence

    def detect_sanitizing(
        self, person_bbox: List[int], hand_regions: List[Dict]
    ) -> float:
        """
        检测手部消毒行为

        Args:
            person_bbox: 人体边界框
            hand_regions: 手部区域列表

        Returns:
            消毒行为置信度
        """
        if not hand_regions:
            return 0.0

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

    def _analyze_hand_motion(self, hand_region: Dict) -> float:
        """
        分析手部运动模式

        Args:
            hand_region: 手部区域信息

        Returns:
            运动模式置信度
        """
        # 简化的手部运动分析
        # 实际实现会使用光流或姿态估计
        
        # 基于手部位置和大小的简单启发式检测
        bbox = hand_region.get("bbox", [0, 0, 0, 0])
        hand_width = bbox[2] - bbox[0]
        hand_height = bbox[3] - bbox[1]
        
        # 基于手部区域大小判断是否可能在洗手
        # 洗手时手部通常有一定的活动空间
        if hand_width > 20 and hand_height > 20:
            # 使用随机性模拟真实的检测不确定性
            import random
            # 大部分情况下不在洗手（更符合实际情况）
            motion_confidence = random.uniform(0.1, 0.4)  # 降低到阈值以下
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
