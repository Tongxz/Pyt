import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config.unified_params import get_unified_params

logger = logging.getLogger(__name__)


class MotionTracker:
    """运动轨迹跟踪器"""

    def __init__(self, max_history: int = 30):
        """
        初始化运动跟踪器

        Args:
            max_history: 最大历史记录数量
        """
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.velocity_history = deque(maxlen=max_history)
        self.last_update_time = None

    def update(self, position: Tuple[float, float], timestamp: Optional[float] = None):
        """
        更新位置信息

        Args:
            position: 当前位置 (x, y)
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()

        self.position_history.append((position, timestamp))

        # 计算速度
        if len(self.position_history) >= 2:
            prev_pos, prev_time = self.position_history[-2]
            curr_pos, curr_time = self.position_history[-1]

            dt = curr_time - prev_time
            if dt > 0:
                velocity = (
                    (curr_pos[0] - prev_pos[0]) / dt,
                    (curr_pos[1] - prev_pos[1]) / dt,
                )
                self.velocity_history.append((velocity, curr_time))

        self.last_update_time = timestamp

    def get_motion_stats(self) -> Dict:
        """
        获取运动统计信息

        Returns:
            运动统计字典
        """
        if len(self.position_history) < 2:
            # 当数据不足时返回默认统计（新增字段保持为0）
            return {
                "avg_speed": 0.0,
                "horizontal_movement": 0.0,
                "vertical_movement": 0.0,
                "movement_ratio": 0.0,
                "position_variance": 0.0,
                "horizontal_move_std": 0.0,
                "vertical_move_std": 0.0,
                "move_frequency_hz": 0.0,
            }

        # 计算平均速度
        speeds: List[float] = []
        horizontal_movements: List[float] = []
        vertical_movements: List[float] = []

        for velocity, _ in self.velocity_history:
            speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            speeds.append(speed)
            horizontal_movements.append(abs(velocity[0]))
            vertical_movements.append(abs(velocity[1]))

        avg_speed = np.mean(speeds) if speeds else 0.0
        avg_horizontal = np.mean(horizontal_movements) if horizontal_movements else 0.0
        avg_vertical = np.mean(vertical_movements) if vertical_movements else 0.0

        # 计算运动比例（水平/垂直）
        movement_ratio = avg_horizontal / (avg_vertical + 1e-6)

        # 计算位置方差
        positions = [pos for pos, _ in self.position_history]
        if len(positions) > 1:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            position_variance = np.var(x_coords) + np.var(y_coords)
        else:
            position_variance = 0.0

        # 计算运动标准差
        horizontal_move_std = np.std(horizontal_movements) if horizontal_movements else 0.0
        vertical_move_std = np.std(vertical_movements) if vertical_movements else 0.0

        # 计算运动采样频率（Hz）
        total_time = self.position_history[-1][1] - self.position_history[0][1]
        move_frequency_hz = (len(self.position_history) - 1) / total_time if total_time > 0 else 0.0

        return {
            "avg_speed": avg_speed,
            "horizontal_movement": avg_horizontal,
            "vertical_movement": avg_vertical,
            "movement_ratio": movement_ratio,
            "position_variance": position_variance,
            "horizontal_move_std": horizontal_move_std,
            "vertical_move_std": vertical_move_std,
            "move_frequency_hz": move_frequency_hz,
        }

    def clear(self):
        """清空历史记录"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.last_update_time = None


class MotionAnalyzer:
    """运动分析器

    分析手部运动模式，识别洗手和消毒行为
    """

    def __init__(self):
        """
        初始化运动分析器
        """
        self.hand_trackers = (
            {}
        )  # track_id -> {'left': MotionTracker, 'right': MotionTracker}

        # 从统一配置中加载阈值
        params = get_unified_params()
        self.detection_rules = params.detection_rules

        # 洗手行为特征阈值（调整为更宽松的阈值）
        self.handwash_thresholds = {
            "min_movement_ratio": 0.8,  # 水平运动/垂直运动比例（降低要求）
            "min_avg_speed": 0.005,  # 最小平均速度（降低要求）
            "max_avg_speed": 0.8,  # 最大平均速度（提高上限）
            "min_position_variance": 0.0005,  # 最小位置方差（降低要求）
            "min_duration": 2.0,  # 最小持续时间（降低到2秒）
            "horizontal_move_std": self.detection_rules.horizontal_move_std,
            "min_move_frequency_hz": self.detection_rules.min_move_frequency_hz,
        }

        # 消毒行为特征阈值
        self.sanitize_thresholds = {
            "max_hand_distance": 0.15,  # 双手最大距离（归一化坐标）
            "min_movement_ratio": 0.5,  # 较小的运动比例
            "min_avg_speed": 0.005,  # 最小平均速度
            "max_avg_speed": 0.3,  # 最大平均速度
            "min_duration": 2.0,  # 最小持续时间（秒）
        }

        logger.info("MotionAnalyzer initialized")

    def update_hand_motion(self, track_id: int, hands_data: List[Dict]):
        """
        更新手部运动数据

        Args:
            track_id: 追踪目标ID
            hands_data: 手部检测数据列表
        """
        if track_id not in self.hand_trackers:
            self.hand_trackers[track_id] = {
                "left": MotionTracker(),
                "right": MotionTracker(),
                "unknown": MotionTracker(),
            }

        current_time = time.time()

        # 重置当前帧的手部状态
        hands_found = {"left": False, "right": False, "unknown": False}

        for hand_data in hands_data:
            try:
                hand_label = hand_data.get("label", "Unknown").lower()
                if hand_label not in ["left", "right"]:
                    hand_label = "unknown"

                # 获取手部中心点
                center_x, center_y = None, None

                if "landmarks" in hand_data and hand_data["landmarks"]:
                    # 使用关键点计算中心
                    landmarks = hand_data["landmarks"]
                    if isinstance(landmarks, list) and len(landmarks) > 0:
                        # 确保landmarks是字典列表格式
                        if isinstance(landmarks[0], dict):
                            center_x = float(
                                np.mean([lm["x"] for lm in landmarks if "x" in lm])
                            )
                            center_y = float(
                                np.mean([lm["y"] for lm in landmarks if "y" in lm])
                            )
                        else:
                            logger.warning(
                                f"Invalid landmarks format: {type(landmarks[0])}"
                            )
                            continue
                    else:
                        logger.warning(f"Empty or invalid landmarks: {landmarks}")
                        continue

                elif "bbox" in hand_data and hand_data["bbox"]:
                    # 使用边界框计算中心
                    bbox = hand_data["bbox"]
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        # 如果bbox是像素坐标，需要归一化（假设图像尺寸）
                        if center_x > 1.0 or center_y > 1.0:
                            # 假设图像尺寸为640x480（需要从外部传入实际尺寸）
                            center_x /= 640
                            center_y /= 480
                    else:
                        logger.warning(f"Invalid bbox format: {bbox}")
                        continue
                else:
                    logger.warning(
                        f"No valid position data in hand_data: {hand_data.keys()}"
                    )
                    continue

                # 验证中心点数据
                if center_x is not None and center_y is not None:
                    # 更新对应手部的运动轨迹
                    self.hand_trackers[track_id][hand_label].update(
                        (float(center_x), float(center_y)), current_time
                    )
                    hands_found[hand_label] = True
                else:
                    logger.warning(
                        f"Failed to calculate center point for hand: {hand_label}"
                    )

            except Exception as e:
                logger.error(f"Error processing hand data: {e}, hand_data: {hand_data}")
                continue
            hands_found[hand_label] = True

    def analyze_handwashing(self, track_id: int) -> float:
        """
        分析洗手行为

        Args:
            track_id: 追踪目标ID

        Returns:
            洗手行为置信度 (0.0-1.0)
        """
        if track_id not in self.hand_trackers:
            return 0.0

        trackers = self.hand_trackers[track_id]
        confidence = 0.0

        # 检查每只手的运动模式
        hand_confidences = []

        for hand_label, tracker in trackers.items():
            if len(tracker.position_history) < 5:  # 需要足够的历史数据
                continue

            motion_stats = tracker.get_motion_stats()
            hand_conf = self._evaluate_handwash_motion(motion_stats)

            if hand_conf > 0:
                hand_confidences.append(hand_conf)
                logger.debug(f"Hand {hand_label} handwash confidence: {hand_conf:.3f}")

        # 如果检测到双手都在洗手，提高置信度
        if len(hand_confidences) >= 2:
            confidence = min(1.0, np.mean(hand_confidences) * 1.2)
        elif len(hand_confidences) == 1:
            confidence = hand_confidences[0] * 0.8

        return confidence

    def analyze_sanitizing(self, track_id: int) -> float:
        """
        分析消毒行为

        Args:
            track_id: 追踪目标ID

        Returns:
            消毒行为置信度 (0.0-1.0)
        """
        if track_id not in self.hand_trackers:
            return 0.0

        trackers = self.hand_trackers[track_id]

        # 需要检测到双手
        active_hands = []
        for hand_label, tracker in trackers.items():
            if len(tracker.position_history) >= 3:
                active_hands.append((hand_label, tracker))

        if len(active_hands) < 2:
            return 0.0

        # 计算双手距离
        hand1_pos = active_hands[0][1].position_history[-1][0]
        hand2_pos = active_hands[1][1].position_history[-1][0]

        distance = np.sqrt(
            (hand1_pos[0] - hand2_pos[0]) ** 2 + (hand1_pos[1] - hand2_pos[1]) ** 2
        )

        # 检查双手是否足够接近
        if distance > self.sanitize_thresholds["max_hand_distance"]:
            return 0.0

        # 分析双手的运动模式
        motion_confidences = []

        for hand_label, tracker in active_hands:
            motion_stats = tracker.get_motion_stats()
            hand_conf = self._evaluate_sanitize_motion(motion_stats)
            motion_confidences.append(hand_conf)

        if motion_confidences:
            # 双手运动模式的平均置信度
            base_confidence = np.mean(motion_confidences)

            # 根据双手距离调整置信度
            distance_factor = 1.0 - (
                distance / self.sanitize_thresholds["max_hand_distance"]
            )

            confidence = base_confidence * distance_factor
            return min(1.0, confidence)

        return 0.0

    def _evaluate_handwash_motion(self, motion_stats: Dict) -> float:
        """
        评估洗手运动模式

        Args:
            motion_stats: 运动统计数据

        Returns:
            运动模式置信度
        """
        confidence = 0.0

        # 检查运动比例（水平运动应该更多）
        if (
            motion_stats["movement_ratio"]
            >= self.handwash_thresholds["min_movement_ratio"]
        ):
            confidence += 0.3

        # 检查平均速度
        avg_speed = motion_stats["avg_speed"]
        if (
            self.handwash_thresholds["min_avg_speed"]
            <= avg_speed
            <= self.handwash_thresholds["max_avg_speed"]
        ):
            confidence += 0.3

        # 检查位置方差（应该有一定的运动范围）
        if (
            motion_stats["position_variance"]
            >= self.handwash_thresholds["min_position_variance"]
        ):
            confidence += 0.2

        # 检查水平运动强度
        if motion_stats["horizontal_movement"] > motion_stats["vertical_movement"]:
            confidence += 0.2

        # 检查运动频率
        if (
            motion_stats["move_frequency_hz"]
            >= self.handwash_thresholds["min_move_frequency_hz"]
        ):
            confidence += 0.2

        return min(1.0, confidence)

    def _evaluate_sanitize_motion(self, motion_stats: Dict) -> float:
        """
        评估消毒运动模式

        Args:
            motion_stats: 运动统计数据

        Returns:
            运动模式置信度
        """
        confidence = 0.0

        # 检查运动比例（消毒时运动比例较小）
        if (
            motion_stats["movement_ratio"]
            >= self.sanitize_thresholds["min_movement_ratio"]
        ):
            confidence += 0.3

        # 检查平均速度
        avg_speed = motion_stats["avg_speed"]
        if (
            self.sanitize_thresholds["min_avg_speed"]
            <= avg_speed
            <= self.sanitize_thresholds["max_avg_speed"]
        ):
            confidence += 0.4

        # 消毒时运动相对较小但持续
        if motion_stats["position_variance"] > 0:
            confidence += 0.3

        return min(1.0, confidence)

    def reset_track(self, track_id: int):
        """
        重置指定追踪目标的运动数据

        Args:
            track_id: 追踪目标ID
        """
        if track_id in self.hand_trackers:
            for tracker in self.hand_trackers[track_id].values():
                tracker.clear()
            del self.hand_trackers[track_id]

        logger.info(f"Motion data reset for track {track_id}")

    def get_motion_summary(self, track_id: int) -> Dict:
        """
        获取运动摘要

        Args:
            track_id: 追踪目标ID

        Returns:
            运动摘要字典
        """
        if track_id not in self.hand_trackers:
            return {"track_id": track_id, "hands": {}}

        summary = {"track_id": track_id, "hands": {}}

        for hand_label, tracker in self.hand_trackers[track_id].items():
            if len(tracker.position_history) > 0:
                motion_stats = tracker.get_motion_stats()
                summary["hands"][hand_label] = {
                    "motion_stats": motion_stats,
                    "history_length": len(tracker.position_history),
                    "last_update": tracker.last_update_time,
                }

        return summary

    def analyze_motion(self, track_id: int, motion_type: str = "handwashing") -> float:
        """
        分析运动模式（统一接口）
        
        Args:
            track_id: 追踪目标ID
            motion_type: 运动类型 ('handwashing' 或 'sanitizing')
            
        Returns:
            运动模式置信度 (0.0-1.0)
        """
        if motion_type.lower() == "handwashing":
            return self.analyze_handwashing(track_id)
        elif motion_type.lower() == "sanitizing":
            return self.analyze_sanitizing(track_id)
        else:
            logger.warning(f"Unknown motion type: {motion_type}, defaulting to handwashing")
            return self.analyze_handwashing(track_id)

    def cleanup(self):
        """清理资源"""
        for trackers in self.hand_trackers.values():
            for tracker in trackers.values():
                tracker.clear()

        self.hand_trackers.clear()
        logger.info("MotionAnalyzer cleaned up")
