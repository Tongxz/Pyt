import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

logger = logging.getLogger(__name__)


class PoseDetector:
    """姿态检测器

    使用MediaPipe或其他方法检测人体关键点，特别是手部关键点
    """

    def __init__(self, use_mediapipe: bool = True):
        """
        初始化姿态检测器

        Args:
            use_mediapipe: 是否使用MediaPipe（如果可用）
        """
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE

        if self.use_mediapipe and mp is not None:
            try:
                # 正确的MediaPipe导入方式 (忽略类型检查器警告)
                mp_pose = mp.solutions.pose  # type: ignore
                mp_hands = mp.solutions.hands  # type: ignore
                mp_drawing = mp.solutions.drawing_utils  # type: ignore

                # 初始化姿态检测 (强制使用CPU模式，避免macOS GPU问题)
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )

                # 初始化手部检测 (强制使用CPU模式)
                self.hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                logger.info("PoseDetector initialized with MediaPipe (CPU mode)")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe: {e}")
                logger.info("Falling back to non-MediaPipe detection")
                self.use_mediapipe = False
                self.pose = None
                self.hands = None

            logger.info("PoseDetector initialized with MediaPipe")
        else:
            self.use_mediapipe = False
            logger.warning("MediaPipe not available, using fallback detection")

    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测人体姿态

        Args:
            image: 输入图像

        Returns:
            姿态检测结果，包含关键点信息
        """
        if not self.use_mediapipe:
            return self._fallback_pose_detection(image)

        try:
            # 检查pose对象是否可用
            if self.pose is None:
                return None

            # 转换颜色空间
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 检测姿态
            results = self.pose.process(rgb_image)

            if results.pose_landmarks:
                # 提取关键点
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append(
                        {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility,
                        }
                    )

                return {
                    "landmarks": landmarks,
                    "pose_landmarks": results.pose_landmarks,
                    "pose_world_landmarks": results.pose_world_landmarks,
                }
        except Exception as e:
            logger.error(f"Pose detection error: {e}")

        return None

    def detect_hands(self, image: np.ndarray) -> List[Dict]:
        """
        检测手部关键点

        Args:
            image: 输入图像

        Returns:
            手部检测结果列表
        """
        if not self.use_mediapipe:
            return self._fallback_hand_detection(image)

        hands_results = []

        try:
            # 检查hands对象是否可用
            if self.hands is None:
                return []

            # 转换颜色空间
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 检测手部
            results = self.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 获取手部标签（左手/右手）
                    hand_label = "Unknown"
                    if results.multi_handedness:
                        hand_label = (
                            results.multi_handedness[idx].classification[0].label
                        )

                    # 提取关键点
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append(
                            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                        )

                    # 计算手部边界框
                    h, w = image.shape[:2]
                    x_coords = [lm["x"] * w for lm in landmarks]
                    y_coords = [lm["y"] * h for lm in landmarks]

                    bbox = [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords)),
                    ]

                    hands_results.append(
                        {
                            "label": hand_label,
                            "landmarks": landmarks,
                            "bbox": bbox,
                            "hand_landmarks": hand_landmarks,
                        }
                    )

        except Exception as e:
            logger.error(f"Hand detection error: {e}")

        return hands_results

    def _fallback_pose_detection(self, image: np.ndarray) -> Optional[Dict]:
        """
        备用姿态检测方法（当MediaPipe不可用时）

        Args:
            image: 输入图像

        Returns:
            简化的姿态信息
        """
        # 简单的基于轮廓的检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用简单的轮廓检测作为备用方案
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到最大轮廓作为人体
            largest_contour = max(contours, key=cv2.contourArea)

            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 估算关键点位置
            center_x = x + w // 2
            center_y = y + h // 2

            # 简化的关键点（头部、肩膀、手腕等）
            landmarks = [
                {
                    "x": center_x / image.shape[1],
                    "y": (y + h * 0.1) / image.shape[0],
                    "z": 0,
                    "visibility": 0.8,
                },  # 头部
                {
                    "x": (center_x - w * 0.2) / image.shape[1],
                    "y": (y + h * 0.3) / image.shape[0],
                    "z": 0,
                    "visibility": 0.7,
                },  # 左肩
                {
                    "x": (center_x + w * 0.2) / image.shape[1],
                    "y": (y + h * 0.3) / image.shape[0],
                    "z": 0,
                    "visibility": 0.7,
                },  # 右肩
                {
                    "x": (center_x - w * 0.3) / image.shape[1],
                    "y": (y + h * 0.6) / image.shape[0],
                    "z": 0,
                    "visibility": 0.6,
                },  # 左手腕
                {
                    "x": (center_x + w * 0.3) / image.shape[1],
                    "y": (y + h * 0.6) / image.shape[0],
                    "z": 0,
                    "visibility": 0.6,
                },  # 右手腕
            ]

            return {
                "landmarks": landmarks,
                "pose_landmarks": None,
                "pose_world_landmarks": None,
            }

        return None

    def _fallback_hand_detection(self, image: np.ndarray) -> List[Dict]:
        """备用手部检测方法.

        当MediaPipe不可用时使用的简单手部检测算法.

        Args:
            image: 输入图像

        Returns:
            List[Dict]: 检测到的手部区域列表
        """
        # 使用更严格的肤色检测作为简单的手部检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 更严格的肤色范围（HSV），减少误检
        lower_skin = np.array([0, 30, 80], dtype=np.uint8)
        upper_skin = np.array([17, 170, 255], dtype=np.uint8)

        # 创建肤色掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 更强的形态学操作，去除噪声
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        hands_results = []

        # 更严格的筛选条件
        for contour in contours:
            area = cv2.contourArea(contour)
            # 更严格的面积范围和形状检查
            if 800 < area < 3000:  # 调整手部面积范围
                x, y, w, h = cv2.boundingRect(contour)

                # 检查长宽比，手部通常不会太细长
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 2.5:  # 合理的长宽比范围
                    # 检查轮廓的凸包面积比，手部轮廓相对规整
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity > 0.6:  # 轮廓的实心度
                            # 简化的手部关键点（手心位置）
                            center_x = x + w // 2
                            center_y = y + h // 2

                            landmarks = [
                                {
                                    "x": center_x / image.shape[1],
                                    "y": center_y / image.shape[0],
                                    "z": 0,
                                }
                            ]

                            hands_results.append(
                                {
                                    "label": "Unknown",
                                    "landmarks": landmarks,
                                    "bbox": [x, y, x + w, y + h],
                                    "hand_landmarks": None,
                                }
                            )

        # 限制最大检测数量，避免过多误检
        return hands_results[:4]  # 最多返回4个手部检测结果

    def get_hand_center(self, hand_landmarks: List[Dict]) -> Tuple[float, float]:
        """获取手部中心点.

        Args:
            hand_landmarks: 手部关键点列表

        Returns:
            手部中心点坐标 (x, y)
        """
        if not hand_landmarks:
            return (0.0, 0.0)

        x_coords = [lm["x"] for lm in hand_landmarks]
        y_coords = [lm["y"] for lm in hand_landmarks]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        return (center_x, center_y)

    def cleanup(self):
        """清理资源."""
        if self.use_mediapipe:
            if hasattr(self, "pose") and self.pose is not None:
                self.pose.close()
            if hasattr(self, "hands") and self.hands is not None:
                self.hands.close()

        logger.info("PoseDetector cleaned up")
