"""姿态检测模块.

提供基于MediaPipe的人体姿态和手部关键点检测功能.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 在导入MediaPipe之前设置环境变量
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

logger = logging.getLogger(__name__)


class PoseDetector:
    """姿态检测器.

    使用MediaPipe或其他方法检测人体关键点，特别是手部关键点.
    """

    def __init__(self, use_mediapipe: bool = True):
        """初始化姿态检测器.

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
        else:
            self.use_mediapipe = False
            logger.warning("MediaPipe not available, using fallback detection")

    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """检测人体姿态.

        Args:
            image: 输入图像

        Returns:
            姿态检测结果，包含关键点信息

        Raises:
            RuntimeError: 当 MediaPipe 不可用时
        """
        if not self.use_mediapipe or self.pose is None:
            raise RuntimeError(
                "MediaPipe 姿态检测器不可用。请检查：\n"
                "1. MediaPipe 是否正确安装\n"
                "2. 系统是否支持 GPU 加速\n"
                "3. 检测器是否正确初始化"
            )

        try:
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
            raise RuntimeError(f"姿态检测失败: {e}")

        return None

    def detect_hands(self, image: np.ndarray) -> List[Dict]:
        """检测手部关键点.

        Args:
            image: 输入图像

        Returns:
            手部检测结果列表，如果MediaPipe不可用则返回空列表
        """
        if not self.use_mediapipe or self.hands is None:
            logger.warning(
                "MediaPipe 手部检测器不可用，返回空结果。原因：\n"
                "1. MediaPipe 未正确安装\n"
                "2. 系统不支持 GPU 加速\n"
                "3. 检测器初始化失败"
            )
            return []

        hands_results = []

        try:
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