import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

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
        
        if self.use_mediapipe:
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            # 初始化姿态检测
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # 初始化手部检测
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("PoseDetector initialized with MediaPipe")
        else:
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
            # 转换颜色空间
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测姿态
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                # 提取关键点
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                return {
                    'landmarks': landmarks,
                    'pose_landmarks': results.pose_landmarks,
                    'pose_world_landmarks': results.pose_world_landmarks
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
            # 转换颜色空间
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测手部
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 获取手部标签（左手/右手）
                    hand_label = "Unknown"
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[idx].classification[0].label
                    
                    # 提取关键点
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    # 计算手部边界框
                    h, w = image.shape[:2]
                    x_coords = [lm['x'] * w for lm in landmarks]
                    y_coords = [lm['y'] * h for lm in landmarks]
                    
                    bbox = [
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    ]
                    
                    hands_results.append({
                        'label': hand_label,
                        'landmarks': landmarks,
                        'bbox': bbox,
                        'hand_landmarks': hand_landmarks
                    })
        
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
                {'x': center_x / image.shape[1], 'y': (y + h * 0.1) / image.shape[0], 'z': 0, 'visibility': 0.8},  # 头部
                {'x': (center_x - w * 0.2) / image.shape[1], 'y': (y + h * 0.3) / image.shape[0], 'z': 0, 'visibility': 0.7},  # 左肩
                {'x': (center_x + w * 0.2) / image.shape[1], 'y': (y + h * 0.3) / image.shape[0], 'z': 0, 'visibility': 0.7},  # 右肩
                {'x': (center_x - w * 0.3) / image.shape[1], 'y': (y + h * 0.6) / image.shape[0], 'z': 0, 'visibility': 0.6},  # 左手腕
                {'x': (center_x + w * 0.3) / image.shape[1], 'y': (y + h * 0.6) / image.shape[0], 'z': 0, 'visibility': 0.6},  # 右手腕
            ]
            
            return {
                'landmarks': landmarks,
                'pose_landmarks': None,
                'pose_world_landmarks': None
            }
        
        return None
    
    def _fallback_hand_detection(self, image: np.ndarray) -> List[Dict]:
        """
        备用手部检测方法
        
        Args:
            image: 输入图像
            
        Returns:
            简化的手部信息
        """
        # 使用肤色检测作为简单的手部检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 肤色范围（HSV）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 创建肤色掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands_results = []
        
        # 筛选可能的手部区域
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # 手部面积范围
                x, y, w, h = cv2.boundingRect(contour)
                
                # 简化的手部关键点（手心位置）
                center_x = x + w // 2
                center_y = y + h // 2
                
                landmarks = [{
                    'x': center_x / image.shape[1],
                    'y': center_y / image.shape[0],
                    'z': 0
                }]
                
                hands_results.append({
                    'label': 'Unknown',
                    'landmarks': landmarks,
                    'bbox': [x, y, x + w, y + h],
                    'hand_landmarks': None
                })
        
        return hands_results
    
    def get_hand_center(self, hand_landmarks: List[Dict]) -> Tuple[float, float]:
        """
        获取手部中心点
        
        Args:
            hand_landmarks: 手部关键点列表
            
        Returns:
            手部中心点坐标 (x, y)
        """
        if not hand_landmarks:
            return (0.0, 0.0)
        
        x_coords = [lm['x'] for lm in hand_landmarks]
        y_coords = [lm['y'] for lm in hand_landmarks]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return (center_x, center_y)
    
    def cleanup(self):
        """清理资源"""
        if self.use_mediapipe:
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'hands'):
                self.hands.close()
        
        logger.info("PoseDetector cleaned up")