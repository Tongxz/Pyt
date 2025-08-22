import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdvancedMotionTracker:
    """高级运动轨迹跟踪器
    
    增强功能：
    - 高级特征提取（频域分析、轨迹形状特征）
    - 运动模式识别
    - 异常检测
    """

    def __init__(self, max_history: int = 60, sampling_rate: float = 30.0):
        """
        初始化高级运动跟踪器

        Args:
            max_history: 最大历史记录数量
            sampling_rate: 采样率（帧/秒）
        """
        self.max_history = max_history
        self.sampling_rate = sampling_rate
        self.position_history = deque(maxlen=max_history)
        self.velocity_history = deque(maxlen=max_history)
        self.acceleration_history = deque(maxlen=max_history)
        self.last_update_time = None
        
        # 特征缓存
        self._feature_cache = {}
        self._cache_timestamp = 0
        
        # 异常检测参数
        self.anomaly_threshold = 2.0  # Z-score阈值
        
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

        # 计算速度和加速度
        if len(self.position_history) >= 2:
            self._calculate_derivatives()
            
        # 清除特征缓存
        self._feature_cache.clear()
        self._cache_timestamp = timestamp
        
        self.last_update_time = timestamp

    def _calculate_derivatives(self):
        """计算速度和加速度"""
        if len(self.position_history) < 2:
            return
            
        # 计算速度
        prev_pos, prev_time = self.position_history[-2]
        curr_pos, curr_time = self.position_history[-1]
        
        dt = curr_time - prev_time
        if dt > 0:
            velocity = (
                (curr_pos[0] - prev_pos[0]) / dt,
                (curr_pos[1] - prev_pos[1]) / dt,
            )
            self.velocity_history.append((velocity, curr_time))
            
            # 计算加速度
            if len(self.velocity_history) >= 2:
                prev_vel, prev_vel_time = self.velocity_history[-2]
                curr_vel, curr_vel_time = self.velocity_history[-1]
                
                dt_vel = curr_vel_time - prev_vel_time
                if dt_vel > 0:
                    acceleration = (
                        (curr_vel[0] - prev_vel[0]) / dt_vel,
                        (curr_vel[1] - prev_vel[1]) / dt_vel,
                    )
                    self.acceleration_history.append((acceleration, curr_vel_time))

    def get_advanced_motion_features(self) -> Dict[str, Any]:
        """获取高级运动特征
        
        Returns:
            包含高级特征的字典
        """
        if len(self.position_history) < 10:
            return self._get_empty_features()
            
        # 检查缓存
        if self._feature_cache:
            return self._feature_cache
            
        features = {}
        
        # 基础统计特征
        features.update(self._get_basic_stats())
        
        # 轨迹形状特征
        features.update(self._get_trajectory_shape_features())
        
        # 频域特征
        features.update(self._get_frequency_features())
        
        # 运动模式特征
        features.update(self._get_motion_pattern_features())
        
        # 异常检测特征
        features.update(self._get_anomaly_features())
        
        # 缓存结果
        self._feature_cache = features
        
        return features

    def _get_empty_features(self) -> Dict[str, Any]:
        """返回空特征字典"""
        return {
            # 基础统计
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "speed_variance": 0.0,
            "avg_acceleration": 0.0,
            "position_variance_x": 0.0,
            "position_variance_y": 0.0,
            
            # 轨迹形状
            "trajectory_length": 0.0,
            "displacement": 0.0,
            "tortuosity": 0.0,
            "convex_hull_area": 0.0,
            "direction_changes": 0,
            
            # 频域特征
            "dominant_frequency_x": 0.0,
            "dominant_frequency_y": 0.0,
            "spectral_centroid_x": 0.0,
            "spectral_centroid_y": 0.0,
            "spectral_rolloff_x": 0.0,
            "spectral_rolloff_y": 0.0,
            
            # 运动模式
            "periodicity_score": 0.0,
            "smoothness_index": 0.0,
            "directional_consistency": 0.0,
            "pause_ratio": 0.0,
            
            # 异常检测
            "anomaly_score": 0.0,
            "outlier_ratio": 0.0
        }

    def _get_basic_stats(self) -> Dict[str, float]:
        """获取基础统计特征"""
        positions = [pos for pos, _ in self.position_history]
        velocities = [vel for vel, _ in self.velocity_history]
        accelerations = [acc for acc, _ in self.acceleration_history]
        
        # 速度统计
        speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities]
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0
        speed_variance = np.var(speeds) if speeds else 0.0
        
        # 加速度统计
        acc_magnitudes = [np.sqrt(a[0]**2 + a[1]**2) for a in accelerations]
        avg_acceleration = np.mean(acc_magnitudes) if acc_magnitudes else 0.0
        
        # 位置方差
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        position_variance_x = np.var(x_coords) if x_coords else 0.0
        position_variance_y = np.var(y_coords) if y_coords else 0.0
        
        return {
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "speed_variance": speed_variance,
            "avg_acceleration": avg_acceleration,
            "position_variance_x": position_variance_x,
            "position_variance_y": position_variance_y,
        }

    def _get_trajectory_shape_features(self) -> Dict[str, float]:
        """获取轨迹形状特征"""
        positions = np.array([pos for pos, _ in self.position_history])
        
        if len(positions) < 3:
            return {
                "trajectory_length": 0.0,
                "displacement": 0.0,
                "tortuosity": 0.0,
                "convex_hull_area": 0.0,
                "direction_changes": 0,
            }
        
        # 轨迹长度
        diffs = np.diff(positions, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        trajectory_length = np.sum(distances)
        
        # 位移（起点到终点的直线距离）
        displacement = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        
        # 弯曲度（轨迹长度/位移）
        tortuosity = trajectory_length / (displacement + 1e-6)
        
        # 凸包面积
        try:
            from scipy.spatial import ConvexHull
            if len(positions) >= 3:
                hull = ConvexHull(positions)
                convex_hull_area = hull.volume  # 在2D中volume就是面积
            else:
                convex_hull_area = 0.0
        except:
            convex_hull_area = 0.0
        
        # 方向变化次数
        direction_changes = self._count_direction_changes(positions)
        
        return {
            "trajectory_length": trajectory_length,
            "displacement": displacement,
            "tortuosity": tortuosity,
            "convex_hull_area": convex_hull_area,
            "direction_changes": direction_changes,
        }

    def _count_direction_changes(self, positions: np.ndarray, threshold: float = 0.1) -> int:
        """计算方向变化次数"""
        if len(positions) < 3:
            return 0
            
        # 计算相邻向量的角度
        vectors = np.diff(positions, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # 计算角度变化
        angle_diffs = np.diff(angles)
        
        # 处理角度跳跃（-π到π）
        angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
        angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
        
        # 计算显著方向变化
        significant_changes = np.abs(angle_diffs) > threshold
        
        return np.sum(significant_changes)

    def _get_frequency_features(self) -> Dict[str, float]:
        """获取频域特征"""
        positions = np.array([pos for pos, _ in self.position_history])
        
        if len(positions) < 16:  # 需要足够的数据进行FFT
            return {
                "dominant_frequency_x": 0.0,
                "dominant_frequency_y": 0.0,
                "spectral_centroid_x": 0.0,
                "spectral_centroid_y": 0.0,
                "spectral_rolloff_x": 0.0,
                "spectral_rolloff_y": 0.0,
            }
        
        # 对x和y坐标分别进行频域分析
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # 去除直流分量
        x_coords = x_coords - np.mean(x_coords)
        y_coords = y_coords - np.mean(y_coords)
        
        # FFT分析
        fft_x = np.fft.fft(x_coords)
        fft_y = np.fft.fft(y_coords)
        
        freqs = np.fft.fftfreq(len(positions), 1.0 / self.sampling_rate)
        
        # 只考虑正频率
        positive_freqs = freqs[:len(freqs)//2]
        magnitude_x = np.abs(fft_x[:len(fft_x)//2])
        magnitude_y = np.abs(fft_y[:len(fft_y)//2])
        
        # 主导频率
        dominant_freq_x = positive_freqs[np.argmax(magnitude_x)] if len(magnitude_x) > 0 else 0.0
        dominant_freq_y = positive_freqs[np.argmax(magnitude_y)] if len(magnitude_y) > 0 else 0.0
        
        # 频谱质心
        spectral_centroid_x = self._calculate_spectral_centroid(positive_freqs, magnitude_x)
        spectral_centroid_y = self._calculate_spectral_centroid(positive_freqs, magnitude_y)
        
        # 频谱滚降点
        spectral_rolloff_x = self._calculate_spectral_rolloff(positive_freqs, magnitude_x)
        spectral_rolloff_y = self._calculate_spectral_rolloff(positive_freqs, magnitude_y)
        
        return {
            "dominant_frequency_x": dominant_freq_x,
            "dominant_frequency_y": dominant_freq_y,
            "spectral_centroid_x": spectral_centroid_x,
            "spectral_centroid_y": spectral_centroid_y,
            "spectral_rolloff_x": spectral_rolloff_x,
            "spectral_rolloff_y": spectral_rolloff_y,
        }

    def _calculate_spectral_centroid(self, freqs: np.ndarray, magnitudes: np.ndarray) -> float:
        """计算频谱质心"""
        if len(magnitudes) == 0 or np.sum(magnitudes) == 0:
            return 0.0
        return np.sum(freqs * magnitudes) / np.sum(magnitudes)

    def _calculate_spectral_rolloff(self, freqs: np.ndarray, magnitudes: np.ndarray, rolloff_ratio: float = 0.85) -> float:
        """计算频谱滚降点"""
        if len(magnitudes) == 0:
            return 0.0
            
        cumsum = np.cumsum(magnitudes)
        total_energy = cumsum[-1]
        
        if total_energy == 0:
            return 0.0
            
        rolloff_threshold = rolloff_ratio * total_energy
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1] if len(freqs) > 0 else 0.0

    def _get_motion_pattern_features(self) -> Dict[str, float]:
        """获取运动模式特征"""
        if len(self.velocity_history) < 10:
            return {
                "periodicity_score": 0.0,
                "smoothness_index": 0.0,
                "directional_consistency": 0.0,
                "pause_ratio": 0.0,
            }
        
        velocities = np.array([vel for vel, _ in self.velocity_history])
        speeds = np.sqrt(np.sum(velocities**2, axis=1))
        
        # 周期性评分（基于自相关）
        periodicity_score = self._calculate_periodicity(speeds)
        
        # 平滑度指数（基于加速度变化）
        smoothness_index = self._calculate_smoothness()
        
        # 方向一致性
        directional_consistency = self._calculate_directional_consistency(velocities)
        
        # 暂停比例（低速度时间占比）
        pause_ratio = self._calculate_pause_ratio(speeds)
        
        return {
            "periodicity_score": periodicity_score,
            "smoothness_index": smoothness_index,
            "directional_consistency": directional_consistency,
            "pause_ratio": pause_ratio,
        }

    def _calculate_periodicity(self, signal_data: np.ndarray) -> float:
        """计算信号的周期性"""
        if len(signal_data) < 20:
            return 0.0
            
        # 计算自相关
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 归一化
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        # 寻找第一个显著峰值（排除零延迟）
        if len(autocorr) > 5:
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            if len(peaks) > 0:
                return autocorr[peaks[0] + 1]  # +1因为我们从索引1开始
        
        return 0.0

    def _calculate_smoothness(self) -> float:
        """计算运动平滑度"""
        if len(self.acceleration_history) < 5:
            return 0.0
            
        accelerations = np.array([acc for acc, _ in self.acceleration_history])
        acc_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))
        
        # 计算加速度的变化率（jerk）
        jerk = np.diff(acc_magnitudes)
        
        # 平滑度与jerk的方差成反比
        jerk_variance = np.var(jerk) if len(jerk) > 0 else 0.0
        
        # 归一化到0-1范围
        smoothness = 1.0 / (1.0 + jerk_variance)
        
        return smoothness

    def _calculate_directional_consistency(self, velocities: np.ndarray) -> float:
        """计算方向一致性"""
        if len(velocities) < 3:
            return 0.0
            
        # 计算速度向量的角度
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        # 计算角度变化
        angle_diffs = np.diff(angles)
        
        # 处理角度跳跃
        angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
        angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
        
        # 计算角度变化的标准差
        angle_std = np.std(angle_diffs) if len(angle_diffs) > 0 else 0.0
        
        # 一致性与角度变化的标准差成反比
        consistency = 1.0 / (1.0 + angle_std)
        
        return consistency

    def _calculate_pause_ratio(self, speeds: np.ndarray, pause_threshold: float = 0.01) -> float:
        """计算暂停比例"""
        if len(speeds) == 0:
            return 0.0
            
        pause_count = np.sum(speeds < pause_threshold)
        return pause_count / len(speeds)

    def _get_anomaly_features(self) -> Dict[str, float]:
        """获取异常检测特征"""
        if len(self.position_history) < 10:
            return {
                "anomaly_score": 0.0,
                "outlier_ratio": 0.0
            }
        
        positions = np.array([pos for pos, _ in self.position_history])
        
        # 计算每个点到轨迹中心的距离
        center = np.mean(positions, axis=0)
        distances = np.sqrt(np.sum((positions - center)**2, axis=1))
        
        # 使用Z-score检测异常
        z_scores = np.abs((distances - np.mean(distances)) / (np.std(distances) + 1e-6))
        
        # 异常评分（最大Z-score）
        anomaly_score = np.max(z_scores) if len(z_scores) > 0 else 0.0
        
        # 异常点比例
        outliers = z_scores > self.anomaly_threshold
        outlier_ratio = np.sum(outliers) / len(z_scores) if len(z_scores) > 0 else 0.0
        
        return {
            "anomaly_score": anomaly_score,
            "outlier_ratio": outlier_ratio
        }

    def clear(self):
        """清除历史数据"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self._feature_cache.clear()
        self.last_update_time = None


class AdaptiveThresholdManager:
    """自适应阈值管理器
    
    根据历史数据和用户行为模式动态调整检测阈值
    """

    def __init__(self):
        """
        初始化自适应阈值管理器
        """
        # 基础阈值（初始值）
        self.base_thresholds = {
            "handwash": {
                "min_movement_ratio": 0.8,
                "min_avg_speed": 0.005,
                "max_avg_speed": 0.8,
                "min_position_variance": 0.0005,
                "min_duration": 2.0,
                "min_trajectory_length": 0.1,
                "min_direction_changes": 3,
            },
            "sanitize": {
                "max_hand_distance": 0.15,
                "min_movement_ratio": 0.5,
                "min_avg_speed": 0.005,
                "max_avg_speed": 0.3,
                "min_duration": 2.0,
                "min_periodicity": 0.3,
            }
        }
        
        # 当前自适应阈值
        self.adaptive_thresholds = self.base_thresholds.copy()
        
        # 历史数据用于自适应调整
        self.feature_history = deque(maxlen=100)  # 保存最近100次检测的特征
        self.detection_history = deque(maxlen=100)  # 保存检测结果
        
        # 自适应参数
        self.adaptation_rate = 0.1  # 适应速率
        self.min_samples_for_adaptation = 20  # 开始自适应的最小样本数
        
    def update_thresholds(self, features: Dict[str, Any], detection_result: Dict[str, float]):
        """
        根据新的特征和检测结果更新阈值
        
        Args:
            features: 提取的特征字典
            detection_result: 检测结果 {'handwash': confidence, 'sanitize': confidence}
        """
        # 保存历史数据
        self.feature_history.append(features)
        self.detection_history.append(detection_result)
        
        # 如果样本数量足够，进行自适应调整
        if len(self.feature_history) >= self.min_samples_for_adaptation:
            self._adapt_handwash_thresholds()
            self._adapt_sanitize_thresholds()
    
    def _adapt_handwash_thresholds(self):
        """自适应调整洗手检测阈值"""
        # 收集正样本（高置信度检测结果）的特征
        positive_features = []
        for i, detection in enumerate(self.detection_history):
            if detection.get('handwash', 0) > 0.7:  # 高置信度阈值
                positive_features.append(self.feature_history[i])
        
        if len(positive_features) < 5:  # 需要足够的正样本
            return
            
        # 计算正样本特征的统计信息
        feature_stats = self._calculate_feature_statistics(positive_features)
        
        # 根据统计信息调整阈值
        current = self.adaptive_thresholds["handwash"]
        
        # 运动比例阈值：使用正样本的25%分位数
        if "movement_ratio" in feature_stats:
            target_ratio = feature_stats["movement_ratio"]["percentile_25"]
            current["min_movement_ratio"] = self._smooth_update(
                current["min_movement_ratio"], target_ratio, self.adaptation_rate
            )
        
        # 平均速度阈值
        if "avg_speed" in feature_stats:
            target_min_speed = feature_stats["avg_speed"]["percentile_10"]
            target_max_speed = feature_stats["avg_speed"]["percentile_90"]
            current["min_avg_speed"] = self._smooth_update(
                current["min_avg_speed"], target_min_speed, self.adaptation_rate
            )
            current["max_avg_speed"] = self._smooth_update(
                current["max_avg_speed"], target_max_speed, self.adaptation_rate
            )
        
        # 轨迹长度阈值
        if "trajectory_length" in feature_stats:
            target_length = feature_stats["trajectory_length"]["percentile_25"]
            current["min_trajectory_length"] = self._smooth_update(
                current["min_trajectory_length"], target_length, self.adaptation_rate
            )
    
    def _adapt_sanitize_thresholds(self):
        """自适应调整消毒检测阈值"""
        # 收集正样本特征
        positive_features = []
        for i, detection in enumerate(self.detection_history):
            if detection.get('sanitize', 0) > 0.7:
                positive_features.append(self.feature_history[i])
        
        if len(positive_features) < 5:
            return
            
        feature_stats = self._calculate_feature_statistics(positive_features)
        current = self.adaptive_thresholds["sanitize"]
        
        # 周期性阈值
        if "periodicity_score" in feature_stats:
            target_periodicity = feature_stats["periodicity_score"]["percentile_25"]
            current["min_periodicity"] = self._smooth_update(
                current["min_periodicity"], target_periodicity, self.adaptation_rate
            )
    
    def _calculate_feature_statistics(self, features_list: List[Dict]) -> Dict[str, Dict[str, float]]:
        """计算特征统计信息"""
        stats = {}
        
        # 收集所有特征值
        feature_values = {}
        for features in features_list:
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if key not in feature_values:
                        feature_values[key] = []
                    feature_values[key].append(value)
        
        # 计算统计信息
        for key, values in feature_values.items():
            if len(values) > 0:
                values_array = np.array(values)
                stats[key] = {
                    "mean": np.mean(values_array),
                    "std": np.std(values_array),
                    "percentile_10": np.percentile(values_array, 10),
                    "percentile_25": np.percentile(values_array, 25),
                    "percentile_75": np.percentile(values_array, 75),
                    "percentile_90": np.percentile(values_array, 90),
                }
        
        return stats
    
    def _smooth_update(self, current_value: float, target_value: float, rate: float) -> float:
        """平滑更新阈值"""
        return current_value + rate * (target_value - current_value)
    
    def get_thresholds(self, behavior_type: str) -> Dict[str, float]:
        """获取指定行为类型的当前阈值
        
        Args:
            behavior_type: 行为类型 ('handwash' 或 'sanitize')
            
        Returns:
            阈值字典
        """
        return self.adaptive_thresholds.get(behavior_type, {})
    
    def reset_to_base(self):
        """重置为基础阈值"""
        self.adaptive_thresholds = self.base_thresholds.copy()
        self.feature_history.clear()
        self.detection_history.clear()


class EnhancedMotionAnalyzer:
    """增强运动分析器
    
    集成高级特征提取和自适应阈值调整
    """

    def __init__(self):
        """
        初始化增强运动分析器
        """
        self.hand_trackers = {}  # track_id -> {'left': AdvancedMotionTracker, 'right': AdvancedMotionTracker}
        self.threshold_manager = AdaptiveThresholdManager()
        
        logger.info("EnhancedMotionAnalyzer initialized")

    def update_hand_motion(self, track_id: int, hands_data: List[Dict]):
        """
        更新手部运动数据

        Args:
            track_id: 追踪目标ID
            hands_data: 手部检测数据列表
        """
        if track_id not in self.hand_trackers:
            self.hand_trackers[track_id] = {
                "left": AdvancedMotionTracker(),
                "right": AdvancedMotionTracker(),
                "unknown": AdvancedMotionTracker(),
            }

        current_time = time.time()

        for hand_data in hands_data:
            try:
                hand_label = hand_data.get("label", "Unknown").lower()
                if hand_label not in ["left", "right"]:
                    hand_label = "unknown"

                # 获取手部中心点（与原版本相同的逻辑）
                center_x, center_y = self._extract_hand_center(hand_data)
                
                if center_x is not None and center_y is not None:
                    self.hand_trackers[track_id][hand_label].update(
                        (float(center_x), float(center_y)), current_time
                    )

            except Exception as e:
                logger.error(f"Error processing hand data: {e}")
                continue

    def _extract_hand_center(self, hand_data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """提取手部中心点（复用原有逻辑）"""
        center_x, center_y = None, None

        if "landmarks" in hand_data and hand_data["landmarks"]:
            landmarks = hand_data["landmarks"]
            if isinstance(landmarks, list) and len(landmarks) > 0:
                if isinstance(landmarks[0], dict):
                    center_x = float(np.mean([lm["x"] for lm in landmarks if "x" in lm]))
                    center_y = float(np.mean([lm["y"] for lm in landmarks if "y" in lm]))

        elif "bbox" in hand_data and hand_data["bbox"]:
            bbox = hand_data["bbox"]
            if isinstance(bbox, list) and len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                # 归一化处理
                if center_x > 1.0 or center_y > 1.0:
                    center_x /= 640
                    center_y /= 480

        return center_x, center_y

    def analyze_handwashing_enhanced(self, track_id: int) -> float:
        """
        增强洗手行为分析

        Args:
            track_id: 追踪目标ID

        Returns:
            洗手行为置信度 (0.0-1.0)
        """
        if track_id not in self.hand_trackers:
            return 0.0

        trackers = self.hand_trackers[track_id]
        thresholds = self.threshold_manager.get_thresholds("handwash")
        
        hand_confidences = []
        combined_features = {}

        for hand_label, tracker in trackers.items():
            if len(tracker.position_history) < 10:
                continue

            # 获取高级特征
            features = tracker.get_advanced_motion_features()
            
            # 合并特征用于阈值更新
            for key, value in features.items():
                if key not in combined_features:
                    combined_features[key] = []
                combined_features[key].append(value)
            
            # 使用高级特征进行洗手检测
            hand_conf = self._evaluate_handwash_enhanced(features, thresholds)
            
            if hand_conf > 0:
                hand_confidences.append(hand_conf)
                logger.debug(f"Hand {hand_label} enhanced handwash confidence: {hand_conf:.3f}")

        # 计算最终置信度
        if len(hand_confidences) >= 2:
            confidence = min(1.0, np.mean(hand_confidences) * 1.2)
        elif len(hand_confidences) == 1:
            confidence = hand_confidences[0] * 0.8
        else:
            confidence = 0.0
        
        # 更新自适应阈值
        if combined_features:
            # 计算平均特征值
            avg_features = {}
            for key, values in combined_features.items():
                avg_features[key] = np.mean(values)
            
            detection_result = {"handwash": confidence, "sanitize": 0.0}
            self.threshold_manager.update_thresholds(avg_features, detection_result)

        return confidence

    def _evaluate_handwash_enhanced(self, features: Dict[str, Any], thresholds: Dict[str, float]) -> float:
        """使用高级特征评估洗手行为"""
        confidence = 0.0
        
        # 基础运动特征（权重：0.3）
        if features["avg_speed"] >= thresholds.get("min_avg_speed", 0.005):
            if features["avg_speed"] <= thresholds.get("max_avg_speed", 0.8):
                confidence += 0.15
        
        if features["position_variance_x"] + features["position_variance_y"] >= thresholds.get("min_position_variance", 0.0005):
            confidence += 0.15
        
        # 轨迹形状特征（权重：0.3）
        if features["trajectory_length"] >= thresholds.get("min_trajectory_length", 0.1):
            confidence += 0.1
            
        if features["direction_changes"] >= thresholds.get("min_direction_changes", 3):
            confidence += 0.1
            
        if features["tortuosity"] > 1.5:  # 弯曲的轨迹
            confidence += 0.1
        
        # 运动模式特征（权重：0.25）
        if features["periodicity_score"] > 0.3:  # 有一定周期性
            confidence += 0.1
            
        if features["smoothness_index"] > 0.5:  # 相对平滑的运动
            confidence += 0.1
            
        if features["pause_ratio"] < 0.3:  # 较少的暂停
            confidence += 0.05
        
        # 频域特征（权重：0.15）
        if 0.5 <= features["dominant_frequency_x"] <= 3.0:  # 合理的主导频率
            confidence += 0.075
            
        if 0.5 <= features["dominant_frequency_y"] <= 3.0:
            confidence += 0.075
        
        return min(1.0, confidence)

    def analyze_sanitizing_enhanced(self, track_id: int) -> float:
        """
        增强消毒行为分析

        Args:
            track_id: 追踪目标ID

        Returns:
            消毒行为置信度 (0.0-1.0)
        """
        if track_id not in self.hand_trackers:
            return 0.0

        trackers = self.hand_trackers[track_id]
        thresholds = self.threshold_manager.get_thresholds("sanitize")
        
        # 需要检测到双手
        active_hands = []
        hand_features = []
        
        for hand_label, tracker in trackers.items():
            if len(tracker.position_history) >= 10:
                features = tracker.get_advanced_motion_features()
                active_hands.append((hand_label, tracker))
                hand_features.append(features)

        if len(active_hands) < 2:
            return 0.0

        # 计算双手距离
        hand1_pos = active_hands[0][1].position_history[-1][0]
        hand2_pos = active_hands[1][1].position_history[-1][0]
        distance = np.sqrt((hand1_pos[0] - hand2_pos[0])**2 + (hand1_pos[1] - hand2_pos[1])**2)

        if distance > thresholds.get("max_hand_distance", 0.15):
            return 0.0

        # 使用高级特征评估消毒行为
        motion_confidences = []
        combined_features = {}
        
        for features in hand_features:
            # 合并特征
            for key, value in features.items():
                if key not in combined_features:
                    combined_features[key] = []
                combined_features[key].append(value)
            
            hand_conf = self._evaluate_sanitize_enhanced(features, thresholds)
            motion_confidences.append(hand_conf)

        if motion_confidences:
            base_confidence = np.mean(motion_confidences)
            distance_factor = 1.0 - (distance / thresholds.get("max_hand_distance", 0.15))
            confidence = base_confidence * distance_factor
            
            # 更新自适应阈值
            if combined_features:
                avg_features = {key: np.mean(values) for key, values in combined_features.items()}
                detection_result = {"handwash": 0.0, "sanitize": confidence}
                self.threshold_manager.update_thresholds(avg_features, detection_result)
            
            return min(1.0, confidence)

        return 0.0

    def _evaluate_sanitize_enhanced(self, features: Dict[str, Any], thresholds: Dict[str, float]) -> float:
        """使用高级特征评估消毒行为"""
        confidence = 0.0
        
        # 基础运动特征（权重：0.4）
        if thresholds.get("min_avg_speed", 0.005) <= features["avg_speed"] <= thresholds.get("max_avg_speed", 0.3):
            confidence += 0.2
            
        if features["position_variance_x"] + features["position_variance_y"] > 0:
            confidence += 0.2
        
        # 运动模式特征（权重：0.4）
        if features["periodicity_score"] >= thresholds.get("min_periodicity", 0.3):
            confidence += 0.2  # 消毒动作通常有周期性
            
        if features["smoothness_index"] > 0.6:  # 相对平滑
            confidence += 0.1
            
        if features["directional_consistency"] > 0.7:  # 方向相对一致
            confidence += 0.1
        
        # 轨迹特征（权重：0.2）
        if 1.0 < features["tortuosity"] < 2.0:  # 适中的弯曲度
            confidence += 0.1
            
        if features["convex_hull_area"] < 0.01:  # 相对紧凑的运动区域
            confidence += 0.1
        
        return min(1.0, confidence)

    def reset_track(self, track_id: int):
        """重置指定追踪目标的运动数据"""
        if track_id in self.hand_trackers:
            for tracker in self.hand_trackers[track_id].values():
                tracker.clear()
            del self.hand_trackers[track_id]
        logger.info(f"Enhanced motion data reset for track {track_id}")

    def get_enhanced_motion_summary(self, track_id: int) -> Dict:
        """获取增强运动摘要"""
        if track_id not in self.hand_trackers:
            return {"track_id": track_id, "hands": {}, "adaptive_thresholds": {}}

        summary = {"track_id": track_id, "hands": {}, "adaptive_thresholds": {}}

        for hand_label, tracker in self.hand_trackers[track_id].items():
            if len(tracker.position_history) > 0:
                features = tracker.get_advanced_motion_features()
                summary["hands"][hand_label] = {
                    "advanced_features": features,
                    "history_length": len(tracker.position_history),
                    "last_update": tracker.last_update_time,
                }
        
        # 添加当前自适应阈值
        summary["adaptive_thresholds"] = {
            "handwash": self.threshold_manager.get_thresholds("handwash"),
            "sanitize": self.threshold_manager.get_thresholds("sanitize"),
        }

        return summary

    def cleanup(self):
        """清理资源"""
        for trackers in self.hand_trackers.values():
            for tracker in trackers.values():
                tracker.clear()
        self.hand_trackers.clear()
        self.threshold_manager.reset_to_base()
        logger.info("EnhancedMotionAnalyzer cleaned up")