import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class UserProfile:
    """用户画像类
    
    存储和管理用户的行为特征和偏好
    """
    
    def __init__(self, user_id: str):
        """
        初始化用户画像
        
        Args:
            user_id: 用户唯一标识
        """
        self.user_id = user_id
        self.created_at = time.time()
        self.last_updated = time.time()
        
        # 行为统计
        self.behavior_counts = defaultdict(int)  # 各种行为的计数
        self.session_count = 0
        self.total_detection_time = 0.0
        
        # 行为特征
        self.avg_handwash_duration = 0.0
        self.avg_sanitize_duration = 0.0
        self.preferred_hand = 'unknown'  # 'left', 'right', 'both', 'unknown'
        self.motion_intensity = 'medium'  # 'low', 'medium', 'high'
        
        # 检测偏好
        self.sensitivity_preference = 0.5  # 0.0-1.0, 越高越敏感
        self.false_positive_tolerance = 0.3  # 0.0-1.0, 对误报的容忍度
        
        # 环境因素
        self.common_environments = []  # 常见的检测环境
        self.lighting_conditions = []  # 光照条件偏好
        
        # 学习历史
        self.feedback_history = []  # 用户反馈历史
        self.adaptation_history = []  # 适配历史
        
        # 特征向量缓存
        self.feature_vectors = deque(maxlen=1000)  # 最近的特征向量
        self.behavior_patterns = {}  # 行为模式
    
    def update_behavior_stats(self, behavior_type: str, duration: float, confidence: float):
        """更新行为统计
        
        Args:
            behavior_type: 行为类型
            duration: 行为持续时间
            confidence: 检测置信度
        """
        self.behavior_counts[behavior_type] += 1
        self.last_updated = time.time()
        
        # 更新平均持续时间
        if behavior_type == 'handwash':
            count = self.behavior_counts['handwash']
            self.avg_handwash_duration = (
                (self.avg_handwash_duration * (count - 1) + duration) / count
            )
        elif behavior_type == 'sanitize':
            count = self.behavior_counts['sanitize']
            self.avg_sanitize_duration = (
                (self.avg_sanitize_duration * (count - 1) + duration) / count
            )
    
    def add_feedback(self, feedback_type: str, context: Dict[str, Any]):
        """添加用户反馈
        
        Args:
            feedback_type: 反馈类型 ('correct', 'false_positive', 'false_negative')
            context: 反馈上下文信息
        """
        feedback = {
            'type': feedback_type,
            'timestamp': time.time(),
            'context': context
        }
        self.feedback_history.append(feedback)
        
        # 保持最近1000条反馈
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """获取用户画像摘要"""
        return {
            'user_id': self.user_id,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'session_count': self.session_count,
            'behavior_counts': dict(self.behavior_counts),
            'avg_handwash_duration': self.avg_handwash_duration,
            'avg_sanitize_duration': self.avg_sanitize_duration,
            'preferred_hand': self.preferred_hand,
            'motion_intensity': self.motion_intensity,
            'sensitivity_preference': self.sensitivity_preference,
            'feedback_count': len(self.feedback_history),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'user_id': self.user_id,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'behavior_counts': dict(self.behavior_counts),
            'session_count': self.session_count,
            'total_detection_time': self.total_detection_time,
            'avg_handwash_duration': self.avg_handwash_duration,
            'avg_sanitize_duration': self.avg_sanitize_duration,
            'preferred_hand': self.preferred_hand,
            'motion_intensity': self.motion_intensity,
            'sensitivity_preference': self.sensitivity_preference,
            'false_positive_tolerance': self.false_positive_tolerance,
            'common_environments': self.common_environments,
            'lighting_conditions': self.lighting_conditions,
            'feedback_history': self.feedback_history[-100:],  # 只保存最近100条
            'behavior_patterns': self.behavior_patterns,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """从字典创建用户画像"""
        profile = cls(data['user_id'])
        profile.created_at = data.get('created_at', time.time())
        profile.last_updated = data.get('last_updated', time.time())
        profile.behavior_counts = defaultdict(int, data.get('behavior_counts', {}))
        profile.session_count = data.get('session_count', 0)
        profile.total_detection_time = data.get('total_detection_time', 0.0)
        profile.avg_handwash_duration = data.get('avg_handwash_duration', 0.0)
        profile.avg_sanitize_duration = data.get('avg_sanitize_duration', 0.0)
        profile.preferred_hand = data.get('preferred_hand', 'unknown')
        profile.motion_intensity = data.get('motion_intensity', 'medium')
        profile.sensitivity_preference = data.get('sensitivity_preference', 0.5)
        profile.false_positive_tolerance = data.get('false_positive_tolerance', 0.3)
        profile.common_environments = data.get('common_environments', [])
        profile.lighting_conditions = data.get('lighting_conditions', [])
        profile.feedback_history = data.get('feedback_history', [])
        profile.behavior_patterns = data.get('behavior_patterns', {})
        return profile


class AdaptiveThresholdOptimizer:
    """自适应阈值优化器
    
    根据用户反馈和行为模式优化检测阈值
    """
    
    def __init__(self):
        self.base_thresholds = {
            'handwash': 0.7,
            'sanitize': 0.6,
            'motion_intensity': 0.5,
            'duration_min': 3.0,
            'duration_max': 60.0,
        }
        
        # 学习率
        self.learning_rate = 0.1
        
        # 优化历史
        self.optimization_history = []
    
    def optimize_thresholds(self, 
                          user_profile: UserProfile, 
                          recent_performance: Dict[str, float]) -> Dict[str, float]:
        """优化检测阈值
        
        Args:
            user_profile: 用户画像
            recent_performance: 最近的性能指标
            
        Returns:
            优化后的阈值
        """
        optimized_thresholds = self.base_thresholds.copy()
        
        # 根据用户敏感度偏好调整
        sensitivity_factor = user_profile.sensitivity_preference
        
        # 调整行为检测阈值
        optimized_thresholds['handwash'] = self._adjust_threshold(
            self.base_thresholds['handwash'],
            sensitivity_factor,
            recent_performance.get('handwash_precision', 0.8),
            recent_performance.get('handwash_recall', 0.8)
        )
        
        optimized_thresholds['sanitize'] = self._adjust_threshold(
            self.base_thresholds['sanitize'],
            sensitivity_factor,
            recent_performance.get('sanitize_precision', 0.8),
            recent_performance.get('sanitize_recall', 0.8)
        )
        
        # 根据用户行为模式调整持续时间阈值
        if user_profile.avg_handwash_duration > 0:
            optimized_thresholds['duration_min'] = max(
                1.0, user_profile.avg_handwash_duration * 0.3
            )
            optimized_thresholds['duration_max'] = min(
                120.0, user_profile.avg_handwash_duration * 3.0
            )
        
        # 根据运动强度调整
        if user_profile.motion_intensity == 'low':
            optimized_thresholds['motion_intensity'] *= 0.8
        elif user_profile.motion_intensity == 'high':
            optimized_thresholds['motion_intensity'] *= 1.2
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'user_id': user_profile.user_id,
            'thresholds': optimized_thresholds.copy(),
            'performance': recent_performance.copy()
        })
        
        return optimized_thresholds
    
    def _adjust_threshold(self, 
                         base_threshold: float, 
                         sensitivity: float, 
                         precision: float, 
                         recall: float) -> float:
        """调整单个阈值"""
        # 目标是平衡精确率和召回率
        target_precision = 0.85
        target_recall = 0.80
        
        # 计算调整因子
        precision_error = precision - target_precision
        recall_error = recall - target_recall
        
        # 如果精确率低（误报多），提高阈值
        # 如果召回率低（漏报多），降低阈值
        adjustment = -precision_error * 0.1 + recall_error * 0.1
        
        # 考虑用户敏感度偏好
        sensitivity_adjustment = (sensitivity - 0.5) * 0.2
        
        # 应用调整
        new_threshold = base_threshold + adjustment + sensitivity_adjustment
        
        # 限制在合理范围内
        return np.clip(new_threshold, 0.1, 0.95)


class BehaviorPatternAnalyzer:
    """行为模式分析器
    
    分析用户的行为模式，识别个性化特征
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.pattern_labels = ['conservative', 'normal', 'active']
    
    def analyze_motion_patterns(self, motion_sequences: List[np.ndarray]) -> Dict[str, Any]:
        """分析运动模式
        
        Args:
            motion_sequences: 运动序列列表
            
        Returns:
            运动模式分析结果
        """
        if not motion_sequences:
            return {'pattern': 'unknown', 'confidence': 0.0}
        
        try:
            # 提取特征
            features = []
            for sequence in motion_sequences:
                if len(sequence) > 0:
                    feature = self._extract_sequence_features(sequence)
                    features.append(feature)
            
            if not features:
                return {'pattern': 'unknown', 'confidence': 0.0}
            
            features = np.array(features)
            
            # 标准化
            if len(features) > 1:
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = features
            
            # 聚类分析
            if len(features_scaled) >= 3:
                clusters = self.kmeans.fit_predict(features_scaled)
                
                # 分析聚类结果
                pattern_analysis = self._analyze_clusters(features_scaled, clusters)
            else:
                # 数据不足，使用简单分析
                pattern_analysis = self._simple_pattern_analysis(features)
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Motion pattern analysis error: {e}")
            return {'pattern': 'unknown', 'confidence': 0.0}
    
    def _extract_sequence_features(self, sequence: np.ndarray) -> np.ndarray:
        """从序列中提取特征"""
        if len(sequence) == 0:
            return np.zeros(10)
        
        features = []
        
        # 基础统计特征
        features.extend([
            np.mean(sequence),
            np.std(sequence),
            np.max(sequence),
            np.min(sequence),
            np.median(sequence)
        ])
        
        # 变化特征
        if len(sequence) > 1:
            diff = np.diff(sequence)
            features.extend([
                np.mean(np.abs(diff)),
                np.std(diff),
                len(np.where(diff > 0)[0]) / len(diff),  # 上升比例
                len(np.where(diff < 0)[0]) / len(diff),  # 下降比例
                np.sum(np.abs(diff))  # 总变化量
            ])
        else:
            features.extend([0.0] * 5)
        
        return np.array(features)
    
    def _analyze_clusters(self, features: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
        """分析聚类结果"""
        # 计算每个聚类的中心特征
        cluster_centers = []
        for i in range(self.kmeans.n_clusters):
            mask = clusters == i
            if np.any(mask):
                center = np.mean(features[mask], axis=0)
                cluster_centers.append(center)
        
        # 确定主要模式
        main_cluster = np.bincount(clusters).argmax()
        pattern_confidence = np.bincount(clusters)[main_cluster] / len(clusters)
        
        # 映射到模式标签
        if main_cluster < len(self.pattern_labels):
            pattern = self.pattern_labels[main_cluster]
        else:
            pattern = 'unknown'
        
        return {
            'pattern': pattern,
            'confidence': float(pattern_confidence),
            'cluster_distribution': np.bincount(clusters).tolist(),
            'main_cluster': int(main_cluster)
        }
    
    def _simple_pattern_analysis(self, features: np.ndarray) -> Dict[str, Any]:
        """简单模式分析（数据不足时使用）"""
        if len(features) == 0:
            return {'pattern': 'unknown', 'confidence': 0.0}
        
        # 基于特征均值进行简单分类
        mean_features = np.mean(features, axis=0)
        
        # 简单的阈值分类
        if mean_features[0] < 0.3:  # 低活动
            pattern = 'conservative'
        elif mean_features[0] > 0.7:  # 高活动
            pattern = 'active'
        else:
            pattern = 'normal'
        
        return {
            'pattern': pattern,
            'confidence': 0.6,  # 中等置信度
            'mean_features': mean_features.tolist()
        }


class PersonalizationEngine:
    """个性化引擎
    
    集成用户画像、自适应学习和个性化推荐
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化个性化引擎
        
        Args:
            storage_path: 用户数据存储路径
        """
        self.storage_path = Path(storage_path) if storage_path else Path("user_profiles")
        self.storage_path.mkdir(exist_ok=True)
        
        # 用户画像管理
        self.user_profiles: Dict[str, UserProfile] = {}
        self.current_user_id: Optional[str] = None
        
        # 组件
        self.threshold_optimizer = AdaptiveThresholdOptimizer()
        self.pattern_analyzer = BehaviorPatternAnalyzer()
        
        # 性能监控
        self.performance_tracker = defaultdict(list)
        
        # 加载已有用户数据
        self._load_user_profiles()
        
        logger.info(f"PersonalizationEngine initialized with {len(self.user_profiles)} user profiles")
    
    def set_current_user(self, user_id: str) -> UserProfile:
        """设置当前用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户画像
        """
        self.current_user_id = user_id
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
            logger.info(f"Created new user profile for {user_id}")
        
        return self.user_profiles[user_id]
    
    def get_current_user_profile(self) -> Optional[UserProfile]:
        """获取当前用户画像"""
        if self.current_user_id:
            return self.user_profiles.get(self.current_user_id)
        return None
    
    def update_user_behavior(self, 
                           behavior_type: str, 
                           duration: float, 
                           confidence: float,
                           motion_data: Optional[Dict[str, Any]] = None):
        """更新用户行为数据
        
        Args:
            behavior_type: 行为类型
            duration: 持续时间
            confidence: 置信度
            motion_data: 运动数据
        """
        profile = self.get_current_user_profile()
        if not profile:
            logger.warning("No current user set")
            return
        
        # 更新行为统计
        profile.update_behavior_stats(behavior_type, duration, confidence)
        
        # 添加特征向量
        if motion_data:
            feature_vector = self._extract_behavior_features(motion_data, behavior_type)
            profile.feature_vectors.append(feature_vector)
        
        # 分析行为模式
        if len(profile.feature_vectors) >= 10:
            pattern_analysis = self.pattern_analyzer.analyze_motion_patterns(
                list(profile.feature_vectors)[-50:]  # 最近50个样本
            )
            profile.behavior_patterns[behavior_type] = pattern_analysis
        
        # 保存更新
        self._save_user_profile(profile)
    
    def add_user_feedback(self, 
                         feedback_type: str, 
                         behavior_type: str, 
                         confidence: float,
                         context: Optional[Dict[str, Any]] = None):
        """添加用户反馈
        
        Args:
            feedback_type: 反馈类型 ('correct', 'false_positive', 'false_negative')
            behavior_type: 行为类型
            confidence: 检测置信度
            context: 上下文信息
        """
        profile = self.get_current_user_profile()
        if not profile:
            logger.warning("No current user set")
            return
        
        feedback_context = {
            'behavior_type': behavior_type,
            'confidence': confidence,
            'context': context or {}
        }
        
        profile.add_feedback(feedback_type, feedback_context)
        
        # 更新性能跟踪
        self.performance_tracker[profile.user_id].append({
            'timestamp': time.time(),
            'feedback_type': feedback_type,
            'behavior_type': behavior_type,
            'confidence': confidence
        })
        
        # 触发阈值优化
        self._trigger_threshold_optimization(profile)
        
        # 保存更新
        self._save_user_profile(profile)
    
    def get_personalized_thresholds(self, user_id: Optional[str] = None) -> Dict[str, float]:
        """获取个性化阈值
        
        Args:
            user_id: 用户ID，如果为None则使用当前用户
            
        Returns:
            个性化阈值
        """
        if user_id:
            profile = self.user_profiles.get(user_id)
        else:
            profile = self.get_current_user_profile()
        
        if not profile:
            return self.threshold_optimizer.base_thresholds.copy()
        
        # 计算最近性能
        recent_performance = self._calculate_recent_performance(profile)
        
        # 优化阈值
        return self.threshold_optimizer.optimize_thresholds(profile, recent_performance)
    
    def get_user_recommendations(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取用户个性化推荐
        
        Args:
            user_id: 用户ID
            
        Returns:
            个性化推荐
        """
        if user_id:
            profile = self.user_profiles.get(user_id)
        else:
            profile = self.get_current_user_profile()
        
        if not profile:
            return {'recommendations': [], 'confidence': 0.0}
        
        recommendations = []
        
        # 基于行为频率的推荐
        if profile.behavior_counts['handwash'] < profile.behavior_counts['sanitize']:
            recommendations.append({
                'type': 'behavior_balance',
                'message': '建议增加洗手频率，保持手部卫生平衡',
                'priority': 'medium'
            })
        
        # 基于持续时间的推荐
        if profile.avg_handwash_duration < 15.0:
            recommendations.append({
                'type': 'duration_improvement',
                'message': '建议延长洗手时间至15-20秒，确保清洁效果',
                'priority': 'high'
            })
        
        # 基于检测质量的推荐
        false_positive_rate = self._calculate_false_positive_rate(profile)
        if false_positive_rate > 0.3:
            recommendations.append({
                'type': 'detection_optimization',
                'message': '检测到较多误报，建议调整检测敏感度',
                'priority': 'medium'
            })
        
        return {
            'recommendations': recommendations,
            'user_profile_summary': profile.get_profile_summary(),
            'confidence': min(1.0, len(profile.feedback_history) / 50.0)
        }
    
    def _extract_behavior_features(self, motion_data: Dict[str, Any], behavior_type: str) -> np.ndarray:
        """从运动数据中提取行为特征"""
        features = []
        
        # 基础运动特征
        features.extend([
            motion_data.get('avg_speed', 0.0),
            motion_data.get('max_speed', 0.0),
            motion_data.get('position_variance_x', 0.0),
            motion_data.get('position_variance_y', 0.0),
            motion_data.get('trajectory_length', 0.0),
        ])
        
        # 行为特定特征
        if behavior_type == 'handwash':
            features.extend([
                motion_data.get('horizontal_motion_ratio', 0.0),
                motion_data.get('vertical_motion_ratio', 0.0),
                motion_data.get('circular_motion_score', 0.0),
            ])
        elif behavior_type == 'sanitize':
            features.extend([
                motion_data.get('rubbing_intensity', 0.0),
                motion_data.get('hand_proximity', 0.0),
                motion_data.get('synchronization_score', 0.0),
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_recent_performance(self, profile: UserProfile) -> Dict[str, float]:
        """计算最近的性能指标"""
        recent_feedback = profile.feedback_history[-50:]  # 最近50条反馈
        
        if not recent_feedback:
            return {
                'handwash_precision': 0.8,
                'handwash_recall': 0.8,
                'sanitize_precision': 0.8,
                'sanitize_recall': 0.8,
            }
        
        # 按行为类型分组
        handwash_feedback = [f for f in recent_feedback 
                           if f['context'].get('behavior_type') == 'handwash']
        sanitize_feedback = [f for f in recent_feedback 
                           if f['context'].get('behavior_type') == 'sanitize']
        
        # 计算精确率和召回率
        def calculate_metrics(feedback_list):
            if not feedback_list:
                return 0.8, 0.8  # 默认值
            
            correct = len([f for f in feedback_list if f['type'] == 'correct'])
            false_positive = len([f for f in feedback_list if f['type'] == 'false_positive'])
            false_negative = len([f for f in feedback_list if f['type'] == 'false_negative'])
            
            precision = correct / (correct + false_positive) if (correct + false_positive) > 0 else 0.8
            recall = correct / (correct + false_negative) if (correct + false_negative) > 0 else 0.8
            
            return precision, recall
        
        handwash_precision, handwash_recall = calculate_metrics(handwash_feedback)
        sanitize_precision, sanitize_recall = calculate_metrics(sanitize_feedback)
        
        return {
            'handwash_precision': handwash_precision,
            'handwash_recall': handwash_recall,
            'sanitize_precision': sanitize_precision,
            'sanitize_recall': sanitize_recall,
        }
    
    def _calculate_false_positive_rate(self, profile: UserProfile) -> float:
        """计算误报率"""
        recent_feedback = profile.feedback_history[-100:]
        if not recent_feedback:
            return 0.0
        
        false_positives = len([f for f in recent_feedback if f['type'] == 'false_positive'])
        return false_positives / len(recent_feedback)
    
    def _trigger_threshold_optimization(self, profile: UserProfile):
        """触发阈值优化"""
        # 每收集到10条新反馈就触发一次优化
        if len(profile.feedback_history) % 10 == 0:
            recent_performance = self._calculate_recent_performance(profile)
            optimized_thresholds = self.threshold_optimizer.optimize_thresholds(
                profile, recent_performance
            )
            
            # 记录优化历史
            profile.adaptation_history.append({
                'timestamp': time.time(),
                'type': 'threshold_optimization',
                'thresholds': optimized_thresholds,
                'performance': recent_performance
            })
    
    def _load_user_profiles(self):
        """加载用户画像"""
        try:
            for profile_file in self.storage_path.glob("*.json"):
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    profile = UserProfile.from_dict(data)
                    self.user_profiles[profile.user_id] = profile
            
            logger.info(f"Loaded {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.error(f"Failed to load user profiles: {e}")
    
    def _save_user_profile(self, profile: UserProfile):
        """保存用户画像"""
        try:
            profile_file = self.storage_path / f"{profile.user_id}.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user profile {profile.user_id}: {e}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        total_users = len(self.user_profiles)
        total_sessions = sum(p.session_count for p in self.user_profiles.values())
        total_behaviors = sum(
            sum(p.behavior_counts.values()) for p in self.user_profiles.values()
        )
        
        return {
            'total_users': total_users,
            'total_sessions': total_sessions,
            'total_behaviors': total_behaviors,
            'current_user': self.current_user_id,
            'storage_path': str(self.storage_path),
            'optimization_count': len(self.threshold_optimizer.optimization_history),
        }