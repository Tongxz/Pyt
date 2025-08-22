#!/usr/bin/env python3
"""
统一参数配置管理
解决系统中检测参数不一致的问题

作者: AI Assistant
创建时间: 2024
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class HumanDetectionParams:
    """人体检测参数配置"""

    # 模型配置
    model_path: str = "models/yolo/yolov8s.pt"
    device: str = "auto"

    # 检测阈值
    confidence_threshold: float = 0.1  # 降低置信度阈值以提高检测敏感度
    iou_threshold: float = 0.5  # IoU阈值

    # 过滤参数
    min_box_area: int = 500  # 降低最小检测框面积
    max_box_ratio: float = 5.0  # 增加最大宽高比
    min_width: int = 20  # 降低最小宽度
    min_height: int = 30  # 降低最小高度

    # NMS参数
    nms_threshold: float = 0.4  # 非极大值抑制阈值
    max_detections: int = 10  # 最大检测数量


@dataclass
class HairnetDetectionParams:
    """发网检测参数配置"""

    # 模型配置
    model_path: Optional[str] = None
    device: str = "auto"
    confidence_threshold: float = 0.6  # 统一发网检测置信度

    # 图像处理参数
    input_size: tuple = (224, 224)

    # 边缘检测参数
    edge_density_threshold: float = 0.006  # 边缘密度阈值
    sensitive_edge_threshold: float = 0.008  # 敏感边缘阈值

    # 轮廓检测参数
    min_contour_count: int = 2  # 最小轮廓数量
    contour_area_threshold: int = 50  # 轮廓面积阈值

    # 颜色检测参数
    light_blue_ratio_threshold: float = 0.001  # 浅蓝色比例阈值
    light_color_ratio_threshold: float = 0.003  # 浅色比例阈值
    white_ratio_threshold: float = 0.002  # 白色比例阈值

    # 区域检测参数
    upper_region_ratio: float = 0.4  # 上部区域比例
    upper_edge_density_threshold: float = 0.008  # 上部边缘密度阈值

    # 综合评分参数
    total_score_threshold: float = 0.8  # 综合得分阈值
    confidence_boost_factor: float = 1.2  # 置信度提升因子


@dataclass
class BehaviorRecognitionParams:
    """行为识别参数配置"""

    # 基础配置
    confidence_threshold: float = 0.6  # 行为识别置信度阈值
    use_advanced_detection: bool = True
    use_mediapipe: bool = True

    # MediaPipe配置
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # 行为持续时间配置（秒）
    hairnet_min_duration: float = 1.0
    handwashing_min_duration: float = 3.0  # 统一洗手最小时间
    handwashing_max_duration: float = 60.0
    sanitizing_min_duration: float = 2.0  # 统一消毒最小时间
    sanitizing_max_duration: float = 30.0

    # 稳定性检查帧数
    hairnet_stability_frames: int = 5
    handwashing_stability_frames: int = 8  # 提高握手稳定性要求
    sanitizing_stability_frames: int = 5

    # 历史记录配置
    history_maxlen: int = 30  # 行为历史最大长度


@dataclass
class PoseDetectionParams:
    """姿态检测参数配置"""

    backend: str = "yolov8"  # 'yolov8' or 'mediapipe'
    model_path: str = "models/yolo/yolov8n-pose.pt"
    device: str = "auto"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.7


@dataclass
class DetectionRulesParams:
    """规则阈值参数配置"""

    consecutive_frames: int = 5
    horizontal_move_std: float = 0.01
    min_move_frequency_hz: float = 1.0
    iou_threshold: float = 0.5
    ioua_threshold: float = 0.6


@dataclass
class SystemParams:
    """系统级参数配置"""

    # 日志配置
    log_level: str = "INFO"
    debug_mode: bool = False

    # 性能配置
    max_workers: int = 4
    batch_size: int = 1

    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 100
    cache_ttl: int = 300  # 缓存生存时间（秒）

    # API配置
    api_timeout: float = 30.0
    max_request_size: int = 10 * 1024 * 1024  # 10MB


@dataclass
class UnifiedParams:
    """统一参数配置类"""

    human_detection: HumanDetectionParams
    hairnet_detection: HairnetDetectionParams
    behavior_recognition: BehaviorRecognitionParams
    pose_detection: PoseDetectionParams
    detection_rules: DetectionRulesParams
    system: SystemParams

    def __init__(self):
        self.human_detection = HumanDetectionParams()
        self.hairnet_detection = HairnetDetectionParams()
        self.behavior_recognition = BehaviorRecognitionParams()
        self.pose_detection = PoseDetectionParams()
        self.detection_rules = DetectionRulesParams()
        self.system = SystemParams()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "human_detection": asdict(self.human_detection),
            "hairnet_detection": asdict(self.hairnet_detection),
            "behavior_recognition": asdict(self.behavior_recognition),
            "pose_detection": asdict(self.pose_detection),
            "detection_rules": asdict(self.detection_rules),
            "system": asdict(self.system),
        }

    def save_to_yaml(self, file_path: str):
        """保存配置到YAML文件"""
        try:
            config_dict = self.to_dict()

            # 转换元组为列表以避免YAML序列化问题
            def convert_tuples_to_lists(obj):
                if isinstance(obj, dict):
                    return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_tuples_to_lists(item) for item in obj]
                else:
                    return obj

            config_dict = convert_tuples_to_lists(config_dict)

            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    @classmethod
    def load_from_yaml(cls, file_path: str) -> "UnifiedParams":
        """从YAML文件加载配置"""
        instance = cls()

        if not os.path.exists(file_path):
            logger.warning(f"配置文件不存在: {file_path}，使用默认配置")
            return instance

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # 更新配置
            if "human_detection" in config_data:
                for key, value in config_data["human_detection"].items():
                    if hasattr(instance.human_detection, key):
                        setattr(instance.human_detection, key, value)

            if "hairnet_detection" in config_data:
                for key, value in config_data["hairnet_detection"].items():
                    if hasattr(instance.hairnet_detection, key):
                        setattr(instance.hairnet_detection, key, value)

            if "behavior_recognition" in config_data:
                for key, value in config_data["behavior_recognition"].items():
                    if hasattr(instance.behavior_recognition, key):
                        setattr(instance.behavior_recognition, key, value)

            if "pose_detection" in config_data:
                for key, value in config_data["pose_detection"].items():
                    if hasattr(instance.pose_detection, key):
                        setattr(instance.pose_detection, key, value)

            if "detection_rules" in config_data:
                for key, value in config_data["detection_rules"].items():
                    if hasattr(instance.detection_rules, key):
                        setattr(instance.detection_rules, key, value)

            if "system" in config_data:
                for key, value in config_data["system"].items():
                    if hasattr(instance.system, key):
                        setattr(instance.system, key, value)

            logger.info(f"配置已从 {file_path} 加载")

        except Exception as e:
            logger.error(f"加载配置失败: {e}，使用默认配置")

        return instance

    def update_param(self, module: str, param: str, value: Any):
        """更新单个参数"""
        try:
            if module == "human_detection" and hasattr(self.human_detection, param):
                setattr(self.human_detection, param, value)
            elif module == "hairnet_detection" and hasattr(
                self.hairnet_detection, param
            ):
                setattr(self.hairnet_detection, param, value)
            elif module == "behavior_recognition" and hasattr(
                self.behavior_recognition, param
            ):
                setattr(self.behavior_recognition, param, value)
            elif module == "pose_detection" and hasattr(self.pose_detection, param):
                setattr(self.pose_detection, param, value)
            elif module == "detection_rules" and hasattr(self.detection_rules, param):
                setattr(self.detection_rules, param, value)
            elif module == "system" and hasattr(self.system, param):
                setattr(self.system, param, value)
            else:
                raise ValueError(f"未知的模块或参数: {module}.{param}")

            logger.info(f"参数已更新: {module}.{param} = {value}")

        except Exception as e:
            logger.error(f"更新参数失败: {e}")

    def get_param(self, module: str, param: str) -> Any:
        """获取单个参数值"""
        try:
            if module == "human_detection":
                return getattr(self.human_detection, param)
            elif module == "hairnet_detection":
                return getattr(self.hairnet_detection, param)
            elif module == "behavior_recognition":
                return getattr(self.behavior_recognition, param)
            elif module == "pose_detection":
                return getattr(self.pose_detection, param)
            elif module == "detection_rules":
                return getattr(self.detection_rules, param)
            elif module == "system":
                return getattr(self.system, param)
            else:
                raise ValueError(f"未知的模块: {module}")
        except AttributeError:
            raise ValueError(f"未知的参数: {module}.{param}")

    def validate_params(self) -> List[str]:
        """验证参数合理性"""
        warnings = []

        # 验证人体检测参数
        if (
            self.human_detection.confidence_threshold < 0.1
            or self.human_detection.confidence_threshold > 0.9
        ):
            warnings.append("人体检测置信度阈值建议在0.1-0.9之间")

        if self.human_detection.min_box_area < 100:
            warnings.append("最小检测框面积过小，可能导致误检")

        # 验证发网检测参数
        if (
            self.hairnet_detection.confidence_threshold < 0.3
            or self.hairnet_detection.confidence_threshold > 0.9
        ):
            warnings.append("发网检测置信度阈值建议在0.3-0.9之间")

        # 验证行为识别参数
        if self.behavior_recognition.handwashing_min_duration < 1.0:
            warnings.append("洗手最小时间过短，建议至少1秒")

        if (
            self.behavior_recognition.handwashing_min_duration
            > self.behavior_recognition.handwashing_max_duration
        ):
            warnings.append("洗手最小时间不能大于最大时间")

        return warnings


# 全局配置实例
_global_params = None


def get_unified_params() -> UnifiedParams:
    """获取全局统一参数配置"""
    global _global_params
    if _global_params is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "unified_params.yaml",
        )
        _global_params = UnifiedParams.load_from_yaml(config_path)
    return _global_params


def update_global_param(module: str, param: str, value: Any):
    """更新全局参数"""
    params = get_unified_params()
    params.update_param(module, param, value)


def save_global_params():
    """保存全局参数到配置文件"""
    params = get_unified_params()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "unified_params.yaml",
    )
    params.save_to_yaml(config_path)


if __name__ == "__main__":
    # 创建默认配置并保存
    params = UnifiedParams()

    # 验证参数
    warnings = params.validate_params()
    if warnings:
        print("参数验证警告:")
        for warning in warnings:
            print(f"  - {warning}")

    # 保存默认配置
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config"
    )
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, "unified_params.yaml")
    params.save_to_yaml(config_path)

    print(f"统一参数配置已创建: {config_path}")
    print("\n配置摘要:")
    print(f"  人体检测置信度: {params.human_detection.confidence_threshold}")
    print(f"  发网检测置信度: {params.hairnet_detection.confidence_threshold}")
    print(f"  行为识别置信度: {params.behavior_recognition.confidence_threshold}")
    print(f"  姿态检测后端: {params.pose_detection.backend}")
    print(f"  洗手最小时间: {params.behavior_recognition.handwashing_min_duration}秒")
    print(f"  规则检测水平移动标准差阈值: {params.detection_rules.horizontal_move_std}")
