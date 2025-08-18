#!/usr/bin/env python3
"""
人体检测器优化脚本
基于测试结果优化检测器配置
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.detector import HumanDetector
from src.utils.logger import get_logger

# 设置日志
logger = get_logger(__name__, level="INFO")


def create_optimized_detector_config():
    """创建优化的检测器配置"""

    print("=== 人体检测器优化建议 ===")
    print()

    # 基于测试结果的分析
    print("测试结果分析:")
    print("1. YOLOv8n: 检测到7人，但包含低置信度检测(0.15-0.17)")
    print("2. YOLOv8s: 检测到4人，置信度较高(0.88+)，速度适中")
    print("3. YOLOv8m: 检测到3人，置信度最高(0.91+)，但速度较慢")
    print()

    print("优化建议:")
    print("1. 对于实时应用: 推荐 YOLOv8s + 置信度阈值0.3")
    print("2. 对于高精度要求: 推荐 YOLOv8m + 置信度阈值0.5")
    print("3. 对于快速检测: 保持 YOLOv8n + 置信度阈值0.25")
    print()

    return {
        "realtime": {
            "model": "models/yolo/yolov8s.pt",
            "confidence_threshold": 0.3,
            "description": "实时应用平衡配置",
        },
        "high_accuracy": {
            "model": "models/yolo/yolov8m.pt",
            "confidence_threshold": 0.5,
            "description": "高精度配置",
        },
        "fast_detection": {
            "model": "models/yolo/yolov8n.pt",
            "confidence_threshold": 0.25,
            "description": "快速检测配置",
        },
    }


def test_optimized_configs():
    """测试优化配置"""

    configs = create_optimized_detector_config()
    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"

    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return

    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法读取图像: {test_image_path}")
        return

    print("=== 优化配置测试结果 ===")
    print()

    for config_name, config in configs.items():
        print(f"测试配置: {config['description']}")
        print(f"模型: {config['model']}, 置信度阈值: {config['confidence_threshold']}")

        try:
            # 创建检测器
            detector = HumanDetector(model_path=config["model"])
            detector.confidence_threshold = config["confidence_threshold"]

            # 执行检测
            detections = detector.detect(image)

            print(f"检测结果: {len(detections)}人")

            # 计算置信度统计
            if detections:
                confidences = [det["confidence"] for det in detections]
                avg_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)

                print(f"置信度统计: 平均={avg_conf:.3f}, 最小={min_conf:.3f}, 最大={max_conf:.3f}")

                # 高置信度检测数量
                high_conf_count = sum(1 for conf in confidences if conf > 0.7)
                print(f"高置信度检测(>0.7): {high_conf_count}/{len(detections)}")

            print()

        except Exception as e:
            print(f"配置测试失败: {e}")
            print()


def update_detector_default_config():
    """更新检测器默认配置"""

    print("=== 更新检测器默认配置 ===")
    print()

    # 读取当前检测器配置
    detector_file = "src/core/detector.py"

    print(f"当前检测器文件: {detector_file}")
    print("建议的配置更新:")
    print()

    print("1. 将默认模型从 models/yolo/yolov8n.pt 改为 models/yolo/yolov8s.pt")
    print("   - 理由: 更好的检测精度，适中的速度")
    print()

    print("2. 调整置信度阈值从 0.1 到 0.25")
    print("   - 理由: 减少低置信度的误检测")
    print()

    print("3. 优化过滤参数:")
    print("   - min_box_area: 从 500 调整到 800 (过滤更小的误检测)")
    print("   - max_box_ratio: 保持 6.0 (合理的宽高比限制)")
    print()

    # 生成配置代码
    print("建议的代码修改:")
    print("-" * 50)
    print("# 在 HumanDetector.__init__ 中:")
    print(
        'def __init__(self, model_path: str = "models/yolo/yolov8s.pt", device: str = "auto"):'
    )
    print("    # ...")
    print("    self.confidence_threshold = 0.25  # 提高置信度阈值")
    print("    self.iou_threshold = 0.5")
    print("    self.min_box_area = 800   # 提高最小面积阈值")
    print("    self.max_box_ratio = 6.0")
    print("-" * 50)
    print()


def create_adaptive_detector():
    """创建自适应检测器类"""

    print("=== 自适应检测器设计 ===")
    print()

    adaptive_code = '''
class AdaptiveHumanDetector(HumanDetector):
    """自适应人体检测器

    根据应用场景自动选择最佳配置
    """

    def __init__(self, mode: str = "balanced", device: str = "auto"):
        """
        初始化自适应检测器

        Args:
            mode: 检测模式 ('fast', 'balanced', 'accurate')
            device: 计算设备
        """

        # 预定义配置
        configs = {
            'fast': {
                'model': 'models/yolo/yolov8n.pt',
                'confidence': 0.25,
                'min_area': 600,
                'description': '快速检测模式'
            },
            'balanced': {
                'model': 'models/yolo/yolov8s.pt',
                'confidence': 0.3,
                'min_area': 800,
                'description': '平衡模式(推荐)'
            },
            'accurate': {
                'model': 'models/yolo/yolov8m.pt',
                'confidence': 0.5,
                'min_area': 1000,
                'description': '高精度模式'
            }
        }

        if mode not in configs:
            raise ValueError(f"不支持的模式: {mode}. 支持的模式: {list(configs.keys())}")

        config = configs[mode]

        # 初始化基类
        super().__init__(model_path=config['model'], device=device)

        # 应用配置
        self.confidence_threshold = config['confidence']
        self.min_box_area = config['min_area']
        self.mode = mode
        self.config_description = config['description']

        logger.info(f"自适应检测器初始化: {self.config_description}")

    def get_mode_info(self) -> dict:
        """获取当前模式信息"""
        return {
            'mode': self.mode,
            'description': self.config_description,
            'model_path': self.model.model_path if hasattr(self.model, 'model_path') else 'unknown',
            'confidence_threshold': self.confidence_threshold,
            'min_box_area': self.min_box_area
        }
'''

    print("自适应检测器代码:")
    print(adaptive_code)

    # 保存到文件
    output_file = "src/core/adaptive_detector.py"
    print(f"\n建议将此代码保存到: {output_file}")

    return adaptive_code


if __name__ == "__main__":
    print("人体检测器优化分析")
    print("=" * 80)

    # 创建优化配置
    create_optimized_detector_config()

    # 测试优化配置
    test_optimized_configs()

    # 更新建议
    update_detector_default_config()

    # 自适应检测器设计
    create_adaptive_detector()

    print("优化分析完成！")
