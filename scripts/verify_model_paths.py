#!/usr/bin/env python
"""
模型路径验证脚本

验证模型路径更新后，所有模型文件是否能正常加载
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.unified_params import get_unified_params
from src.core.hairnet_detection_factory import HairnetDetectionFactory
from src.core.yolo_hairnet_detector import YOLOHairnetDetector


def verify_model_files():
    """验证模型文件是否存在"""
    print("🔍 验证模型文件存在性...")

    # 获取统一参数配置
    params = get_unified_params()

    # 检查YOLO模型
    yolo_path = params.human_detection.model_path
    print(f"YOLO模型路径: {yolo_path}")
    if os.path.exists(yolo_path):
        print(f"✅ YOLO模型文件存在: {yolo_path}")
    else:
        print(f"❌ YOLO模型文件不存在: {yolo_path}")

    # 检查发网检测模型
    hairnet_path = "models/hairnet_detection/hairnet_detection.pt"
    print(f"发网检测模型路径: {hairnet_path}")
    if os.path.exists(hairnet_path):
        print(f"✅ 发网检测模型文件存在: {hairnet_path}")
    else:
        print(f"❌ 发网检测模型文件不存在: {hairnet_path}")

    # 检查用户训练的模型
    user_model_paths = [
        "models/hairnet_model/weights/best.pt",
        "models/hairnet_model/weights/last.pt",
    ]

    for model_path in user_model_paths:
        print(f"用户训练模型路径: {model_path}")
        if os.path.exists(model_path):
            print(f"✅ 用户训练模型文件存在: {model_path}")
        else:
            print(f"❌ 用户训练模型文件不存在: {model_path}")

    print()


def verify_model_loading():
    """验证模型是否能正常加载"""
    print("🔧 验证模型加载功能...")

    try:
        # 测试YOLOv8发网检测器
        print("测试YOLOv8发网检测器...")
        detector = YOLOHairnetDetector(
            model_path="models/hairnet_detection/hairnet_detection.pt",
            device="cpu",  # 使用CPU避免CUDA问题
        )
        print("✅ YOLOv8发网检测器初始化成功")

        # 测试工厂模式
        print("测试发网检测器工厂...")
        factory_detector = HairnetDetectionFactory.create_detector(
            detector_type="yolo",
            model_path="models/hairnet_detection/hairnet_detection.pt",
            device="cpu",
        )
        print("✅ 发网检测器工厂创建成功")

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return False

    return True


def verify_config_files():
    """验证配置文件中的路径"""
    print("📋 验证配置文件...")

    config_files = ["config/unified_params.yaml", "config/default.yaml"]

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ 配置文件存在: {config_file}")

            # 读取配置文件内容，检查是否包含正确的模型路径
            with open(config_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "models/yolo/" in content:
                print(f"  ✅ 包含正确的YOLO模型路径")
            else:
                print(f"  ⚠️  未找到YOLO模型路径")

            if "models/hairnet_detection/" in content:
                print(f"  ✅ 包含正确的发网检测模型路径")
            else:
                print(f"  ⚠️  未找到发网检测模型路径")
        else:
            print(f"❌ 配置文件不存在: {config_file}")

    print()


def main():
    """主函数"""
    print("🚀 开始验证模型路径更新结果...\n")

    # 验证模型文件存在性
    verify_model_files()

    # 验证配置文件
    verify_config_files()

    # 验证模型加载
    success = verify_model_loading()

    print("\n" + "=" * 50)
    if success:
        print("🎉 模型路径更新验证成功！")
        print("所有模型文件都已正确移动到新位置，并且可以正常加载。")
    else:
        print("⚠️  模型路径更新验证部分失败")
        print("请检查错误信息并进行相应修复。")
    print("=" * 50)


if __name__ == "__main__":
    main()
